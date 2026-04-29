import * as vscode from "vscode";
import type { HFModelItem, RetryConfig } from "./types";
import { OpenAIFunctionToolDef } from "./openai/openaiTypes";

import { logger } from "./logger";

const RETRY_MAX_ATTEMPTS = 3;
const RETRY_INTERVAL_MS = 1000;
const RETRY_BACKOFF_FACTOR = 2;
const RETRY_MAX_INTERVAL_MS = 60000;

/** Sliding-window duration for rate limiting (1 minute in milliseconds). */
const RATE_LIMIT_WINDOW_MS = 60_000;
/** Buffer added to avoid boundary race when computing throttle wait time. */
const BOUNDARY_BUFFER_MS = 1;

/**
 * Parse the value of a `Retry-After` HTTP response header into milliseconds.
 * Accepts both delay-seconds (integer) and HTTP-date formats.
 * Returns undefined when the header value cannot be parsed.
 */
function throwIfCancellationRequested(token?: vscode.CancellationToken): void {
	if (token?.isCancellationRequested) {
		throw new vscode.CancellationError();
	}
}

export function sleep(ms: number, token?: vscode.CancellationToken): Promise<void> {
	throwIfCancellationRequested(token);
	if (ms <= 0) {
		return Promise.resolve();
	}
	return new Promise<void>((resolve, reject) => {
		let disposable: vscode.Disposable | undefined;
		const timeout = setTimeout(() => {
			disposable?.dispose();
			resolve();
		}, ms);
		disposable = token?.onCancellationRequested(() => {
			clearTimeout(timeout);
			disposable?.dispose();
			reject(new vscode.CancellationError());
		});
	});
}

export async function fetchWithCancellation(
	input: RequestInfo | URL,
	init: RequestInit,
	token?: vscode.CancellationToken
): Promise<Response> {
	throwIfCancellationRequested(token);
	if (!token) {
		return fetch(input, init);
	}
	const controller = new AbortController();
	const disposable = token.onCancellationRequested(() => controller.abort());
	try {
		return await fetch(input, { ...init, signal: controller.signal });
	} finally {
		disposable.dispose();
	}
}

export function parseRetryAfterMs(retryAfterHeader: string): number | undefined {
	const trimmed = retryAfterHeader.trim();
	if (!trimmed) {
		return undefined;
	}

	// Try numeric seconds first (most common for rate-limit responses)
	const seconds = parseFloat(trimmed);
	if (!isNaN(seconds) && seconds >= 0) {
		return Math.ceil(seconds * 1000);
	}

	// Try HTTP-date format
	const date = new Date(trimmed);
	if (!isNaN(date.getTime())) {
		const delayMs = date.getTime() - Date.now();
		return delayMs > 0 ? delayMs : 0;
	}

	return undefined;
}

/**
 * Sliding-window rate limiter.
 * Call `throttle(maxRpm)` before each request; it will await until the request
 * can be sent without exceeding `maxRpm` requests per minute.
 */
export class RequestRateLimiter {
	private _timestamps: number[] = [];

	/**
	 * Wait until sending the next request stays within the given RPM limit.
	 * @param maxRequestsPerMinute Maximum number of requests allowed per 60-second window.
	 */
	async throttle(maxRequestsPerMinute: number, token?: vscode.CancellationToken): Promise<void> {
		if (maxRequestsPerMinute <= 0) {
			return;
		}

		const windowMs = RATE_LIMIT_WINDOW_MS;

		while (true) {
			throwIfCancellationRequested(token);
			const now = Date.now();
			// Evict timestamps older than the sliding window
			this._timestamps = this._timestamps.filter((t) => t > now - windowMs);

			if (this._timestamps.length < maxRequestsPerMinute) {
				// Under the limit – record this request and proceed
				this._timestamps.push(now);
				return;
			}

			// Over the limit – wait until the oldest timestamp falls outside the window
			const oldest = this._timestamps[0];
			const waitMs = Math.max(0, oldest + windowMs - now + BOUNDARY_BUFFER_MS);
			logger.warn("rateLimiter.throttle", {
				maxRequestsPerMinute,
				currentCount: this._timestamps.length,
				waitMs,
			});
			await sleep(waitMs, token);
		}
	}
}

// HTTP status codes that should trigger a retry
const RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504];

// Network error patterns to retry
const networkErrorPatterns = [
	"fetch failed",
	"ECONNRESET",
	"ETIMEDOUT",
	"ENOTFOUND",
	"ECONNREFUSED",
	"timeout",
	"TIMEOUT",
	"network error",
	"NetworkError",
];

// Model ID parsing helper
export interface ParsedModelId {
	baseId: string;
	configId?: string;
}

export function getModelProviderId(model: unknown): string {
	if (!model || typeof model !== "object") {
		return "";
	}
	const obj = model as Record<string, unknown>;
	const pick = (v: unknown): string => (typeof v === "string" ? v.trim() : "");
	return (
		pick(obj.owned_by) ||
		pick(obj.provide) ||
		pick(obj.provider) ||
		pick(obj.ownedBy) ||
		pick(obj.owner) ||
		pick(obj.vendor)
	);
}

export function normalizeUserModels(models: unknown): HFModelItem[] {
	const list = Array.isArray(models) ? models : [];
	const out: HFModelItem[] = [];
	for (const item of list) {
		if (!item || typeof item !== "object") {
			continue;
		}
		const provider = getModelProviderId(item);
		out.push({ ...(item as HFModelItem), owned_by: provider });
	}
	return out;
}

/**
 * Parse a model ID that may contain a configuration ID separator.
 * Format: "baseId::configId" or just "baseId"
 */
export function parseModelId(modelId: string): ParsedModelId {
	const parts = modelId.split("::");
	if (parts.length >= 2) {
		return {
			baseId: parts[0],
			configId: parts.slice(1).join("::"), // In case configId itself contains '::'
		};
	}
	return {
		baseId: modelId,
	};
}

/**
 * Map VS Code message role to OpenAI message role string.
 * @param message The message whose role is mapped.
 */
export function mapRole(message: vscode.LanguageModelChatRequestMessage): "user" | "assistant" | "system" {
	const USER = vscode.LanguageModelChatMessageRole.User as unknown as number;
	const ASSISTANT = vscode.LanguageModelChatMessageRole.Assistant as unknown as number;
	const r = message.role as unknown as number;
	if (r === USER) {
		return "user";
	}
	if (r === ASSISTANT) {
		return "assistant";
	}
	return "system";
}

/**
 * Convert VS Code tool definitions to OpenAI function tool definitions.
 * @param options Request options containing tools and toolMode.
 */
export function convertToolsToOpenAI(options?: vscode.ProvideLanguageModelChatResponseOptions): {
	tools?: OpenAIFunctionToolDef[];
	tool_choice?: "auto" | { type: "function"; function: { name: string } };
} {
	const tools = options?.tools ?? [];
	if (!tools || tools.length === 0) {
		return {};
	}

	const toolDefs: OpenAIFunctionToolDef[] = tools
		.filter((t) => t && typeof t === "object")
		.map((t) => {
			const name = t.name;
			const description = typeof t.description === "string" ? t.description : "";
			const params = t.inputSchema ?? { type: "object", properties: {} };
			return {
				type: "function" as const,
				function: {
					name,
					description,
					parameters: params,
				},
			} satisfies OpenAIFunctionToolDef;
		});

	let tool_choice: "auto" | { type: "function"; function: { name: string } } = "auto";
	if (options?.toolMode === vscode.LanguageModelChatToolMode.Required) {
		if (tools.length !== 1) {
			console.error("[OAI Compatible Model Provider] ToolMode.Required but multiple tools:", tools.length);
			throw new Error("LanguageModelChatToolMode.Required is not supported with more than one tool");
		}
		tool_choice = { type: "function", function: { name: tools[0].name } };
	}

	return { tools: toolDefs, tool_choice };
}

export interface OpenAIResponsesFunctionToolDef {
	type: "function";
	name: string;
	description?: string;
	parameters?: object;
}

export type OpenAIResponsesToolChoice = "auto" | { type: "function"; name: string };

/**
 * Convert VS Code tool definitions to OpenAI Responses API tool definitions.
 * Responses uses `{ type:"function", name, description, parameters }` (no nested `function` object).
 */
export function convertToolsToOpenAIResponses(options?: vscode.ProvideLanguageModelChatResponseOptions): {
	tools?: OpenAIResponsesFunctionToolDef[];
	tool_choice?: OpenAIResponsesToolChoice;
} {
	const toolConfig = convertToolsToOpenAI(options);
	if (!toolConfig.tools || toolConfig.tools.length === 0) {
		return {};
	}

	const tools: OpenAIResponsesFunctionToolDef[] = toolConfig.tools.map((t) => {
		const out: OpenAIResponsesFunctionToolDef = {
			type: "function",
			name: t.function.name,
		};
		if (t.function.description) {
			out.description = t.function.description;
		}
		if (t.function.parameters) {
			out.parameters = t.function.parameters;
		}
		return out;
	});

	let tool_choice: OpenAIResponsesToolChoice | undefined;
	if (toolConfig.tool_choice === "auto") {
		tool_choice = "auto";
	} else if (toolConfig.tool_choice?.type === "function") {
		tool_choice = { type: "function", name: toolConfig.tool_choice.function.name };
	}

	return { tools, tool_choice };
}

/**
 * 检查是否为图片MIME类型
 */
export function isImageMimeType(mimeType: string): boolean {
	return mimeType.startsWith("image/") && ["image/jpeg", "image/png", "image/gif", "image/webp"].includes(mimeType);
}

/**
 * 创建图片的data URL
 */
export function createDataUrl(dataPart: vscode.LanguageModelDataPart): string {
	const base64Data = Buffer.from(dataPart.data).toString("base64");
	return `data:${dataPart.mimeType};base64,${base64Data}`;
}

/**
 * Type guard for LanguageModelToolResultPart-like values.
 * @param value Unknown value to test.
 */
export function isToolResultPart(value: unknown): value is { callId: string; content?: ReadonlyArray<unknown> } {
	if (!value || typeof value !== "object") {
		return false;
	}
	const obj = value as Record<string, unknown>;
	const hasCallId = typeof obj.callId === "string";
	const hasContent = "content" in obj;
	return hasCallId && hasContent;
}

/**
 * Concatenate tool result content into a single text string.
 * @param pr Tool result-like object with content array.
 */
export function collectToolResultText(pr: { content?: ReadonlyArray<unknown> }): string {
	let text = "";
	for (const c of pr.content ?? []) {
		if (c instanceof vscode.LanguageModelTextPart) {
			text += c.value;
		} else if (typeof c === "string") {
			text += c;
		} else if (c instanceof vscode.LanguageModelDataPart && c.mimeType === "cache_control") {
			/* ignore */
		} else {
			try {
				text += JSON.stringify(c);
			} catch {
				/* ignore */
			}
		}
	}
	return text;
}

/**
 * Try to parse a JSON object from a string.
 * @param text The input string.
 * @returns Parsed object or ok:false.
 */
export function tryParseJSONObject(text: string): { ok: true; value: Record<string, unknown> } | { ok: false } {
	try {
		if (!text || !/[{]/.test(text)) {
			return { ok: false };
		}
		const value = JSON.parse(text);
		if (value && typeof value === "object" && !Array.isArray(value)) {
			return { ok: true, value };
		}
		return { ok: false };
	} catch {
		return { ok: false };
	}
}

/**
 * Create retry configuration from VS Code workspace settings.
 * @returns Retry configuration with default values.
 */
export function createRetryConfig(): RetryConfig {
	const config = vscode.workspace.getConfiguration();
	const retryConfig = config.get<RetryConfig>("oaicopilot.retry", {
		enabled: true,
		max_attempts: RETRY_MAX_ATTEMPTS,
		interval_ms: RETRY_INTERVAL_MS,
	});

	return {
		enabled: retryConfig.enabled ?? true,
		max_attempts: retryConfig.max_attempts ?? RETRY_MAX_ATTEMPTS,
		interval_ms: retryConfig.interval_ms ?? RETRY_INTERVAL_MS,
		status_codes: retryConfig.status_codes,
	};
}

/**
 * Execute a function with retry logic for rate limiting.
 * @param fn The async function to execute
 * @param retryConfig Retry configuration
 * @param token Cancellation token
 * @returns Result of the function execution
 */
export async function executeWithRetry<T>(
	fn: () => Promise<T>,
	retryConfig: RetryConfig,
	token?: vscode.CancellationToken
): Promise<T> {
	throwIfCancellationRequested(token);
	if (!retryConfig.enabled) {
		return await fn();
	}

	const maxAttempts = retryConfig.max_attempts ?? RETRY_MAX_ATTEMPTS;
	const baseIntervalMs = retryConfig.interval_ms ?? RETRY_INTERVAL_MS;
	// Merge user-configured status codes with default ones, removing duplicates
	const retryableStatusCodes = retryConfig.status_codes
		? [...new Set([...RETRYABLE_STATUS_CODES, ...retryConfig.status_codes])]
		: RETRYABLE_STATUS_CODES;
	let lastError: Error | undefined;

	for (let attempt = 0; attempt < maxAttempts; attempt++) {
		throwIfCancellationRequested(token);
		try {
			return await fn();
		} catch (error) {
			lastError = error instanceof Error ? error : new Error(String(error));

			// Check if error is retryable based on status codes
			const isRetryableStatusError = retryableStatusCodes.some((code) => lastError?.message.includes(`[${code}]`));
			// Check if error is retryable based on network error patterns
			const isRetryableNetworkError = networkErrorPatterns.some((pattern) => lastError?.message.includes(pattern));
			const isRetryableError = isRetryableStatusError || isRetryableNetworkError;

			if (!isRetryableError || attempt === maxAttempts - 1) {
				throw lastError;
			}

			// Honor the Retry-After header value when present (attached by the caller)
			const retryAfterMs = (lastError as { retryAfterMs?: number }).retryAfterMs;

			// Exponential backoff: interval doubles each attempt, capped at 60s
			const backoffMs = Math.min(baseIntervalMs * Math.pow(RETRY_BACKOFF_FACTOR, attempt), RETRY_MAX_INTERVAL_MS);
			const delayMs = retryAfterMs !== undefined ? Math.max(retryAfterMs, backoffMs) : backoffMs;

			logger.warn("retry.attempt", {
				attempt: attempt + 1,
				maxAttempts,
				delayMs,
				retryAfterMs,
				errorName: lastError.name,
				errorMessage: lastError.message,
			});

			console.error(
				`[OAI Compatible Model Provider] Retryable error detected, retrying in ${delayMs}ms (attempt ${attempt + 1}/${maxAttempts}). Error:`,
				lastError instanceof Error ? { name: lastError.name, message: lastError.message } : String(lastError)
			);

			// Wait for the calculated interval before retrying
			await sleep(delayMs, token);
		}
	}

	// This should never be reached, but TypeScript needs it
	logger.error("retry.exhausted", {
		maxAttempts,
		lastError: lastError ? { name: lastError.name, message: lastError.message } : String(lastError),
	});
	throw lastError || new Error("Retry failed");
}

/**
 * Thinking level values exposed in the model picker.
 * "none" disables thinking; "low"–"max" enable it with increasing budget/effort.
 */
export type ThinkingLevel = "none" | "low" | "medium" | "high" | "max";

/** Maps a thinking level to an effort string for reasoning_effort-style APIs. */
const THINKING_LEVEL_TO_EFFORT: Record<ThinkingLevel, string> = {
	none: "low",
	low: "low",
	medium: "medium",
	high: "high",
	max: "max",
};

/** Maps a thinking level to a token budget for thinking_budget-style APIs. */
const THINKING_LEVEL_TO_BUDGET: Record<ThinkingLevel, number> = {
	none: 0,
	low: 1024,
	medium: 8192,
	high: 32768,
	max: 131072,
};

/**
 * Convert a thinking level string to a reasoning effort string.
 * Falls back to the raw level value if not recognized.
 */
export function thinkingLevelToEffort(level: string): string {
	return THINKING_LEVEL_TO_EFFORT[level as ThinkingLevel] ?? level;
}

/**
 * Convert a thinking level string to a token budget number.
 */
export function thinkingLevelToBudget(level: string): number {
	return THINKING_LEVEL_TO_BUDGET[level as ThinkingLevel] ?? 8192;
}

/**
 * Derive the closest ThinkingLevel label from a token budget number.
 */
export function budgetToThinkingLevel(budget: number | undefined): ThinkingLevel {
	if (budget === undefined || budget <= 0) {
		return "none";
	}
	// Midpoints between THINKING_LEVEL_TO_BUDGET entries: low=1024, medium=8192, high=32768, max=131072
	if (budget <= 4608) {
		return "low";
	}
	if (budget <= 20480) {
		return "medium";
	}
	if (budget <= 81920) {
		return "high";
	}
	return "max";
}