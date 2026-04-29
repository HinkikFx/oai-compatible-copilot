import * as vscode from "vscode";
import {
	CancellationToken,
	LanguageModelChatInformation,
	LanguageModelChatProvider,
	LanguageModelChatRequestMessage,
	ProvideLanguageModelChatResponseOptions,
	LanguageModelResponsePart2,
	Progress,
} from "vscode";

import type { ApiUsage, HFModelItem } from "./types";

import type { OllamaRequestBody } from "./ollama/ollamaTypes";

import {
	parseModelId,
	createRetryConfig,
	executeWithRetry,
	normalizeUserModels,
	parseRetryAfterMs,
	RequestRateLimiter,
	sleep,
	fetchWithCancellation,
} from "./utils";

import { prepareLanguageModelChatInformation } from "./provideModel";
import { countMessageTokens, countToolTokens } from "./provideToken";
import { updateAgentUsageStatusBar, updateContextStatusBar } from "./statusBar";
import { OllamaApi } from "./ollama/ollamaApi";
import { OpenaiApi } from "./openai/openaiApi";
import { OpenaiResponsesApi } from "./openai/openaiResponsesApi";
import { AnthropicApi } from "./anthropic/anthropicApi";
import { AnthropicRequestBody } from "./anthropic/anthropicTypes";
import { GeminiApi, buildGeminiGenerateContentUrl, type GeminiToolCallMeta } from "./gemini/geminiApi";
import type { GeminiGenerateContentRequest } from "./gemini/geminiTypes";
import { CommonApi } from "./commonApi";
import { logger } from "./logger";
import { usageTracker } from "./usageTracker";

/**
 * VS Code Chat provider backed by Hugging Face Inference Providers.
 */
export class HuggingFaceChatModelProvider implements LanguageModelChatProvider {
	/** Track last request completion time for delay calculation. */
	private _lastRequestTime: number | null = null;

	/** Per-model-key rate limiters (keyed by model id or configId). */
	private readonly _rateLimiters = new Map<string, RequestRateLimiter>();

	private readonly _geminiToolCallMetaByCallId = new Map<string, GeminiToolCallMeta>();
	private readonly _openaiResponsesPreviousResponseIdUnsupportedBaseUrls = new Set<string>();
	private _sessionPromptTokens = 0;
	private _sessionCompletionTokens = 0;
	private _sessionCostUsd = 0;

	static readonly OPENAI_RESPONSES_STATEFUL_MARKER_MIME = "application/vnd.oaicopilot.stateful-marker";

	/**
	 * Create a provider using the given secret storage for the API key.
	 * @param secrets VS Code secret storage.
	 */
	constructor(
		private readonly secrets: vscode.SecretStorage,
		private readonly statusBarItem: vscode.StatusBarItem
	) {}

	/**
	 * Get the list of available language models contributed by this provider
	 * @param options Options which specify the calling context of this function
	 * @param token A cancellation token which signals if the user cancelled the request or not
	 * @returns A promise that resolves to the list of available language models
	 */
	async provideLanguageModelChatInformation(
		options: vscode.PrepareLanguageModelChatModelOptions,
		_token: CancellationToken
	): Promise<LanguageModelChatInformation[]> {
		// VS Code may pass `silent: true` at runtime for background refresh (pre-v5 behaviour).
		const opts = options as unknown as Record<string, unknown>;
		const silent = typeof opts.silent === "boolean" ? opts.silent : true;
		return prepareLanguageModelChatInformation({ silent }, _token, this.secrets);
	}

	/**
	 * Returns the number of tokens for a given text using the model specific tokenizer logic
	 * @param model The language model to use
	 * @param text The text to count tokens for
	 * @param token A cancellation token for the request
	 * @returns A promise that resolves to the number of tokens
	 */
	async provideTokenCount(
		model: LanguageModelChatInformation,
		text: string | LanguageModelChatRequestMessage,
		_token: CancellationToken
	): Promise<number> {
		const config = vscode.workspace.getConfiguration();
		const userModels = normalizeUserModels(config.get<unknown>("oaicopilot.models", []));
		const parsedModelId = parseModelId(model.id);
		const um =
			userModels.find(
				(um) =>
					um.id === parsedModelId.baseId &&
					((parsedModelId.configId && um.configId === parsedModelId.configId) ||
						(!parsedModelId.configId && !um.configId))
			) ?? userModels.find((um) => um.id === parsedModelId.baseId);
		return countMessageTokens(text, {
			includeReasoningInRequest: um?.include_reasoning_in_request ?? true,
			tokenizerMode: um?.apiMode ?? "openai",
			modelId: parsedModelId.baseId,
		});
	}

	/**
	 * Returns the response for a chat request, passing the results to the progress callback.
	 * The {@linkcode LanguageModelChatProvider} must emit the response parts to the progress callback as they are received from the language model.
	 * @param model The language model to use
	 * @param messages The messages to include in the request
	 * @param options Options for the request
	 * @param progress The progress to emit the streamed response chunks to
	 * @param token A cancellation token for the request
	 * @returns A promise that resolves when the response is complete. Results are actually passed to the progress callback.
	 */
	async provideLanguageModelChatResponse(
		model: LanguageModelChatInformation,
		messages: readonly LanguageModelChatRequestMessage[],
		options: ProvideLanguageModelChatResponseOptions,
		progress: Progress<LanguageModelResponsePart2>,
		token: CancellationToken
	): Promise<void> {
		const requestUsage = {
			promptTokens: 0,
			completionTokens: 0,
			reasoningTokens: 0,
			toolCallTokens: 0,
		};
		const estimateTokens = (text: string): number => Math.max(0, Math.ceil(text.length / 4));
		const trackingProgress: Progress<LanguageModelResponsePart2> = {
			report: (part) => {
				try {
					if (part instanceof vscode.LanguageModelTextPart) {
						requestUsage.completionTokens += estimateTokens(part.value);
					} else if (part instanceof vscode.LanguageModelThinkingPart) {
						const value = Array.isArray(part.value) ? part.value.join("") : part.value;
						const tokens = estimateTokens(value);
						requestUsage.reasoningTokens += tokens;
						requestUsage.completionTokens += tokens;
					} else if (part instanceof vscode.LanguageModelToolCallPart) {
						let serialized = part.name;
						try {
							serialized += JSON.stringify(part.input ?? {});
						} catch {
							/* ignore */
						}
						const tokens = estimateTokens(serialized);
						requestUsage.toolCallTokens += tokens;
						requestUsage.completionTokens += tokens;
					}
					progress.report(part);
				} catch (e) {
					console.error("[OAI Compatible Model Provider] Progress.report failed", {
						modelId: model.id,
						error: e instanceof Error ? { name: e.name, message: e.message } : String(e),
					});
				}
			},
		};
		const requestStartTime = Date.now();
		let inputPricePerMillion: number | undefined;
		let outputPricePerMillion: number | undefined;
		let apiUsage: ApiUsage | undefined;
		let apiModeForUsage = "unknown";
		try {
			// get model config from user settings
			const config = vscode.workspace.getConfiguration();
			const userModels = normalizeUserModels(config.get<unknown>("oaicopilot.models", []));

			// Parse model ID to handle config ID
			const parsedModelId = parseModelId(model.id);

			// Find matching user model configuration
			// Prioritize matching models with same base ID and config ID
			// If no config ID, match models with same base ID
			let um: HFModelItem | undefined = userModels.find(
				(um) =>
					um.id === parsedModelId.baseId &&
					((parsedModelId.configId && um.configId === parsedModelId.configId) ||
						(!parsedModelId.configId && !um.configId))
			);

			// If still no model found, try to find any model matching the base ID (most lenient match, for backward compatibility)
			if (!um) {
				um = userModels.find((um) => um.id === parsedModelId.baseId);
			}

			// Check if using Ollama native API mode
			const apiMode = um?.apiMode ?? "openai";
			apiModeForUsage = apiMode;
			const baseUrl = um?.baseUrl || config.get<string>("oaicopilot.baseUrl", "");

			logger.info("request.start", {
				modelId: model.id,
				messageCount: messages.length,
				apiMode,
				baseUrl,
			});

			// Prepare model configuration
			const modelConfig = {
				includeReasoningInRequest: um?.include_reasoning_in_request ?? false,
				tokenizerMode: apiMode,
				modelId: parsedModelId.baseId,
			};
			inputPricePerMillion = um?.inputPricePerMillionTokens ?? um?.input_price_per_million_tokens;
			outputPricePerMillion = um?.outputPricePerMillionTokens ?? um?.output_price_per_million_tokens;
			const promptTokenCounts = await Promise.all(
				messages.map((message) => countMessageTokens(message, modelConfig).catch(() => 0))
			);
			const toolDefinitionTokens = options.tools && options.tools.length > 0 ? await countToolTokens(options.tools).catch(() => 0) : 0;
			requestUsage.promptTokens = promptTokenCounts.reduce((sum, count) => sum + count, 0) + toolDefinitionTokens;

			// Update Token Usage
			void updateContextStatusBar(messages, options.tools, model, this.statusBarItem, modelConfig).catch((error) => {
				logger.warn("token.status.update.failed", {
					modelId: model.id,
					error: error instanceof Error ? error.message : String(error),
				});
			});

			// Apply delay between consecutive requests
			const modelDelay = um?.delay;
			const globalDelay = config.get<number>("oaicopilot.delay", 0);
			const delayMs = modelDelay !== undefined ? modelDelay : globalDelay;

			if (delayMs > 0 && this._lastRequestTime !== null) {
				const elapsed = Date.now() - this._lastRequestTime;
				if (elapsed < delayMs) {
					const remainingDelay = delayMs - elapsed;
					logger.debug("request.delay", {
						delayMs,
						elapsed,
						remainingDelay,
					});
					await sleep(remainingDelay, token);
				}
			}

			// Apply per-minute rate limiting (proactive throttling to avoid 429s)
			const modelMaxRpm = um?.maxRequestsPerMinute;
			const globalMaxRpm = config.get<number>("oaicopilot.maxRequestsPerMinute", 0);
			const maxRpm = modelMaxRpm !== undefined ? modelMaxRpm : globalMaxRpm;
			if (maxRpm > 0) {
				// Use the full model.id (which may include ::configId suffix) so different
				// configurations of the same base model get independent rate limiters.
				const rateLimiterKey = model.id;
				if (!this._rateLimiters.has(rateLimiterKey)) {
					this._rateLimiters.set(rateLimiterKey, new RequestRateLimiter());
				}
				const rateLimiter = this._rateLimiters.get(rateLimiterKey)!;
				logger.debug("rateLimiter.check", { modelId: model.id, maxRpm });
				await rateLimiter.throttle(maxRpm, token);
			}

			// Get API key for the model's provider
			const provider = um?.owned_by;
			const useGenericKey = !um?.baseUrl;
			const modelApiKey = await this.ensureApiKey(useGenericKey, provider);
			if (!modelApiKey) {
				logger.warn("apiKey.missing", {
					provider: provider ?? "",
					useGenericKey,
				});
				throw new Error("OAI Compatible API key not found");
			}

			// send chat request
			const BASE_URL = baseUrl;
			if (!BASE_URL || !BASE_URL.startsWith("http")) {
				throw new Error(`Invalid base URL configuration.`);
			}

			// get retry config
			const retryConfig = createRetryConfig();

			// prepare headers with custom headers if specified
			const requestHeaders = CommonApi.prepareHeaders(modelApiKey, apiMode, um?.headers);
			logger.debug("request.headers", {
				headers: logger.sanitizeHeaders(requestHeaders as Record<string, string>),
			});
			logger.debug("request.messages.origin", {
				messages: messages,
			});
			if (apiMode === "ollama") {
				// Ollama native API mode
				const ollamaApi = new OllamaApi(model.id);
				const ollamaMessages = ollamaApi.convertMessages(messages, modelConfig);

				let ollamaRequestBody: OllamaRequestBody = {
					model: parsedModelId.baseId,
					messages: ollamaMessages,
					stream: true,
				};
				ollamaRequestBody = ollamaApi.prepareRequestBody(ollamaRequestBody, um, options);

				// send Ollama chat request with retry
				const url = `${BASE_URL.replace(/\/+$/, "")}/api/chat`;
				logger.debug("request.body", {
					url: url,
					requestBody: ollamaRequestBody,
				});
				const response = await executeWithRetry(async () => {
					const res = await fetchWithCancellation(url, {
						method: "POST",
						headers: requestHeaders,
						body: JSON.stringify(ollamaRequestBody),
					});

					if (!res.ok) {
						const errorText = await res.text();
						console.error("[Ollama Provider] Ollama API error response", errorText);
						const error = new Error(
							`Ollama API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`
						);
						const retryAfterMs = parseRetryAfterMs(res.headers.get("Retry-After") ?? "");
						if (retryAfterMs !== undefined) {
							(error as { retryAfterMs?: number }).retryAfterMs = retryAfterMs;
						}
						throw error;
					}

					return res;
				}, retryConfig, token);

				if (!response.body) {
					throw new Error("No response body from Ollama API");
				}
				apiUsage = await ollamaApi.processStreamingResponse(response.body, trackingProgress, token);
			} else if (apiMode === "anthropic") {
				// Anthropic API mode
				const anthropicApi = new AnthropicApi(model.id);
				const anthropicMessages = anthropicApi.convertMessages(messages, modelConfig);

				// requestBody
				let requestBody: AnthropicRequestBody = {
					model: parsedModelId.baseId,
					messages: anthropicMessages,
					stream: true,
				};
				requestBody = anthropicApi.prepareRequestBody(requestBody, um, options);

				// send Anthropic chat request with retry
				const normalizedBaseUrl = BASE_URL.replace(/\/+$/, "");
				// Some providers require configuring the baseUrl with a version suffix (e.g. .../v1).
				// Avoid double-appending (e.g. .../v1/v1/messages).
				const url = normalizedBaseUrl.endsWith("/v1")
					? `${normalizedBaseUrl}/messages`
					: `${normalizedBaseUrl}/v1/messages`;
				logger.debug("request.body", { url, requestBody });
				const response = await executeWithRetry(async () => {
					const res = await fetchWithCancellation(url, {
						method: "POST",
						headers: requestHeaders,
						body: JSON.stringify(requestBody),
					});

					if (!res.ok) {
						const errorText = await res.text();
						console.error("[Anthropic Provider] Anthropic API error response", errorText);
						const error = new Error(
							`Anthropic API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`
						);
						const retryAfterMs = parseRetryAfterMs(res.headers.get("Retry-After") ?? "");
						if (retryAfterMs !== undefined) {
							(error as { retryAfterMs?: number }).retryAfterMs = retryAfterMs;
						}
						throw error;
					}

					return res;
				}, retryConfig, token);

				if (!response.body) {
					throw new Error("No response body from Anthropic API");
				}
				apiUsage = await anthropicApi.processStreamingResponse(response.body, trackingProgress, token);
			} else if (apiMode === "openai-responses") {
				// OpenAI Responses API mode
				const openaiResponsesApi = new OpenaiResponsesApi(model.id);
				const normalizedBaseUrl = BASE_URL.replace(/\/+$/, "");
				const statefulModelId = parsedModelId.baseId;

				// Convert full history once (also extracts system `instructions`).
				const fullInput = openaiResponsesApi.convertMessages(messages, modelConfig);

				const marker = findLastOpenAIResponsesStatefulMarker(statefulModelId, messages);
				let deltaInput: unknown[] | null = null;
				if (marker && marker.index >= 0 && marker.index < messages.length - 1) {
					const deltaMessages = messages.slice(marker.index + 1);
					const converted = openaiResponsesApi.convertMessages(deltaMessages, modelConfig);
					if (converted.length > 0) {
						deltaInput = converted;
					}
				}

				const canUsePreviousResponseId =
					!!marker?.marker &&
					!this._openaiResponsesPreviousResponseIdUnsupportedBaseUrls.has(normalizedBaseUrl) &&
					Array.isArray(deltaInput) &&
					deltaInput.length > 0;

				const input = canUsePreviousResponseId ? deltaInput! : fullInput;

				// requestBody
				let requestBody: Record<string, unknown> = {
					model: parsedModelId.baseId,
					input,
					stream: true,
				};

				requestBody = openaiResponsesApi.prepareRequestBody(requestBody, um, options);

				// Add prompt_cache_key to enable OpenAI prompt caching.
				// Without this parameter, cached_tokens is always 0 even with identical requests.
				if (!requestBody.prompt_cache_key) {
					requestBody.prompt_cache_key = `oaicopilot-${parsedModelId.baseId}`;
				}
				// send Responses API request with retry
				const url = `${normalizedBaseUrl}/responses`;
				logger.debug("request.body", { url, requestBody });

				// If the user explicitly set `previous_response_id` via `extra`, don't apply stateful slicing.
				let addedPreviousResponseId = false;
				if (requestBody.previous_response_id !== undefined) {
					requestBody.input = fullInput;
				} else if (canUsePreviousResponseId) {
					requestBody.previous_response_id = marker!.marker;
					addedPreviousResponseId = true;
				}

				const sendRequest = async (body: Record<string, unknown>) =>
					await executeWithRetry(async () => {
						const res = await fetchWithCancellation(url, {
							method: "POST",
							headers: requestHeaders,
							body: JSON.stringify(body),
						});

						if (!res.ok) {
							const errorText = await res.text();
							const error = new Error(
								`Responses API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`
							);
							(error as { status?: number; errorText?: string }).status = res.status;
							(error as { status?: number; errorText?: string }).errorText = errorText;
							const retryAfterMs = parseRetryAfterMs(res.headers.get("Retry-After") ?? "");
							if (retryAfterMs !== undefined) {
								(error as { retryAfterMs?: number }).retryAfterMs = retryAfterMs;
							}
							throw error;
						}

						return res;
					}, retryConfig, token);

				let response: Response;
				try {
					response = await sendRequest(requestBody);
				} catch (err) {
					// Some Responses-compatible gateways don't support `previous_response_id`.
					// Fall back to sending full history when the previous-response attempt fails.
					const status = (err as { status?: unknown })?.status;
					const shouldFallback =
						addedPreviousResponseId && typeof status === "number" && status >= 400 && status < 500 && status !== 429;
					if (!shouldFallback) {
						throw err;
					}

					this._openaiResponsesPreviousResponseIdUnsupportedBaseUrls.add(normalizedBaseUrl);

					let fallbackBody: Record<string, unknown> = {
						model: parsedModelId.baseId,
						input: fullInput,
						stream: true,
					};
					fallbackBody = openaiResponsesApi.prepareRequestBody(fallbackBody, um, options);
					delete fallbackBody.previous_response_id;
					response = await sendRequest(fallbackBody);
				}

				if (!response.body) {
					throw new Error("No response body from Responses API");
				}
				apiUsage = await openaiResponsesApi.processStreamingResponse(response.body, trackingProgress, token);

				// Append a stateful marker so future requests can reuse `previous_response_id` (Copilot Chat style).
				const responseId = openaiResponsesApi.responseId;
				if (responseId) {
					trackingProgress.report(createOpenAIResponsesStatefulMarkerPart(statefulModelId, responseId));
				}
			} else if (apiMode === "gemini") {
				// Gemini native API mode
				const geminiApi = new GeminiApi(model.id, this._geminiToolCallMetaByCallId);
				const geminiMessages = geminiApi.convertMessages(messages, modelConfig);

				const systemParts: string[] = [];
				const contents: GeminiGenerateContentRequest["contents"] = [];
				for (const msg of geminiMessages) {
					if (msg.role === "system") {
						const text = msg.parts
							.map((p) =>
								p && typeof p === "object" && typeof (p as { text?: unknown }).text === "string"
									? String((p as { text: string }).text)
									: ""
							)
							.join("")
							.trim();
						if (text) {
							systemParts.push(text);
						}
						continue;
					}
					contents.push({ role: msg.role, parts: msg.parts });
				}

				let requestBody: GeminiGenerateContentRequest = {
					contents,
				};
				if (systemParts.length > 0) {
					requestBody.systemInstruction = { role: "user", parts: [{ text: systemParts.join("\n") }] };
				}
				requestBody = geminiApi.prepareRequestBody(requestBody, um, options);

				const url = buildGeminiGenerateContentUrl(BASE_URL, parsedModelId.baseId, true);
				logger.debug("request.body", { url, requestBody });
				if (!url) {
					throw new Error("Invalid Gemini base URL configuration.");
				}

				const response = await executeWithRetry(async () => {
					const res = await fetchWithCancellation(url, {
						method: "POST",
						headers: requestHeaders,
						body: JSON.stringify(requestBody),
					});

					if (!res.ok) {
						const errorText = await res.text();
						console.error("[Gemini Provider] Gemini API error response", errorText);
						const error = new Error(
							`Gemini API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`
						);
						const retryAfterMs = parseRetryAfterMs(res.headers.get("Retry-After") ?? "");
						if (retryAfterMs !== undefined) {
							(error as { retryAfterMs?: number }).retryAfterMs = retryAfterMs;
						}
						throw error;
					}

					return res;
				}, retryConfig, token);

				if (!response.body) {
					throw new Error("No response body from Gemini API");
				}
				apiUsage = await geminiApi.processStreamingResponse(response.body, trackingProgress, token);
			} else {
				// OpenAI compatible API mode (default)
				const openaiApi = new OpenaiApi(model.id);
				const openaiMessages = openaiApi.convertMessages(messages, modelConfig);

				// requestBody
				let requestBody: Record<string, unknown> = {
					model: parsedModelId.baseId,
					messages: openaiMessages,
					stream: true,
					stream_options: { include_usage: true },
				};
				requestBody = openaiApi.prepareRequestBody(requestBody, um, options);

				// send chat request with retry
				const url = `${BASE_URL.replace(/\/+$/, "")}/chat/completions`;
				logger.debug("request.body", { url, requestBody });
				const response = await executeWithRetry(async () => {
					const res = await fetchWithCancellation(url, {
						method: "POST",
						headers: requestHeaders,
						body: JSON.stringify(requestBody),
					});

					if (!res.ok) {
						const errorText = await res.text();
						console.error("[OAI Compatible Model Provider] OAI Compatible API error response", errorText);
						const error = new Error(
							`OAI Compatible API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`
						);
						const retryAfterMs = parseRetryAfterMs(res.headers.get("Retry-After") ?? "");
						if (retryAfterMs !== undefined) {
							(error as { retryAfterMs?: number }).retryAfterMs = retryAfterMs;
						}
						throw error;
					}

					return res;
				}, retryConfig, token);

				if (!response.body) {
					throw new Error("No response body from OAI Compatible API");
				}
				apiUsage = await openaiApi.processStreamingResponse(response.body, trackingProgress, token);
			}
		} catch (err) {
			console.error("[OAI Compatible Model Provider] Chat request failed", {
				modelId: model.id,
				messageCount: messages.length,
				error: err instanceof Error ? { name: err.name, message: err.message } : String(err),
			});
			logger.error("request.error", {
				modelId: model.id,
				messageCount: messages.length,
				errorName: err instanceof Error ? err.name : String(err),
				errorMessage: err instanceof Error ? err.message : String(err),
			});
			throw err;
		} finally {
			const durationMs = Date.now() - requestStartTime;
			const effectivePromptTokens = apiUsage?.promptTokens ?? requestUsage.promptTokens;
			const effectiveCompletionTokens = apiUsage?.completionTokens ?? requestUsage.completionTokens;
			const effectiveReasoningTokens = apiUsage?.reasoningTokens ?? requestUsage.reasoningTokens;
			const effectiveTotalTokens = apiUsage?.totalTokens ?? effectivePromptTokens + effectiveCompletionTokens;
			const effectiveCachedPromptTokens = apiUsage?.cachedPromptTokens ?? 0;
			const usageSource: ApiUsage["source"] = apiUsage?.source ?? "estimate";
			const requestCostUsd =
				inputPricePerMillion !== undefined || outputPricePerMillion !== undefined
					? ((effectivePromptTokens * (inputPricePerMillion ?? 0)) +
							(effectiveCompletionTokens * (outputPricePerMillion ?? 0))) /
						1_000_000
					: undefined;
			this._sessionPromptTokens += effectivePromptTokens;
			this._sessionCompletionTokens += effectiveCompletionTokens;
			if (requestCostUsd !== undefined) {
				this._sessionCostUsd += requestCostUsd;
			}
			usageTracker.record({
				modelId: model.id,
				apiMode: apiModeForUsage,
				requestInitiator: options.requestInitiator,
				durationMs,
				promptTokens: effectivePromptTokens,
				completionTokens: effectiveCompletionTokens,
				totalTokens: effectiveTotalTokens,
				reasoningTokens: effectiveReasoningTokens,
				cachedPromptTokens: effectiveCachedPromptTokens,
				toolCallTokens: requestUsage.toolCallTokens,
				usageSource,
				requestCostUsd,
			});
			updateAgentUsageStatusBar(this.statusBarItem, {
				modelId: model.id,
				promptTokens: effectivePromptTokens,
				completionTokens: effectiveCompletionTokens,
				reasoningTokens: effectiveReasoningTokens,
				toolCallTokens: requestUsage.toolCallTokens,
				requestCostUsd,
				sessionPromptTokens: this._sessionPromptTokens,
				sessionCompletionTokens: this._sessionCompletionTokens,
				sessionCostUsd: this._sessionCostUsd > 0 ? this._sessionCostUsd : requestCostUsd,
			});
			logger.info("request.end", {
				modelId: model.id,
				durationMs,
				promptTokens: effectivePromptTokens,
				completionTokens: effectiveCompletionTokens,
				totalTokens: effectiveTotalTokens,
				reasoningTokens: effectiveReasoningTokens,
				cachedPromptTokens: effectiveCachedPromptTokens,
				toolCallTokens: requestUsage.toolCallTokens,
				usageSource,
				requestCostUsd,
			});
			// Update last request time after successful completion
			this._lastRequestTime = Date.now();
		}
	}

	/**
	 * Ensure an API key exists in SecretStorage, optionally prompting the user when not silent.
	 * @param useGenericKey If true, use generic API key.
	 * @param provider Optional provider name to get provider-specific API key.
	 */
	private async ensureApiKey(useGenericKey: boolean, provider?: string): Promise<string | undefined> {
		// Try to get provider-specific API key first
		let apiKey: string | undefined;
		if (provider && provider.trim() !== "") {
			const normalizedProvider = provider.trim().toLowerCase();
			const providerKey = `oaicopilot.apiKey.${normalizedProvider}`;
			apiKey = await this.secrets.get(providerKey);

			if (!apiKey && !useGenericKey) {
				const entered = await vscode.window.showInputBox({
					title: `OAI Compatible API Key for ${normalizedProvider}`,
					prompt: `Enter your OAI Compatible API key for ${normalizedProvider}`,
					ignoreFocusOut: true,
					password: true,
				});
				if (entered && entered.trim()) {
					apiKey = entered.trim();
					await this.secrets.store(providerKey, apiKey);
				}
			}
		}

		// Fall back to generic API key
		if (!apiKey) {
			apiKey = await this.secrets.get("oaicopilot.apiKey");
		}

		if (!apiKey && useGenericKey) {
			const entered = await vscode.window.showInputBox({
				title: "OAI Compatible API Key",
				prompt: "Enter your OAI Compatible API key",
				ignoreFocusOut: true,
				password: true,
			});
			if (entered && entered.trim()) {
				apiKey = entered.trim();
				await this.secrets.store("oaicopilot.apiKey", apiKey);
			}
		}
		return apiKey;
	}
}

type OpenAIResponsesStatefulMarkerLocation = { marker: string; index: number };

function createOpenAIResponsesStatefulMarkerPart(modelId: string, marker: string): vscode.LanguageModelDataPart {
	const payload = `${modelId}\\${marker}`;
	const bytes = new TextEncoder().encode(payload);
	return new vscode.LanguageModelDataPart(bytes, HuggingFaceChatModelProvider.OPENAI_RESPONSES_STATEFUL_MARKER_MIME);
}

function parseOpenAIResponsesStatefulMarkerPart(part: unknown): { modelId: string; marker: string } | null {
	const maybe = part as { mimeType?: unknown; data?: unknown };
	if (!maybe || typeof maybe !== "object") {
		return null;
	}
	if (typeof maybe.mimeType !== "string") {
		return null;
	}
	if (!(maybe.data instanceof Uint8Array)) {
		return null;
	}
	if (maybe.mimeType !== HuggingFaceChatModelProvider.OPENAI_RESPONSES_STATEFUL_MARKER_MIME) {
		return null;
	}

	try {
		const decoded = new TextDecoder().decode(maybe.data);
		const separator = "\\";
		const sep = decoded.indexOf(separator);
		if (sep <= 0) {
			return null;
		}
		const modelId = decoded.slice(0, sep).trim();
		const marker = decoded.slice(sep + separator.length).trim();
		if (!modelId || !marker) {
			return null;
		}
		return { modelId, marker };
	} catch {
		return null;
	}
}

function findLastOpenAIResponsesStatefulMarker(
	modelId: string,
	messages: readonly LanguageModelChatRequestMessage[]
): OpenAIResponsesStatefulMarkerLocation | null {
	for (let i = messages.length - 1; i >= 0; i--) {
		if (messages[i].role !== vscode.LanguageModelChatMessageRole.Assistant) {
			continue;
		}
		for (const part of messages[i].content ?? []) {
			const parsed = parseOpenAIResponsesStatefulMarkerPart(part);
			if (parsed && parsed.modelId === modelId) {
				return { marker: parsed.marker, index: i };
			}
		}
	}
	return null;
}