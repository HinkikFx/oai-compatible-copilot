import * as vscode from "vscode";
import { LanguageModelChatInformation, LanguageModelChatRequestMessage, LanguageModelChatTool } from "vscode";
import { countMessageTokens, countToolTokens } from "./provideToken";

export function initStatusBar(context: vscode.ExtensionContext): vscode.StatusBarItem {
	// Create status bar item for token count display
	const tokenCountStatusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
	tokenCountStatusBarItem.name = "Token Count";
	tokenCountStatusBarItem.text = "$(symbol-numeric) Ready";
	tokenCountStatusBarItem.tooltip = "Current model token usage - Click to Open Configuration UI";
	tokenCountStatusBarItem.command = "oaicopilot.openConfig";
	context.subscriptions.push(tokenCountStatusBarItem);
	// Show the status bar item initially
	tokenCountStatusBarItem.show();
	return tokenCountStatusBarItem;
}

/**
 * Format number to thousands (K, M, B) format
 * @param value The number to format
 * @returns Formatted string (e.g., "2.3K", "168.0K")
 */
export function formatTokenCount(value: number): string {
	if (value >= 1_000_000_000) {
		return (value / 1_000_000_000).toFixed(1) + "B";
	} else if (value >= 1_000_000) {
		return (value / 1_000_000).toFixed(1) + "M";
	} else if (value >= 1_000) {
		return (value / 1_000).toFixed(1) + "K";
	}
	return value.toLocaleString();
}

/**
 * Create a visual progress bar showing token usage
 * @param usedTokens Tokens used
 * @param maxTokens Maximum tokens available
 * @returns Progress bar string (e.g., "▆ 75.2%")
 */
export function createProgressBar(usedTokens: number, maxTokens: number): string {
	const blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
	const usagePercentage = Math.min((usedTokens / maxTokens) * 100, 100);
	const blockIndex = Math.min(Math.floor((usagePercentage / 100) * blocks.length), blocks.length - 1);

	return `${blocks[blockIndex]} ${usagePercentage.toFixed(1)}%`;
}

/**
 * Update the status bar with token usage information
 * @param messages The chat messages to count tokens for
 * @param tools Optional tool definitions to count tokens for
 * @param model The language model information
 * @param statusBarItem The status bar item to update
 * @param modelConfig Configuration including reasoning settings
 */
export async function updateContextStatusBar(
	messages: readonly LanguageModelChatRequestMessage[],
	tools: readonly LanguageModelChatTool[] | undefined,
	model: LanguageModelChatInformation,
	statusBarItem: vscode.StatusBarItem,
	modelConfig: { includeReasoningInRequest: boolean }
): Promise<void> {
	// Calculate tokens for all messages in parallel
	const tokenCountPromises = messages.map((message) => countMessageTokens(message, modelConfig).catch(() => 0));

	const tokenCounts = await Promise.all(tokenCountPromises);
	const messagesTokens = tokenCounts.reduce((sum, count) => sum + count, 0);

	// Calculate tool definition tokens
	let toolTokens = 0;
	if (tools && tools.length > 0) {
		toolTokens = await countToolTokens(tools);
	}

	// Total tokens: messages + tool definitions + reserved output
	const totalTokenCount = messagesTokens + toolTokens;
	const maxTokens = Math.max(1, model.maxInputTokens + model.maxOutputTokens);

	// Create visual progress bar with single progressive block
	const progressBar = createProgressBar(totalTokenCount, maxTokens);
	const displayText = `$(symbol-parameter) ${progressBar}`;
	statusBarItem.text = displayText;
	statusBarItem.tooltip = `Token Usage: ${formatTokenCount(totalTokenCount)} / ${formatTokenCount(maxTokens)}\n
${progressBar}\n
  - Messages: ${formatTokenCount(messagesTokens)}  (${Math.min((messagesTokens / maxTokens) * 100, 100).toFixed(1)}%)
  - Tools: ${formatTokenCount(toolTokens)}  (${Math.min((toolTokens / maxTokens) * 100, 100).toFixed(1)}%) \n
Click to Open Configuration UI`;

	// Add color coding based on token usage
	const usagePercentage = (totalTokenCount / maxTokens) * 100;
	if (usagePercentage >= 90) {
		statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
	} else if (usagePercentage >= 70) {
		statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
	} else {
		statusBarItem.backgroundColor = undefined;
	}

	statusBarItem.show();
}

export interface AgentUsageSnapshot {
	modelId: string;
	promptTokens: number;
	completionTokens: number;
	reasoningTokens?: number;
	toolCallTokens?: number;
	requestCostUsd?: number;
	sessionPromptTokens: number;
	sessionCompletionTokens: number;
	sessionCostUsd?: number;
}

export function formatUsd(value: number | undefined): string {
	if (value === undefined || !Number.isFinite(value)) {
		return "n/a";
	}
	if (value > 0 && value < 0.0001) {
		return `$${value.toExponential(2)}`;
	}
	return `$${value.toFixed(4)}`;
}

export function updateAgentUsageStatusBar(
	statusBarItem: vscode.StatusBarItem,
	usage: AgentUsageSnapshot
): void {
	const requestTotal = usage.promptTokens + usage.completionTokens;
	const sessionTotal = usage.sessionPromptTokens + usage.sessionCompletionTokens;
	statusBarItem.text = `$(symbol-parameter) ${formatTokenCount(requestTotal)} req / ${formatTokenCount(sessionTotal)} session`;
	statusBarItem.tooltip = [
		`Agent token usage for ${usage.modelId}`,
		``,
		`Last request:`,
		` - Prompt: ${formatTokenCount(usage.promptTokens)}`,
		` - Completion: ${formatTokenCount(usage.completionTokens)}`,
		` - Reasoning: ${formatTokenCount(usage.reasoningTokens ?? 0)}`,
		` - Tool calls: ${formatTokenCount(usage.toolCallTokens ?? 0)}`,
		` - Total: ${formatTokenCount(requestTotal)}`,
		` - Estimated cost: ${formatUsd(usage.requestCostUsd)}`,
		``,
		`Current VS Code session:`,
		` - Prompt: ${formatTokenCount(usage.sessionPromptTokens)}`,
		` - Completion: ${formatTokenCount(usage.sessionCompletionTokens)}`,
		` - Total: ${formatTokenCount(sessionTotal)}`,
		` - Estimated cost: ${formatUsd(usage.sessionCostUsd)}`,
		``,
		`Click to Open Configuration UI`,
	].join("\n");
	statusBarItem.backgroundColor = undefined;
	statusBarItem.show();
}
