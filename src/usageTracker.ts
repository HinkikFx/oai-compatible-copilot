import type { ApiUsage } from "./types";

export interface UsageRecord {
	id: string;
	timestamp: string;
	modelId: string;
	apiMode: string;
	requestInitiator?: string;
	durationMs: number;
	promptTokens: number;
	completionTokens: number;
	totalTokens: number;
	reasoningTokens: number;
	cachedPromptTokens: number;
	toolCallTokens: number;
	usageSource: ApiUsage["source"];
	requestCostUsd?: number;
}

export interface UsageSummary {
	records: UsageRecord[];
	totals: {
		promptTokens: number;
		completionTokens: number;
		totalTokens: number;
		reasoningTokens: number;
		cachedPromptTokens: number;
		toolCallTokens: number;
		costUsd: number;
	};
}

const MAX_USAGE_RECORDS = 500;

class UsageTracker {
	private readonly records: UsageRecord[] = [];

	record(record: Omit<UsageRecord, "id" | "timestamp">): UsageRecord {
		const full: UsageRecord = {
			...record,
			id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
			timestamp: new Date().toISOString(),
		};
		this.records.unshift(full);
		if (this.records.length > MAX_USAGE_RECORDS) {
			this.records.length = MAX_USAGE_RECORDS;
		}
		return full;
	}

	getSummary(): UsageSummary {
		const totals = this.records.reduce(
			(acc, record) => {
				acc.promptTokens += record.promptTokens;
				acc.completionTokens += record.completionTokens;
				acc.totalTokens += record.totalTokens;
				acc.reasoningTokens += record.reasoningTokens;
				acc.cachedPromptTokens += record.cachedPromptTokens;
				acc.toolCallTokens += record.toolCallTokens;
				acc.costUsd += record.requestCostUsd ?? 0;
				return acc;
			},
			{
				promptTokens: 0,
				completionTokens: 0,
				totalTokens: 0,
				reasoningTokens: 0,
				cachedPromptTokens: 0,
				toolCallTokens: 0,
				costUsd: 0,
			}
		);
		return { records: [...this.records], totals };
	}

	clear(): void {
		this.records.length = 0;
	}
}

export const usageTracker = new UsageTracker();
