import { request } from "../shared/api";

export type RunRow = {
  id: string;
  email: string;
  status: string;
  route: string | null;
  task_kind: string | null;
  created_at: string;
  model_calls: number;
  tokens: number | null;
  cost_usd: number | null;
  unknown_usage: number;
  searched: boolean;
  handoff_step: string | null;
};

export const ROUTES = ["rule", "template", "orchestrated", "handoff"] as const;

export type RequestPage = { items: RunRow[]; total: number; page: number; size: number };

export type CallRow = {
  called_at: string;
  component: string;
  model: string;
  input_tokens: number | null;
  output_tokens: number | null;
  cache_hit_tokens: number | null;
  reasoning_tokens: number | null;
  cost_usd: number | null;
  latency_ms: number;
  outcome: string;
};

export type RequestDetail = {
  id: string;
  email: string;
  status: string;
  route: string | null;
  task_kind: string | null;
  created_at: string;
  calls: CallRow[];
  events: { kind: string; at: string; data: Record<string, unknown> }[];
};

export type GroupRow = {
  name: string | null;
  calls: number;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_usd: number | null;
  unknown_usage: number;
  cache_hit_rate: number | null;
};

export type ModelCallSummary = {
  days: number;
  totals: { calls: number; cost_usd: number | null; unknown_usage: number };
  cache_hit_rate: number | null;
  by_component: GroupRow[];
  by_model: GroupRow[];
  by_task_kind: GroupRow[];
  outcomes: { name: string; calls: number }[];
  top_runs: { run_id: string; cost_usd: number; calls: number }[];
};

export const adminApi = {
  requests: (query: string, page: number, size: number, route = "") =>
    request<RequestPage>(
      `/admin/requests?${new URLSearchParams({ q: query, page: String(page), size: String(size), route })}`,
    ),
  request: (runId: string) => request<RequestDetail>(`/admin/requests/${runId}`),
  modelCalls: (days: number) => request<ModelCallSummary>(`/admin/model-calls?days=${days}`),
};
