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

type Window = {
  runs: { route: string; status: string; runs: number }[];
  stopped: { reason: string; runs: number }[];
  calls: {
    calls: number;
    cost_usd: number | null;
    unknown_cost: number;
    errors: number;
    truncated: number;
  };
};

export type Overview = { today: Window; last_7_days: Window; global_budget_used: number | null };

export type UserRow = {
  student_id: string;
  email: string;
  runs_today: number;
  runs_30d: number;
  tokens_today: number | null;
  tokens_30d: number | null;
  cost_today: number | null;
  cost_30d: number | null;
  unknown_usage: number;
  budget_used_today: number | null;
};

export type UserUsage = {
  student_id: string;
  email: string;
  budget_used_today: number | null;
  days: { day: string; runs: number; tokens: number | null; cost_usd: number | null; unknown_usage: number }[];
  runs: { run_id: string; created_at: string; route: string | null; task_kind: string | null; status: string }[];
};

export type Savings = {
  days: number;
  how: Record<string, string>;
  cache: {
    cache_hit_tokens: number | null;
    cache_miss_tokens: number | null;
    saved_usd: number | null;
    cache_hit_rate: number | null;
  };
  summaries: { calls: number; tokens_removed: number | null; cost_usd: number | null };
  ladder: Record<string, number>;
  integrity: Record<string, number>;
  joined: number;
  runs: { without_agent: number; with_agent: number };
};

export type QualityRow = {
  prompt_version: string;
  runs: number;
  clarify_rate: number | null;
  stopped_rate: number | null;
  learn_answers: number;
  learn_citation_rate: number | null;
  integrity_checked: number;
  integrity_passed: number;
  integrity_retried: number;
  integrity_blocked: number;
  integrity_blocked_rate: number | null;
};

export type EvalFile = {
  file: string;
  date: string | null;
  prompt_version: string | null;
  large_model: string | null;
  scores: Record<string, number | null>;
};

export type Quality = { days: number; live: QualityRow[]; evals: EvalFile[] };

export type Reliability = {
  days: number;
  trouble: { day: string; status: string; reason: string; runs: number }[];
  runs_per_day: { day: string; runs: number }[];
  workers_used: boolean;
  worker_status_error?: string;
  workers?: { worker: string; seconds_ago: number }[];
  jobs?: { job: string; success: number | null; failure: number | null }[];
  dead_letters?: { run_id: string | null; deliveries: number; at: number }[];
  dead_letter_total?: number;
  queue?: { lag: number; pending: number };
};

export const adminApi = {
  requests: (query: string, page: number, size: number, route = "") =>
    request<RequestPage>(
      `/admin/requests?${new URLSearchParams({ q: query, page: String(page), size: String(size), route })}`,
    ),
  request: (runId: string) => request<RequestDetail>(`/admin/requests/${runId}`),
  modelCalls: (days: number) => request<ModelCallSummary>(`/admin/model-calls?days=${days}`),
  overview: () => request<Overview>("/admin/overview"),
  users: (query: string) =>
    request<{ users: UserRow[]; daily_tokens: number }>(`/admin/users?${new URLSearchParams({ q: query })}`),
  user: (studentId: string) => request<UserUsage>(`/admin/users/${studentId}`),
  savings: (days: number) => request<Savings>(`/admin/savings?days=${days}`),
  quality: (days: number) => request<Quality>(`/admin/quality?days=${days}`),
  reliability: (days: number) => request<Reliability>(`/admin/reliability?days=${days}`),
};
