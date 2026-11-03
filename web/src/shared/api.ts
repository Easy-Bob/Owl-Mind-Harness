// The server API. The login token lives in an HttpOnly cookie, so no code here can read it.

export const MAX_MESSAGE_CHARS = 4000;

export type User = { email: string; role: string };
export type Thread = { id: string; title: string; created_at: string };
export type HistoryMessage = { role: "user" | "assistant"; content: string };
export type RunStarted = { thread_id: string; run_id: string };
export type SimilarPost = { title: string; url: string };
export type ForumReply = {
  id: string;
  author: string;
  body: string;
  created_at: string;
  is_answer: boolean;
  new: boolean;
};
export type ForumAccount = {
  available: boolean;
  connected: boolean;
  login: string | null;
  needs_reconnect: boolean;
};
export type Handoff = {
  id: string;
  status: "draft" | "posting" | "posted" | "retrying" | "fallback" | "cancelled";
  title: string;
  body: string;
  graded: boolean;
  url: string | null;
  forum_url: string | null;
  // A GitHub login, or "bot" for the Owl-Mind app.
  post_as: string | null;
  replies: ForumReply[];
  // AI rewrites left for this draft. 0 hides the button.
  improve_left?: number;
};
export type Rewrite = {
  title: string;
  body: string;
  removed: Record<string, number>;
  remaining: number;
};

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    // The response body, for the few answers that carry data with an error status.
    readonly data: unknown = null,
  ) {
    super(message);
  }
}

const SERVER_UNAVAILABLE = "Can't reach the server. Check that it's running, then try again.";
const FIELD_LABELS: Record<string, string> = {
  email: "Enter a valid email address.",
  password: "The password must be 10 to 128 characters.",
  invite_code: "That invite code isn't valid.",
  message: `A message can be up to ${MAX_MESSAGE_CHARS} characters.`,
};

function errorMessage(status: number, data: { detail?: unknown } | null): string {
  // The API's own message comes first. A proxy that can't reach the API sends no JSON.
  if (typeof data?.detail === "string") {
    return data.detail;
  }
  if (status >= 502 && status <= 504) {
    return SERVER_UNAVAILABLE;
  }
  // Validation errors name the field but never repeat what was typed.
  if (Array.isArray(data?.detail)) {
    const field = data.detail[0]?.field?.at(-1);
    return FIELD_LABELS[field] ?? "Please check what you entered.";
  }
  return "Something went wrong on the server. Please try again.";
}

export async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  }).catch(() => {
    throw new ApiError(0, SERVER_UNAVAILABLE);
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new ApiError(response.status, errorMessage(response.status, data), data);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

export const api = {
  me: () => request<User>("/auth/me"),
  notice: () => request<{ notice: string }>("/auth/notice"),
  login: (email: string, password: string) =>
    request<User>("/auth/login", { email, password }),
  signup: (email: string, password: string, invite_code: string, accepted_notice: boolean) =>
    request<User>("/auth/signup", { email, password, invite_code, accepted_notice }),
  logout: () => request<void>("/auth/logout", {}),
  threads: () => request<Thread[]>("/threads"),
  history: (threadId: string) => request<HistoryMessage[]>(`/threads/${threadId}/messages`),
  startThread: (message: string) => request<RunStarted>("/threads", { message }),
  sendMessage: (threadId: string, message: string) =>
    request<RunStarted>(`/threads/${threadId}/messages`, { message }),
  handoffs: (threadId: string) =>
    request<Handoff[]>(`/handoffs?thread_id=${encodeURIComponent(threadId)}`),
  approveHandoff: (id: string, title: string, body: string, graded: boolean) =>
    request<Handoff>(`/handoffs/${id}/approve`, { title, body, graded }),
  cancelHandoff: (id: string) => request<Handoff>(`/handoffs/${id}/cancel`, {}),
  improveHandoff: (id: string, title: string, body: string) =>
    request<Rewrite>(`/handoffs/${id}/improve`, { title, body }),
  unreadReplies: () => request<Record<string, number>>("/handoffs/unread"),
  markRepliesSeen: (id: string) => request<void>(`/handoffs/${id}/seen`, {}),
  forumAccount: () => request<ForumAccount>("/forum/account"),
  disconnectForum: () => request<void>("/forum/account/disconnect", {}),
};
