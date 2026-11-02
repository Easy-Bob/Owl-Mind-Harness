// The server API. The login token lives in an HttpOnly cookie, so no code here can read it.

export const MAX_MESSAGE_CHARS = 4000;

export type User = { email: string; role: string };
export type Thread = { id: string; title: string; created_at: string };
export type HistoryMessage = { role: "user" | "assistant"; content: string };
export type RunStarted = { thread_id: string; run_id: string };

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
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
  if (status >= 502 && status <= 504) {
    return SERVER_UNAVAILABLE;
  }
  if (typeof data?.detail === "string") {
    return data.detail;
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
    throw new ApiError(response.status, errorMessage(response.status, data));
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
};
