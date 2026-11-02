// Follows one run's server-sent events. If the connection drops, the browser reconnects by itself
// and sends Last-Event-ID, so the server continues from the event after the last one received.

export type RunUpdate =
  | { kind: "text"; text: string }
  | { kind: "progress"; label: string }
  | { kind: "finished"; status: string; error?: string };

const PROGRESS_LABELS: Record<string, (data: { name?: string }) => string> = {
  subagent_start: ({ name }) => `Asking ${(name ?? "a tutor").replaceAll("_", " ")}`,
  tool_start: ({ name }) => `Using ${(name ?? "a tool").replaceAll("_", " ")}`,
  todo_delta: () => "Planning",
};

export function followRun(runId: string, onUpdate: (update: RunUpdate) => void): () => void {
  const source = new EventSource(`/runs/${runId}/stream`);
  const finish = (update: RunUpdate) => {
    source.close();
    onUpdate(update);
  };

  source.addEventListener("text_delta", (event) => {
    onUpdate({ kind: "text", text: JSON.parse(event.data).text });
  });
  for (const [kind, label] of Object.entries(PROGRESS_LABELS)) {
    source.addEventListener(kind, (event) => {
      onUpdate({ kind: "progress", label: label(JSON.parse(event.data)) });
    });
  }
  source.addEventListener("done", (event) => {
    finish({ kind: "finished", status: JSON.parse(event.data).status });
  });
  // The server's "error" event carries data. The browser's own connection error doesn't.
  source.addEventListener("error", (event) => {
    if (event instanceof MessageEvent) {
      finish({ kind: "finished", status: "failed", error: JSON.parse(event.data).message });
    } else if (source.readyState === EventSource.CLOSED) {
      // The browser gave up, as it does on a refused response. It retries a dropped connection.
      finish({
        kind: "finished",
        status: "failed",
        error: "The connection was lost. Reload the page to see the latest messages.",
      });
    }
  });
  return () => source.close();
}
