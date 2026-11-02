import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  ApiError,
  MAX_MESSAGE_CHARS,
  type Handoff,
  type HistoryMessage,
  type SimilarPost,
  type Thread,
  type User,
} from "../shared/api";
import { HandoffCard } from "./HandoffCard";
import { Message } from "./Message";
import { followRun } from "./runStream";

type Props = { user: User; onLoggedOut: () => void };

export function ChatPage({ user, onLoggedOut }: Props) {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [messages, setMessages] = useState<HistoryMessage[]>([]);
  const [answer, setAnswer] = useState("");
  const [progress, setProgress] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [draft, setDraft] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [handoffs, setHandoffs] = useState<Handoff[]>([]);
  // Similar posts arrive only with the live event, so they're kept per draft here.
  const [similar, setSimilar] = useState<Record<string, SimilarPost[]>>({});
  // Set straight away, so a double click can't start two runs before React re-renders.
  const sending = useRef(false);
  const stopFollowing = useRef<(() => void) | null>(null);
  const log = useRef<HTMLDivElement>(null);

  const handleFailure = useCallback(
    (caught: unknown) => {
      if (caught instanceof ApiError && caught.status === 401) {
        onLoggedOut();
      } else {
        setError(caught instanceof Error ? caught.message : "Something went wrong.");
      }
    },
    [onLoggedOut],
  );

  useEffect(() => {
    api.threads().then(setThreads).catch(handleFailure);
    return () => stopFollowing.current?.();
  }, [handleFailure]);

  // Keep the newest text in view as it arrives.
  useEffect(() => {
    if (log.current) {
      log.current.scrollTop = log.current.scrollHeight;
    }
  }, [messages, answer, progress, error]);

  // Handoff is optional on the server, so failing to list drafts is not an error to show.
  const loadHandoffs = useCallback((threadId: string) => {
    api
      .handoffs(threadId)
      .then(setHandoffs)
      .catch(() => setHandoffs([]));
  }, []);

  function follow(runId: string, threadId: string) {
    let text = "";
    let drafted = false;
    setAnswer("");
    setProgress("Thinking");
    setRunning(true);
    stopFollowing.current = followRun(runId, (update) => {
      if (update.kind === "text") {
        text += update.text;
        setAnswer(text);
      } else if (update.kind === "progress") {
        setProgress(update.label);
      } else if (update.kind === "handoff") {
        drafted = true;
        setSimilar((current) => ({ ...current, [update.id]: update.similar }));
      } else {
        if (text) {
          setMessages((current) => [...current, { role: "assistant", content: text }]);
        }
        setAnswer("");
        setProgress("");
        setRunning(false);
        sending.current = false;
        if (update.error) {
          setError(update.error);
        }
        if (drafted) {
          loadHandoffs(threadId);
        }
      }
    });
  }

  async function send() {
    const message = draft.trim();
    if (!message || sending.current) {
      return;
    }
    sending.current = true;
    setError("");
    setDraft("");
    setMessages((current) => [...current, { role: "user", content: message }]);
    try {
      if (selected === null) {
        const started = await api.startThread(message);
        setSelected(started.thread_id);
        follow(started.run_id, started.thread_id);
        setThreads(await api.threads());
      } else {
        follow((await api.sendMessage(selected, message)).run_id, selected);
      }
    } catch (caught) {
      sending.current = false;
      // Nothing started, so give the message back to send again.
      setMessages((current) => current.slice(0, -1));
      setDraft(message);
      handleFailure(caught);
    }
  }

  async function open(threadId: string) {
    if (running) {
      return;
    }
    setMenuOpen(false);
    setError("");
    setSelected(threadId);
    setMessages([]);
    setHandoffs([]);
    loadHandoffs(threadId);
    try {
      setMessages(await api.history(threadId));
    } catch (caught) {
      handleFailure(caught);
    }
  }

  function newChat() {
    if (running) {
      return;
    }
    setMenuOpen(false);
    setError("");
    setSelected(null);
    setMessages([]);
    setHandoffs([]);
    setDraft("");
  }

  async function logout() {
    stopFollowing.current?.();
    await api.logout().catch(() => undefined);
    onLoggedOut();
  }

  return (
    <div className="chat">
      <button
        type="button"
        className="menu-button"
        aria-expanded={menuOpen}
        onClick={() => setMenuOpen(!menuOpen)}
      >
        Sessions
      </button>
      <nav className={`sessions ${menuOpen ? "sessions-open" : ""}`} aria-label="Sessions">
        <button type="button" className="new-chat" onClick={newChat} disabled={running}>
          + New chat
        </button>
        <ul>
          {threads.map((thread) => (
            <li key={thread.id}>
              <button
                type="button"
                aria-current={thread.id === selected ? "true" : undefined}
                onClick={() => open(thread.id)}
                disabled={running}
              >
                {thread.title}
              </button>
            </li>
          ))}
        </ul>
        <div className="account">
          <span>{user.email}</span>
          <button type="button" onClick={logout}>
            Log out
          </button>
        </div>
      </nav>

      <main className="conversation">
        <div className="messages" role="log" aria-live="polite" ref={log}>
          {messages.length === 0 && !running && (
            <p className="empty">Ask a question to start studying.</p>
          )}
          {messages.map((message, index) => (
            <Message key={index} role={message.role} content={message.content} />
          ))}
          {handoffs.map((handoff) => (
            <HandoffCard
              key={handoff.id}
              handoff={handoff}
              similar={similar[handoff.id] ?? []}
              onFailure={handleFailure}
              onChange={(changed) =>
                setHandoffs((current) =>
                  current.map((item) => (item.id === changed.id ? { ...item, ...changed } : item)),
                )
              }
            />
          ))}
          {answer && <Message role="assistant" content={answer} />}
          {running && <p className="progress">{progress}...</p>}
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </div>
        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void send();
          }}
        >
          <label htmlFor="message">Your message</label>
          <textarea
            id="message"
            value={draft}
            maxLength={MAX_MESSAGE_CHARS}
            rows={3}
            disabled={running}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send();
              }
            }}
          />
          <div className="composer-footer">
            <span>
              {draft.length} / {MAX_MESSAGE_CHARS}
            </span>
            <button type="submit" disabled={running || !draft.trim()}>
              Send
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
