import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { MAX_MESSAGE_CHARS } from "../shared/api";
import { Message } from "./Message";
import { followRun, type RunUpdate } from "./runStream";

class FakeEventSource {
  static CLOSED = 2;
  static instances: FakeEventSource[] = [];
  readyState = 1;
  listeners: Record<string, ((event: Event) => void)[]> = {};

  constructor(readonly url: string) {
    FakeEventSource.instances.push(this);
  }

  addEventListener(kind: string, listener: (event: Event) => void) {
    (this.listeners[kind] ??= []).push(listener);
  }

  close() {
    this.readyState = FakeEventSource.CLOSED;
  }

  emit(kind: string, data: unknown) {
    const event = new MessageEvent(kind, { data: JSON.stringify(data) });
    this.listeners[kind]?.forEach((listener) => listener(event));
  }

  connectionError() {
    this.listeners.error?.forEach((listener) => listener(new Event("error")));
  }
}

type Reply = { status?: number; body?: unknown };

function fakeServer(routes: Record<string, Reply | ((body: unknown) => Reply)>) {
  const calls: { key: string; body: unknown }[] = [];
  const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${path}`;
    const body = init?.body ? JSON.parse(init.body as string) : undefined;
    calls.push({ key, body });
    const route = routes[key];
    const reply = typeof route === "function" ? route(body) : (route ?? { status: 404, body: {} });
    return new Response(reply.status === 204 ? null : JSON.stringify(reply.body ?? {}), {
      status: reply.status ?? 200,
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

const student = { email: "alice@example.com", role: "student" };
const loggedIn = { "GET /auth/me": { body: student }, "GET /threads": { body: [] } };
const started = { body: { thread_id: "t1", run_id: "r1" }, status: 201 };

beforeEach(() => {
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const lastStream = () => FakeEventSource.instances.at(-1)!;

describe("followRun", () => {
  function follow() {
    const updates: RunUpdate[] = [];
    const stop = followRun("r1", (update) => updates.push(update));
    return { updates, stop, source: lastStream() };
  }

  it("follows the run's stream and reports text and progress", () => {
    const { updates, source } = follow();

    source.emit("text_delta", { text: "hello" });
    source.emit("subagent_start", { name: "practice_coach" });
    source.emit("tool_start", { name: "build_hint" });
    source.emit("todo_delta", { todos: [] });

    expect(source.url).toBe("/runs/r1/stream");
    expect(updates).toEqual([
      { kind: "text", text: "hello" },
      { kind: "progress", label: "Asking practice coach" },
      { kind: "progress", label: "Using build hint" },
      { kind: "progress", label: "Planning" },
    ]);
  });

  it.each(["completed", "stopped"])("closes on done with status %s", (status) => {
    const { updates, source } = follow();

    source.emit("done", { status });

    expect(updates).toEqual([{ kind: "finished", status }]);
    expect(source.readyState).toBe(FakeEventSource.CLOSED);
  });

  it("closes on the server's error event and reports its message", () => {
    const { updates, source } = follow();

    source.emit("error", { message: "Something went wrong" });

    expect(updates).toEqual([{ kind: "finished", status: "failed", error: "Something went wrong" }]);
    expect(source.readyState).toBe(FakeEventSource.CLOSED);
  });

  it("leaves a dropped connection for the browser to retry", () => {
    const { updates, source } = follow();

    source.connectionError();

    expect(updates).toEqual([]);
    expect(source.readyState).not.toBe(FakeEventSource.CLOSED);
  });

  it("reports the connection as lost once the browser gives up", () => {
    const { updates, source } = follow();

    source.readyState = FakeEventSource.CLOSED;
    source.connectionError();

    expect(updates[0]).toMatchObject({ kind: "finished", status: "failed" });
  });

  it("can stop following", () => {
    const { stop, source } = follow();

    stop();

    expect(source.readyState).toBe(FakeEventSource.CLOSED);
  });
});

describe("Message", () => {
  it("renders model Markdown without raw HTML", () => {
    const { container } = render(
      <Message role="assistant" content={'**bold** <script>x</script> <img src="x" onerror="y">'} />,
    );

    expect(container.querySelector("strong")).toHaveTextContent("bold");
    expect(container.querySelector("script, img")).toBeNull();
  });

  it("doesn't turn a script link into a live link", () => {
    const { container } = render(
      <Message role="assistant" content="[click](javascript:alert(1))" />,
    );

    expect(container.querySelector("a")?.getAttribute("href") ?? "").not.toContain("javascript");
  });

  it("shows a student's text as plain text", () => {
    const { container } = render(<Message role="user" content="**not bold** <b>x</b>" />);

    expect(container.querySelector("strong, b")).toBeNull();
    expect(container).toHaveTextContent("**not bold** <b>x</b>");
  });
});

describe("App", () => {
  it("shows login when logged out", async () => {
    fakeServer({ "GET /auth/me": { status: 401 }, "GET /auth/notice": { body: { notice: "n" } } });

    render(<App />);

    expect(await screen.findByRole("button", { name: "Log in" })).toBeVisible();
  });

  it("shows a login error and stays on the form", async () => {
    fakeServer({
      "GET /auth/me": { status: 401 },
      "GET /auth/notice": { body: { notice: "n" } },
      "POST /auth/login": { status: 401, body: { detail: "Incorrect email or password" } },
    });
    const user = userEvent.setup();
    render(<App />);

    await user.type(await screen.findByLabelText("Email"), "a@example.com");
    await user.type(screen.getByLabelText("Password"), "wrong password");
    await user.click(screen.getByRole("button", { name: "Log in" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Incorrect email or password");
  });

  it.each([
    [502, {}, "Can't reach the server"],
    [503, {}, "Can't reach the server"],
    [422, { detail: [{ field: ["body", "email"], type: "value_error" }] }, "valid email"],
    [422, { detail: [{ field: ["body", "password"], type: "too_short" }] }, "10 to 128"],
    [422, { detail: [{ field: ["body", "other"], type: "x" }] }, "check what you entered"],
    [500, {}, "on the server"],
  ])("explains a %i login failure", async (status, body, expected) => {
    fakeServer({
      "GET /auth/me": { status: 401 },
      "GET /auth/notice": { body: { notice: "n" } },
      "POST /auth/login": { status, body },
    });
    const user = userEvent.setup();
    render(<App />);

    await user.type(await screen.findByLabelText("Email"), "a@example.com");
    await user.type(screen.getByLabelText("Password"), "a password here");
    await user.click(screen.getByRole("button", { name: "Log in" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(expected);
  });

  it("explains when the server can't be reached at all", async () => {
    fakeServer({ "GET /auth/me": { status: 401 }, "GET /auth/notice": { body: { notice: "n" } } });
    const user = userEvent.setup();
    render(<App />);
    await user.type(await screen.findByLabelText("Email"), "a@example.com");
    await user.type(screen.getByLabelText("Password"), "a password here");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    await user.click(screen.getByRole("button", { name: "Log in" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Can't reach the server");
  });

  it("requires the notice to be accepted, and sends it with signup", async () => {
    const calls = fakeServer({
      "GET /auth/me": { status: 401 },
      "GET /auth/notice": { body: { notice: "Stored in China" } },
      "POST /auth/signup": { status: 201, body: student },
      "POST /auth/login": { body: student },
      "GET /threads": { body: [] },
    });
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Create an account" }));
    expect(await screen.findByText("Stored in China")).toBeVisible();
    await user.type(screen.getByLabelText("Email"), "a@example.com");
    await user.type(screen.getByLabelText("Password"), "correct horse battery");
    await user.type(screen.getByLabelText("Invite code"), "code");
    await user.click(screen.getByRole("button", { name: "Sign up" }));
    expect(calls.some((call) => call.key === "POST /auth/signup")).toBe(false);

    await user.click(screen.getByLabelText("I understand this notice"));
    await user.click(screen.getByRole("button", { name: "Sign up" }));

    await screen.findByLabelText("Your message");
    expect(calls.find((call) => call.key === "POST /auth/signup")?.body).toMatchObject({
      invite_code: "code",
      accepted_notice: true,
    });
  });

  it("can't send a blank message", async () => {
    const calls = fakeServer({ ...loggedIn });
    const user = userEvent.setup();
    render(<App />);

    const input = await screen.findByLabelText("Your message");
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    await user.type(input, "   {Enter}");

    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(calls.some((call) => call.key === "POST /threads")).toBe(false);
  });

  it("shows an empty state for a user with no sessions", async () => {
    fakeServer({ ...loggedIn });

    render(<App />);

    expect(await screen.findByText("Ask a question to start studying.")).toBeVisible();
  });

  it("limits a message to the server's length and shows the count", async () => {
    fakeServer({ ...loggedIn });
    const user = userEvent.setup();
    render(<App />);

    const input = await screen.findByLabelText("Your message");
    expect(input).toHaveAttribute("maxlength", String(MAX_MESSAGE_CHARS));
    await user.type(input, "abc");

    expect(screen.getByText(`3 / ${MAX_MESSAGE_CHARS}`)).toBeVisible();
  });

  it("shows text as it arrives, then keeps the finished answer", async () => {
    fakeServer({ ...loggedIn, "POST /threads": started });
    const user = userEvent.setup();
    render(<App />);

    await user.type(await screen.findByLabelText("Your message"), "hello{Enter}");
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    lastStream().emit("subagent_start", { name: "concept_tutor" });
    lastStream().emit("text_delta", { text: "part one " });

    expect(await screen.findByText("part one")).toBeVisible();
    expect(await screen.findByText("Asking concept tutor...")).toBeVisible();
    expect(screen.getByLabelText("Your message")).toBeDisabled();

    lastStream().emit("text_delta", { text: "part two" });
    lastStream().emit("done", { status: "completed" });

    expect(await screen.findByText("part one part two")).toBeVisible();
    await waitFor(() => expect(screen.getByLabelText("Your message")).toBeEnabled());
    expect(screen.queryByText(/Asking/)).toBeNull();
  });

  it("shows a failed run's error in the conversation and keeps its partial text", async () => {
    fakeServer({ ...loggedIn, "POST /threads": started });
    const user = userEvent.setup();
    render(<App />);

    await user.type(await screen.findByLabelText("Your message"), "hello{Enter}");
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    lastStream().emit("text_delta", { text: "partial" });
    lastStream().emit("error", { message: "Something went wrong with this answer." });

    expect(await screen.findByRole("alert")).toHaveTextContent("Something went wrong");
    expect(screen.getByText("partial")).toBeVisible();
    await waitFor(() => expect(screen.getByLabelText("Your message")).toBeEnabled());
  });

  it("gives a message back to the student if sending it fails", async () => {
    fakeServer({ ...loggedIn, "POST /threads": { status: 500, body: { detail: "Server trouble" } } });
    const user = userEvent.setup();
    render(<App />);

    await user.type(await screen.findByLabelText("Your message"), "keep me{Enter}");

    expect(await screen.findByRole("alert")).toHaveTextContent("Server trouble");
    expect(screen.getByLabelText("Your message")).toHaveValue("keep me");
    expect(screen.queryByRole("article", { name: "You" })).toBeNull();
    expect(screen.getByLabelText("Your message")).toBeEnabled();
  });

  it("explains a refused second message on a busy thread", async () => {
    fakeServer({
      ...loggedIn,
      "GET /threads": { body: [{ id: "t1", title: "busy", created_at: "2026-11-01T00:00:00Z" }] },
      "GET /threads/t1/messages": { body: [] },
      "POST /threads/t1/messages": { status: 409, body: { detail: "A run is already active" } },
    });
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "busy" }));
    await user.type(screen.getByLabelText("Your message"), "again{Enter}");

    expect(await screen.findByRole("alert")).toHaveTextContent("A run is already active");
  });

  it("returns to login if the session ends", async () => {
    fakeServer({
      ...loggedIn,
      "POST /threads": { status: 401, body: { detail: "Login required" } },
      "GET /auth/notice": { body: { notice: "n" } },
    });
    const user = userEvent.setup();
    render(<App />);

    await user.type(await screen.findByLabelText("Your message"), "hello{Enter}");

    expect(await screen.findByRole("button", { name: "Log in" })).toBeVisible();
  });

  it("shows no Admin mode control to a student", async () => {
    fakeServer({ ...loggedIn });

    render(<App />);
    await screen.findByLabelText("Your message");

    expect(screen.queryByText(/admin/i)).toBeNull();
  });
});

describe("HandoffCard", () => {
  const draft = {
    id: "h1",
    status: "draft",
    title: "Help with BFS",
    body: "**Question**\n\nWhy is my BFS wrong?",
    graded: false,
    url: null,
    forum_url: "https://github.com/owner/forum/discussions",
  };
  const similar = [{ title: "Old BFS question", url: "https://github.com/owner/forum/2" }];

  async function askForTutor(routes: Record<string, Reply | ((body: unknown) => Reply)>) {
    const calls = fakeServer({
      ...loggedIn,
      "POST /threads": started,
      "GET /handoffs?thread_id=t1": { body: [draft] },
      ...routes,
    });
    const user = userEvent.setup();
    render(<App />);
    await user.type(await screen.findByLabelText("Your message"), "talk to a TA{Enter}");
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    lastStream().emit("handoff_draft", { id: "h1", removed: {}, similar });
    lastStream().emit("text_delta", { text: "I drafted a forum post." });
    lastStream().emit("done", { status: "completed" });
    await screen.findByRole("region", { name: "Forum post draft" });
    return { calls, user };
  }

  it("shows the draft for editing, with similar posts and what posting means", async () => {
    await askForTutor({});

    expect(screen.getByLabelText("Title")).toHaveValue("Help with BFS");
    expect(screen.getByLabelText("Post")).toHaveValue(draft.body);
    expect(screen.getByRole("link", { name: "Old BFS question" })).toBeVisible();
    expect(screen.getByText(/Nothing is posted until you approve/)).toBeVisible();
    expect(screen.getByText(/stays on the forum/)).toBeVisible();
  });

  it("posts exactly the edited text once, then links to the post", async () => {
    const posted = {
      ...draft,
      status: "posted",
      url: "https://github.com/owner/forum/discussions/5",
    };
    const { calls, user } = await askForTutor({
      "POST /handoffs/h1/approve": { body: { ...posted, title: "Edited title" } },
    });

    await user.clear(screen.getByLabelText("Title"));
    await user.type(screen.getByLabelText("Title"), "Edited title");
    await user.click(screen.getByLabelText(/graded assignment/));
    await user.dblClick(screen.getByRole("button", { name: "Approve and post" }));

    expect(await screen.findByRole("link", { name: "Edited title" })).toHaveAttribute(
      "href",
      posted.url,
    );
    const approvals = calls.filter((call) => call.key === "POST /handoffs/h1/approve");
    expect(approvals).toEqual([
      {
        key: "POST /handoffs/h1/approve",
        body: { title: "Edited title", body: draft.body, graded: true },
      },
    ]);
  });

  it("shows the redacted text when the server changed it, and asks again", async () => {
    const revised = { ...draft, body: "my password = [secret removed]" };
    const { user } = await askForTutor({
      "POST /handoffs/h1/approve": { status: 409, body: revised },
    });

    await user.click(screen.getByRole("button", { name: "Approve and post" }));

    expect(await screen.findByRole("status")).toHaveTextContent("approve again");
    expect(screen.getByLabelText("Post")).toHaveValue(revised.body);
  });

  it("gives copyable text and the forum link when posting failed", async () => {
    const { user } = await askForTutor({
      "POST /handoffs/h1/approve": { body: { ...draft, status: "fallback" } },
    });

    await user.click(screen.getByRole("button", { name: "Approve and post" }));

    expect(await screen.findByLabelText("Post to copy")).toHaveValue(
      `${draft.title}\n\n${draft.body}`,
    );
    expect(screen.getByRole("link", { name: "the forum" })).toHaveAttribute(
      "href",
      draft.forum_url,
    );
  });

  it("removes the card when cancelled", async () => {
    const { user } = await askForTutor({
      "POST /handoffs/h1/cancel": { body: { ...draft, status: "cancelled" } },
    });

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() =>
      expect(screen.queryByRole("region", { name: "Forum post draft" })).toBeNull(),
    );
  });

  it("shows a refusal on the card", async () => {
    const { user } = await askForTutor({
      "POST /handoffs/h1/approve": {
        status: 400,
        body: { detail: "You can post once every 10 minutes. Try again shortly." },
      },
    });

    await user.click(screen.getByRole("button", { name: "Approve and post" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("10 minutes");
  });
});
