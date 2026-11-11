import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";

const admin = { email: "owner@example.com", role: "admin" };
const student = { email: "alice@example.com", role: "student" };

const run = {
  id: "r1",
  email: "alice@example.com",
  status: "completed",
  route: "orchestrated",
  task_kind: "learn",
  created_at: "2026-11-02T00:00:00Z",
  model_calls: 2,
  tokens: 30,
  cost_usd: 0.000123,
  unknown_usage: 0,
  searched: false,
  handoff_step: null,
};

const emptySummary = {
  days: 30,
  totals: { calls: 0, cost_usd: null, unknown_usage: 0 },
  cache_hit_rate: null,
  by_component: [],
  by_model: [],
  by_task_kind: [],
  outcomes: [],
  top_runs: [],
};

function server(routes: Record<string, unknown>, user = admin) {
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string) => {
      calls.push(path);
      if (path === "/auth/me") {
        return Response.json(user);
      }
      if (path === "/threads") {
        return Response.json([]);
      }
      // The longest match, so that a run's detail doesn't match the list.
      const key = Object.keys(routes)
        .sort((a, b) => b.length - a.length)
        .find((route) => path.startsWith(route));
      return key ? Response.json(routes[key]) : Response.json({ detail: "Missing" }, { status: 404 });
    }),
  );
  return calls;
}

async function openAdmin(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("button", { name: "Admin" }));
}

afterEach(() => vi.unstubAllGlobals());

describe("Admin mode", () => {
  it("shows no mode switch to a student, and never asks for admin data", async () => {
    const calls = server({}, student);

    render(<App />);
    await screen.findByLabelText("Your message");

    expect(screen.queryByRole("button", { name: "Admin" })).toBeNull();
    expect(calls.some((path) => path.startsWith("/admin"))).toBe(false);
  });

  it("switches an admin between Experience and Admin modes", async () => {
    server({ "/admin/requests": { items: [], total: 0, page: 1, size: 25 } });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);
    expect(await screen.findByRole("heading", { name: "Requests" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Chat" }));
    expect(await screen.findByLabelText("Your message")).toBeVisible();
  });

  it("shows the seven Admin pages", async () => {
    server({ "/admin/requests": { items: [], total: 0, page: 1, size: 25 } });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);

    const nav = await screen.findByRole("navigation", { name: "Admin pages" });
    expect(within(nav).getAllByRole("button").map((button) => button.textContent)).toEqual([
      "Overview",
      "Requests",
      "Users",
      "Model calls",
      "Savings",
      "Quality",
      "Reliability",
    ]);
  });

  const quality = {
    days: 30,
    evals: [
      {
        file: "2026-11-11-m4a.json",
        date: "2026-11-11",
        prompt_version: "m4a",
        large_model: "deepseek-flash",
        scores: {
          "intent.accuracy": 0.9923,
          "intent.follow_up_accuracy": 1,
          "dialogue.delegation_rate": 0.9474,
          "dialogue.citation_rate": null,
          "adversarial.leak_rate": 0,
          "dialogue.judge.correctness": 4.525,
          "dialogue.cost_per_turn_usd": 0.00272,
        },
      },
    ],
    live: [
      {
        prompt_version: "m4a",
        runs: 40,
        clarify_rate: 0.025,
        stopped_rate: null,
        learn_answers: 10,
        learn_citation_rate: 0.8,
        integrity_checked: 12,
        integrity_passed: 10,
        integrity_retried: 1,
        integrity_blocked: 1,
        integrity_blocked_rate: 1 / 12,
      },
    ],
  };

  it("shows eval results by prompt version and live quality, unknown where missing", async () => {
    server({ "/admin/requests": { items: [], total: 0, page: 1, size: 25 }, "/admin/quality": quality });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);
    await user.click(screen.getByRole("button", { name: "Quality" }));

    const page = await screen.findByRole("region", { name: "Quality" });
    const [evals, live] = within(page).getAllByRole("table");
    expect(within(evals).getByRole("row", { name: /m4a/ })).toHaveTextContent("99%");
    expect(within(evals).getByRole("row", { name: /m4a/ })).toHaveTextContent("unknown");
    expect(within(evals).getByRole("row", { name: /m4a/ })).toHaveTextContent("4.53");
    expect(within(live).getByRole("row", { name: /m4a/ })).toHaveTextContent("80% of 10");
    expect(within(live).getByRole("row", { name: /m4a/ })).toHaveTextContent("12 (1 retried)");
  });

  const reliability = {
    days: 14,
    trouble: [{ day: "2026-11-11", status: "stopped", reason: "time_limit", runs: 2 }],
    runs_per_day: [{ day: "2026-11-11", runs: 30 }],
    workers_used: true,
    workers: [
      { worker: "host-1", seconds_ago: 4 },
      { worker: "host-2", seconds_ago: 300 },
    ],
    jobs: [{ job: "approvals", success: Date.now() / 1000 - 30, failure: null }],
    dead_letters: [{ run_id: "r9", deliveries: 4, at: 1762862400 }],
    dead_letter_total: 1,
    queue: { lag: 3, pending: 1 },
  };

  it("shows troubled runs, workers, jobs and dead letters, and opens a dead run's timeline", async () => {
    const calls = server({
      "/admin/requests/r9": { ...run, id: "r9", events: [], calls: [] },
      "/admin/requests": { items: [], total: 0, page: 1, size: 25 },
      "/admin/reliability": reliability,
    });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);
    await user.click(screen.getByRole("button", { name: "Reliability" }));

    const page = await screen.findByRole("region", { name: "Reliability" });
    expect(page).toHaveTextContent("2026-11-11: 2 stopped, time limit");
    expect(page).toHaveTextContent("3 waiting, 1 being run");
    expect(page).toHaveTextContent("host-2: reported 300 s ago (not reporting)");
    expect(page).toHaveTextContent("approvals: last success 30 s ago, last failure never");
    await user.click(within(page).getByRole("button", { name: "r9" }));
    await waitFor(() => expect(calls).toContain("/admin/requests/r9"));
  });

  it("says when there are no workers to show", async () => {
    server({
      "/admin/requests": { items: [], total: 0, page: 1, size: 25 },
      "/admin/reliability": { days: 14, trouble: [], runs_per_day: [], workers_used: false },
    });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);
    await user.click(screen.getByRole("button", { name: "Reliability" }));

    expect(await screen.findByText(/no workers or queue to show/)).toBeVisible();
  });

  const window = {
    runs: [{ route: "orchestrated", status: "stopped", runs: 2 }],
    stopped: [{ reason: "token_limit", runs: 2 }],
    calls: { calls: 9, cost_usd: null, unknown_cost: 9, errors: 1, truncated: 0 },
  };

  it("shows the overview, with unknown where nothing was reported", async () => {
    server({ "/admin/requests": { items: [], total: 0, page: 1, size: 25 }, "/admin/overview": { today: window, last_7_days: window, global_budget_used: null } });
    const user = userEvent.setup();
    render(<App />);
    await openAdmin(user);
    await user.click(await screen.findByRole("button", { name: "Overview" }));

    const page = await screen.findByRole("region", { name: "Overview" });
    expect(page).toHaveTextContent("Everyone's daily budget used today: unknown");
    expect(page).toHaveTextContent("9 model calls, unknown (9 with unknown cost), 1 errors");
    expect(page).toHaveTextContent("stopped for token limit: 2");
  });

  it("lists users and opens one student's usage", async () => {
    const calls = server({
      "/admin/requests": { items: [], total: 0, page: 1, size: 25 },
      "/admin/users?": {
        users: [
          {
            student_id: "s1",
            email: "ana@example.com",
            runs_today: 1,
            runs_30d: 4,
            tokens_today: null,
            tokens_30d: 900,
            cost_today: null,
            cost_30d: 0.25,
            unknown_usage: 1,
            budget_used_today: 0.4,
          },
        ],
        daily_tokens: 1000000,
      },
      "/admin/users/s1": {
        student_id: "s1",
        email: "ana@example.com",
        budget_used_today: 0.4,
        days: [{ day: "2026-11-11", runs: 1, tokens: null, cost_usd: null, unknown_usage: 1 }],
        runs: [],
      },
    });
    const user = userEvent.setup();
    render(<App />);
    await openAdmin(user);
    await user.click(await screen.findByRole("button", { name: "Users" }));

    const row = (await screen.findByRole("cell", { name: "ana@example.com" })).closest("tr");
    expect(row).toHaveTextContent("unknown");
    expect(row).toHaveTextContent("40%");
    expect(calls.some((path) => path.startsWith("/admin/users/"))).toBe(false);
    await user.click(within(row as HTMLElement).getByRole("button", { name: "Open" }));

    const usage = await screen.findByRole("region", { name: "Student usage" });
    expect(usage).toHaveTextContent("2026-11-11: 1 runs, unknown tokens, unknown");
    expect(calls.filter((path) => path === "/admin/users/s1")).toHaveLength(1);
  });

  it("shows each saving with how it's worked out", async () => {
    server({
      "/admin/requests": { items: [], total: 0, page: 1, size: 25 },
      "/admin/savings": {
        days: 30,
        how: { cache: "Cache-hit tokens x (normal - cache price).", summaries: "s", ladder: "l", integrity: "i", joined: "j", no_agent: "n" },
        cache: { cache_hit_tokens: 750, cache_miss_tokens: 250, saved_usd: 0.0002, cache_hit_rate: 0.75 },
        summaries: { calls: 0, tokens_removed: null, cost_usd: null },
        ladder: { wind_down: 3 },
        integrity: { passed: 5, retried: 1 },
        joined: 2,
        runs: { without_agent: 4, with_agent: 6 },
      },
    });
    const user = userEvent.setup();
    render(<App />);
    await openAdmin(user);
    await user.click(await screen.findByRole("button", { name: "Savings" }));

    const page = await screen.findByRole("region", { name: "Savings" });
    expect(page).toHaveTextContent("75% of input tokens cached, saving about $0.000200.");
    expect(page).toHaveTextContent("Cache-hit tokens x (normal - cache price).");
    expect(page).toHaveTextContent("0 summaries removed about unknown tokens");
    expect(page).toHaveTextContent("wind down: 3");
    expect(page).toHaveTextContent("4 of 10 runs.");
  });

  it("shows empty states on both pages", async () => {
    server({
      "/admin/requests": { items: [], total: 0, page: 1, size: 25 },
      "/admin/model-calls": emptySummary,
    });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);
    expect(await screen.findByText("No requests found.")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Model calls" }));
    expect(await screen.findByText("No model calls in the last 30 days.")).toBeVisible();
  });

  it("searches and pages through requests", async () => {
    const calls = server({ "/admin/requests": { items: [run], total: 30, page: 1, size: 25 } });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);
    await user.type(await screen.findByLabelText("Search by email or run ID"), "alice");
    await user.click(screen.getByRole("button", { name: "Search" }));
    await user.click(await screen.findByRole("button", { name: "Next" }));

    expect(calls).toContain("/admin/requests?q=alice&page=1&size=25&route=");
    expect(calls).toContain("/admin/requests?q=alice&page=2&size=25&route=");
  });

  it("shows unknown values as unknown, never as 0", async () => {
    server({
      "/admin/requests": {
        items: [{ ...run, tokens: null, cost_usd: null, unknown_usage: 1 }],
        total: 1,
        page: 1,
        size: 25,
      },
      "/admin/requests/r1": {
        ...run,
        calls: [
          {
            called_at: "2026-11-02T00:00:00Z",
            component: "intent",
            model: "deepseek-flash",
            input_tokens: null,
            output_tokens: null,
            cache_hit_tokens: null,
            reasoning_tokens: null,
            cost_usd: null,
            latency_ms: 5,
            outcome: "success",
          },
        ],
        events: [],
      },
    });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);
    expect(await screen.findAllByText("unknown (1 unknown)")).toHaveLength(2);
    await user.click(screen.getByRole("button", { name: "Open" }));

    const row = (await screen.findByRole("cell", { name: "intent" })).closest("tr")!;
    expect(within(row).getAllByText("unknown")).toHaveLength(5);
    expect(within(row).queryByText("0")).toBeNull();
  });

  it("shows the timeline and says the view was logged", async () => {
    server({
      "/admin/requests": { items: [run], total: 1, page: 1, size: 25 },
      "/admin/requests/r1": {
        ...run,
        calls: [],
        events: [{ kind: "tool_start", at: "2026-11-02T00:00:00Z", data: { name: "build_hint" } }],
      },
    });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);
    await user.click(await screen.findByRole("button", { name: "Open" }));

    expect(await screen.findByText(/recorded in the access log/)).toBeVisible();
    expect(screen.getByText(/name: build_hint/)).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Back to requests" }));
    expect(await screen.findByRole("heading", { name: "Requests" })).toBeVisible();
  });

  it("shows a refused admin request as an error", async () => {
    server({});
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);

    expect(await screen.findByRole("alert")).toHaveTextContent("Missing");
  });

  it("shows the model call summary", async () => {
    server({
      "/admin/requests": { items: [], total: 0, page: 1, size: 25 },
      "/admin/model-calls": {
        ...emptySummary,
        totals: { calls: 3, cost_usd: 0.5, unknown_usage: 0 },
        cache_hit_rate: 0.25,
        by_component: [
          {
            name: "intent",
            calls: 3,
            input_tokens: 30,
            output_tokens: 6,
            cost_usd: 0.5,
            unknown_usage: 0,
            cache_hit_rate: 0.8,
          },
          {
            name: "summarization",
            calls: 1,
            input_tokens: null,
            output_tokens: null,
            cost_usd: null,
            unknown_usage: 1,
            cache_hit_rate: null,
          },
        ],
        outcomes: [{ name: "success", calls: 3 }],
        top_runs: [{ run_id: "r1", cost_usd: 0.5, calls: 3 }],
      },
    });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);
    await user.click(await screen.findByRole("button", { name: "Model calls" }));

    expect(await screen.findByText("25.0%")).toBeVisible();
    expect(screen.getByText("success: 3")).toBeVisible();
    expect(screen.getByRole("cell", { name: "intent" })).toBeVisible();
    // Cache hits per component, and unknown rather than 0 when nothing reported them.
    const intentRow = screen.getByRole("cell", { name: "intent" }).closest("tr");
    expect(intentRow).toHaveTextContent("80%");
    const summaryRow = screen.getByRole("cell", { name: "summarization" }).closest("tr");
    expect(summaryRow).toHaveTextContent("unknown");
    expect(screen.getByText(/r1: \$0\.500000, 3 calls/)).toBeVisible();
  });

  it("filters by route and marks runs that searched or handed off", async () => {
    const handoff = { ...run, id: "r2", route: "handoff", searched: false, handoff_step: "replies_arrived" };
    const calls = server({
      "/admin/requests": { items: [{ ...run, searched: true }, handoff], total: 2, page: 1, size: 25 },
    });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);
    await user.selectOptions(await screen.findByLabelText("Route"), "handoff");

    expect(calls).toContain("/admin/requests?q=&page=1&size=25&route=handoff");
    expect(await screen.findByText("searched")).toBeVisible();
    expect(screen.getByText("handoff: replies arrived")).toBeVisible();
  });

  it("shows retrieval and handoff steps in the timeline, with the post as a link", async () => {
    server({
      "/admin/requests": { items: [run], total: 1, page: 1, size: 25 },
      "/admin/requests/r1": {
        ...run,
        calls: [],
        events: [
          {
            kind: "retrieval",
            at: "2026-11-02T00:00:01Z",
            data: { result_count: 3, top_score: 0.032, corpus_version: 7, cache_hit: true, latency_ms: 9 },
          },
          { kind: "handoff", at: "2026-11-02T00:00:02Z", data: { step: "drafted", removed: { email: 2 } } },
          {
            kind: "handoff",
            at: "2026-11-02T00:00:03Z",
            data: { step: "post_failed", attempts: 1, status: "retrying" },
          },
          {
            kind: "handoff",
            at: "2026-11-02T00:00:04Z",
            data: { step: "posted", url: "https://github.com/o/f/discussions/3", posted_as: "bot" },
          },
          { kind: "budget_step", at: "2026-11-02T00:00:05Z", data: { step: "wind_down", percent: 72 } },
        ],
      },
    });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);
    await user.click(await screen.findByRole("button", { name: "Open" }));
    const items = (await screen.findAllByRole("listitem")).map((item) => item.textContent);

    expect(items).toContain("retrieval 3 results, top score 0.032, corpus version 7, cache hit, 9 ms");
    expect(items).toContain("handoff drafted: 2 email removed");
    expect(items).toContain("handoff post failed: attempts: 1, status: retrying");
    expect(items).toContain("handoff posted: posted as: bot forum post");
    expect(items).toContain("budget_step wind down at 72% of the run's token budget");
    expect(screen.getByRole("link", { name: "forum post" })).toHaveAttribute(
      "href",
      "https://github.com/o/f/discussions/3",
    );
  });
});
