import { render, screen, within } from "@testing-library/react";
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

  it("shows only the two pages that exist", async () => {
    server({ "/admin/requests": { items: [], total: 0, page: 1, size: 25 } });
    const user = userEvent.setup();
    render(<App />);

    await openAdmin(user);

    const nav = await screen.findByRole("navigation", { name: "Admin pages" });
    expect(within(nav).getAllByRole("button").map((button) => button.textContent)).toEqual([
      "Requests",
      "Model calls",
    ]);
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

    expect(calls).toContain("/admin/requests?q=alice&page=1&size=25");
    expect(calls).toContain("/admin/requests?q=alice&page=2&size=25");
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
          { name: "intent", calls: 3, input_tokens: 30, output_tokens: 6, cost_usd: 0.5, unknown_usage: 0 },
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
    expect(screen.getByText(/r1: \$0\.500000, 3 calls/)).toBeVisible();
  });
});
