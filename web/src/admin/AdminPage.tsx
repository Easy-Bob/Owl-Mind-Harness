import { useCallback, useEffect, useState } from "react";
import {
  adminApi,
  ROUTES,
  type GroupRow,
  type ModelCallSummary,
  type RequestDetail,
  type RequestPage,
  type RunRow,
} from "./adminApi";
import { OverviewPage, SavingsPage, UsersPage } from "./AdminReports";

// Null means the provider or the run didn't report it. It must never read as 0.
const unknown = "unknown";
const count = (value: number | null | undefined) => (value == null ? unknown : value.toLocaleString("en-US"));
const usd = (value: number | null | undefined) => (value == null ? unknown : `$${value.toFixed(6)}`);
const time = (value: string) => new Date(value).toLocaleString("en-US");
const label = (value: string | null | undefined) => value ?? unknown;

const PAGE_SIZE = 25;
const words = (value: string) => value.replaceAll("_", " ");

// Events hold counts, IDs and statuses only. The server has already dropped anything else.
function describe(kind: string, data: Record<string, unknown>) {
  if (kind === "retrieval") {
    const score = data.top_score == null ? "no results" : `top score ${data.top_score}`;
    return `${data.result_count} results, ${score}, corpus version ${data.corpus_version}, ${
      data.cache_hit ? "cache hit" : "no cache hit"
    }, ${data.latency_ms} ms`;
  }
  if (kind === "budget_step" && typeof data.step === "string") {
    return `${words(data.step)} at ${data.percent}% of the run's token budget`;
  }
  if (kind === "handoff" && typeof data.step === "string") {
    const removed = data.removed as Record<string, number> | undefined;
    const details = [
      ...Object.entries(removed ?? {}).map(([what, n]) => `${n} ${what} removed`),
      ...Object.entries(data)
        .filter(([key]) => !["step", "removed", "url"].includes(key))
        .map(([key, value]) => `${words(key)}: ${typeof value === "string" ? words(value) : value}`),
    ];
    return (
      <>
        {words(data.step)}
        {details.length > 0 && `: ${details.join(", ")}`}
        {typeof data.url === "string" && (
          <>
            {" "}
            <a href={data.url}>forum post</a>
          </>
        )}
      </>
    );
  }
  return Object.entries(data)
    .map(([key, value]) => `${key}: ${value}`)
    .join(", ");
}

function RequestsPage() {
  const [query, setQuery] = useState("");
  const [applied, setApplied] = useState("");
  const [route, setRoute] = useState("");
  const [page, setPage] = useState(1);
  const [data, setData] = useState<RequestPage | null>(null);
  const [detail, setDetail] = useState<RequestDetail | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(() => {
    adminApi
      .requests(applied, page, PAGE_SIZE, route)
      .then(setData)
      .catch((caught: Error) => setError(caught.message));
  }, [applied, page, route]);

  useEffect(load, [load]);

  async function open(run: RunRow) {
    setError("");
    try {
      setDetail(await adminApi.request(run.id));
    } catch (caught) {
      setError((caught as Error).message);
    }
  }

  if (detail) {
    return <RequestTimeline detail={detail} onBack={() => setDetail(null)} />;
  }
  const lastPage = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;
  return (
    <section>
      <h2>Requests</h2>
      <form
        className="admin-search"
        onSubmit={(event) => {
          event.preventDefault();
          setPage(1);
          setApplied(query);
        }}
      >
        <label htmlFor="request-search">Search by email or run ID</label>
        <input id="request-search" value={query} onChange={(event) => setQuery(event.target.value)} />
        <button type="submit">Search</button>
        <label htmlFor="request-route">Route</label>
        <select
          id="request-route"
          value={route}
          onChange={(event) => {
            setPage(1);
            setRoute(event.target.value);
          }}
        >
          <option value="">All routes</option>
          {ROUTES.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </form>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {data && data.items.length === 0 && <p className="empty">No requests found.</p>}
      {data && data.items.length > 0 && (
        <>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>User</th>
                  <th>Route</th>
                  <th>Kind</th>
                  <th>Status</th>
                  <th>Steps</th>
                  <th>Model calls</th>
                  <th>Tokens</th>
                  <th>Cost</th>
                  <th>
                    <span className="sr-only">Open</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((run) => (
                  <tr key={run.id}>
                    <td>{time(run.created_at)}</td>
                    <td>{run.email}</td>
                    <td>{label(run.route)}</td>
                    <td>{run.task_kind ?? "none"}</td>
                    <td>{run.status}</td>
                    <td>{steps(run)}</td>
                    <td>{run.model_calls}</td>
                    <td>{usageCell(count(run.tokens), run.unknown_usage)}</td>
                    <td>{usageCell(run.model_calls ? usd(run.cost_usd) : "$0", run.unknown_usage)}</td>
                    <td>
                      <button type="button" onClick={() => open(run)}>
                        Open
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pager">
            <button type="button" disabled={page <= 1} onClick={() => setPage(page - 1)}>
              Previous
            </button>
            <span>
              Page {page} of {lastPage} ({data.total} requests)
            </span>
            <button type="button" disabled={page >= lastPage} onClick={() => setPage(page + 1)}>
              Next
            </button>
          </div>
        </>
      )}
    </section>
  );
}

function steps(run: RunRow) {
  const parts = [
    ...(run.searched ? ["searched"] : []),
    ...(run.handoff_step ? [`handoff: ${words(run.handoff_step)}`] : []),
  ];
  return parts.length > 0 ? parts.join(", ") : "none";
}

function usageCell(value: string, unknownUsage: number) {
  return unknownUsage > 0 ? `${value} (${unknownUsage} ${unknown})` : value;
}

function RequestTimeline({ detail, onBack }: { detail: RequestDetail; onBack: () => void }) {
  return (
    <section>
      <button type="button" className="link" onClick={onBack}>
        Back to requests
      </button>
      <h2>Request detail</h2>
      <p className="admin-note">Opening this page was recorded in the access log.</p>
      <dl className="facts">
        <dt>User</dt>
        <dd>{detail.email}</dd>
        <dt>Status</dt>
        <dd>{detail.status}</dd>
        <dt>Route</dt>
        <dd>{label(detail.route)}</dd>
        <dt>Kind</dt>
        <dd>{detail.task_kind ?? "none"}</dd>
        <dt>Started</dt>
        <dd>{time(detail.created_at)}</dd>
      </dl>
      <h3>Model calls</h3>
      {detail.calls.length === 0 ? (
        <p className="empty">This request made no recorded model calls.</p>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Component</th>
                <th>Model</th>
                <th>Input</th>
                <th>Cache hit</th>
                <th>Output</th>
                <th>Reasoning</th>
                <th>Cost</th>
                <th>Latency</th>
                <th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {detail.calls.map((call, index) => (
                <tr key={index}>
                  <td>{call.component}</td>
                  <td>{call.model}</td>
                  <td>{count(call.input_tokens)}</td>
                  <td>{count(call.cache_hit_tokens)}</td>
                  <td>{count(call.output_tokens)}</td>
                  <td>{count(call.reasoning_tokens)}</td>
                  <td>{usd(call.cost_usd)}</td>
                  <td>{call.latency_ms} ms</td>
                  <td>{call.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <h3>Timeline</h3>
      {detail.events.length === 0 ? (
        <p className="empty">No events were recorded.</p>
      ) : (
        <ol className="timeline">
          {detail.events.map((event, index) => (
            <li key={index}>
              <strong>{event.kind}</strong> {describe(event.kind, event.data)}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function GroupTable({ title, rows }: { title: string; rows: GroupRow[] }) {
  return (
    <>
      <h3>{title}</h3>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Calls</th>
              <th>Input tokens</th>
              <th>Output tokens</th>
              <th>Cost</th>
              <th>Cache hits</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name ?? "none"}>
                <td>{row.name ?? "none"}</td>
                <td>{row.calls}</td>
                <td>{usageCell(count(row.input_tokens), row.unknown_usage)}</td>
                <td>{usageCell(count(row.output_tokens), row.unknown_usage)}</td>
                <td>{usageCell(usd(row.cost_usd), row.unknown_usage)}</td>
                <td>{row.cache_hit_rate == null ? unknown : `${Math.round(row.cache_hit_rate * 100)}%`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function ModelCallsPage() {
  const [data, setData] = useState<ModelCallSummary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    adminApi
      .modelCalls(30)
      .then(setData)
      .catch((caught: Error) => setError(caught.message));
  }, []);

  return (
    <section>
      <h2>Model calls</h2>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {data && data.totals.calls === 0 && <p className="empty">No model calls in the last 30 days.</p>}
      {data && data.totals.calls > 0 && (
        <>
          <dl className="facts">
            <dt>Last {data.days} days</dt>
            <dd>{data.totals.calls.toLocaleString("en-US")} calls</dd>
            <dt>Cost</dt>
            <dd>{usageCell(usd(data.totals.cost_usd), data.totals.unknown_usage)}</dd>
            <dt>Cache hit rate</dt>
            <dd>{data.cache_hit_rate == null ? unknown : `${(data.cache_hit_rate * 100).toFixed(1)}%`}</dd>
            <dt>Outcomes</dt>
            <dd>{data.outcomes.map((row) => `${row.name}: ${row.calls}`).join(", ")}</dd>
          </dl>
          <GroupTable title="By component" rows={data.by_component} />
          <GroupTable title="By model" rows={data.by_model} />
          <GroupTable title="By kind" rows={data.by_task_kind} />
          <h3>Most expensive runs</h3>
          <ol className="timeline">
            {data.top_runs.map((run) => (
              <li key={run.run_id}>
                {run.run_id}: {usd(run.cost_usd)}, {run.calls} calls
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}

const PAGES = [
  ["overview", "Overview"],
  ["requests", "Requests"],
  ["users", "Users"],
  ["calls", "Model calls"],
  ["savings", "Savings"],
] as const;

export function AdminPage() {
  const [page, setPage] = useState<(typeof PAGES)[number][0]>("requests");
  return (
    <div className="admin">
      <nav className="admin-nav" aria-label="Admin pages">
        {PAGES.map(([id, name]) => (
          <button key={id} type="button" aria-current={page === id ? "page" : undefined} onClick={() => setPage(id)}>
            {name}
          </button>
        ))}
      </nav>
      <main className="admin-main">
        {page === "overview" && <OverviewPage />}
        {page === "requests" && <RequestsPage />}
        {page === "users" && <UsersPage />}
        {page === "calls" && <ModelCallsPage />}
        {page === "savings" && <SavingsPage />}
      </main>
    </div>
  );
}
