import { useEffect, useState } from "react";
import { adminApi, type Overview, type Savings, type UserRow, type UserUsage } from "./adminApi";

// Null means nothing reported it. It must never read as 0.
const unknown = "unknown";
const count = (value: number | null | undefined) => (value == null ? unknown : value.toLocaleString("en-US"));
const usd = (value: number | null | undefined) => (value == null ? unknown : `$${value.toFixed(6)}`);
const percent = (value: number | null | undefined) => (value == null ? unknown : `${Math.round(value * 100)}%`);
const words = (value: string) => value.replaceAll("_", " ");

function useLoad<T>(load: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let current = true;
    load()
      .then((loaded) => current && setData(loaded))
      .catch((caught) => current && setError(caught instanceof Error ? caught.message : "Something went wrong."));
    return () => {
      current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error };
}

function Window({ title, data }: { title: string; data: Overview["today"] }) {
  return (
    <>
      <h3>{title}</h3>
      <p>
        {count(data.calls.calls)} model calls, {usd(data.calls.cost_usd)}
        {data.calls.unknown_cost > 0 && ` (${data.calls.unknown_cost} with unknown cost)`},{" "}
        {data.calls.errors} errors, {data.calls.truncated} cut off
      </p>
      <ul>
        {data.runs.map((row) => (
          <li key={`${row.route}-${row.status}`}>
            {row.route} runs {row.status}: {row.runs}
          </li>
        ))}
        {data.stopped.map((row) => (
          <li key={row.reason}>
            stopped for {words(row.reason)}: {row.runs}
          </li>
        ))}
      </ul>
    </>
  );
}

export function OverviewPage() {
  const { data, error } = useLoad(adminApi.overview);
  return (
    <section aria-label="Overview">
      <h2>Overview</h2>
      {error && <p role="alert">{error}</p>}
      {data && (
        <>
          <p>Everyone's daily budget used today: {percent(data.global_budget_used)}</p>
          <Window title="Today" data={data.today} />
          <Window title="Last 7 days" data={data.last_7_days} />
        </>
      )}
    </section>
  );
}

function UserDetail({ studentId }: { studentId: string }) {
  // Opening this writes one row to the access log, on the server.
  const { data, error } = useLoad<UserUsage>(() => adminApi.user(studentId), [studentId]);
  if (error) {
    return <p role="alert">{error}</p>;
  }
  if (!data) {
    return null;
  }
  return (
    <section aria-label="Student usage">
      <h3>{data.email}</h3>
      <p>Daily budget used today: {percent(data.budget_used_today)}</p>
      <ul>
        {data.days.map((day) => (
          <li key={day.day}>
            {day.day}: {day.runs} runs, {count(day.tokens)} tokens, {usd(day.cost_usd)}
          </li>
        ))}
      </ul>
      <ol className="timeline">
        {data.runs.map((run) => (
          <li key={run.run_id}>
            {new Date(run.created_at).toLocaleString("en-US")} {run.route ?? unknown}/{run.task_kind ?? "none"}:{" "}
            {run.status}
          </li>
        ))}
      </ol>
    </section>
  );
}

export function UsersPage() {
  const [query, setQuery] = useState("");
  const [applied, setApplied] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const { data, error } = useLoad(() => adminApi.users(applied), [applied]);
  return (
    <section aria-label="Users">
      <h2>Users</h2>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          setApplied(query);
        }}
      >
        <label htmlFor="user-search">Email</label>
        <input id="user-search" value={query} onChange={(event) => setQuery(event.target.value)} />
        <button type="submit">Search</button>
      </form>
      {error && <p role="alert">{error}</p>}
      {data && (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Email</th>
                <th>Runs today</th>
                <th>Runs, 30 days</th>
                <th>Tokens today</th>
                <th>Cost, 30 days</th>
                <th>Budget used today</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {data.users.map((row: UserRow) => (
                <tr key={row.student_id}>
                  <td>{row.email}</td>
                  <td>{row.runs_today}</td>
                  <td>{row.runs_30d}</td>
                  <td>{count(row.tokens_today)}</td>
                  <td>{usd(row.cost_30d)}</td>
                  <td>{percent(row.budget_used_today)}</td>
                  <td>
                    <button type="button" onClick={() => setOpen(row.student_id)}>
                      Open
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {open && <UserDetail studentId={open} />}
    </section>
  );
}

export function SavingsPage() {
  const { data, error } = useLoad<Savings>(() => adminApi.savings(30));
  return (
    <section aria-label="Savings">
      <h2>Savings, last 30 days</h2>
      {error && <p role="alert">{error}</p>}
      {data && (
        <dl className="savings">
          <dt>Prompt cache</dt>
          <dd>
            {percent(data.cache.cache_hit_rate)} of input tokens cached, saving about {usd(data.cache.saved_usd)}.{" "}
            <small>{data.how.cache}</small>
          </dd>
          <dt>Summaries</dt>
          <dd>
            {data.summaries.calls} summaries removed about {count(data.summaries.tokens_removed)} tokens of history, for{" "}
            {usd(data.summaries.cost_usd)}. <small>{data.how.summaries}</small>
          </dd>
          <dt>Budget step-downs</dt>
          <dd>
            {Object.entries(data.ladder).map(([step, n]) => `${words(step)}: ${n}`).join(", ") || "none"}.{" "}
            <small>{data.how.ladder}</small>
          </dd>
          <dt>Integrity check</dt>
          <dd>
            {Object.entries(data.integrity).map(([result, n]) => `${result}: ${n}`).join(", ") || "none"}.{" "}
            <small>{data.how.integrity}</small>
          </dd>
          <dt>Joined answers</dt>
          <dd>
            {data.joined}. <small>{data.how.joined}</small>
          </dd>
          <dt>No agent needed</dt>
          <dd>
            {data.runs.without_agent} of {data.runs.without_agent + data.runs.with_agent} runs.{" "}
            <small>{data.how.no_agent}</small>
          </dd>
        </dl>
      )}
    </section>
  );
}
