# Alert: more than 5% of runs failing

**Means:** over 5% of runs ended `failed` in the last 10 minutes.

1. Admin, then Reliability: which days and reasons. Admin, then Requests: open a failed run's timeline.
2. Service health dashboard: are model calls failing (DeepSeek down or the key revoked)? Check https://status.deepseek.com.
3. API and worker logs: errors are logged by type only (never message text), so look for a repeated type.
4. Database or Redis down: `docker compose ps`.
5. If it can't be fixed quickly, stop agent runs ([stop-agent-runs.md](stop-agent-runs.md)).
