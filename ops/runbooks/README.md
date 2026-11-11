# Runbooks

One page per task and per alert. Each alert in `ops/grafana/provisioning/alerting/owl-alerts.yml` links to its page here.

| Task | Page |
| --- | --- |
| Deploy a release | [deploy.md](deploy.md) |
| Roll back a release | [rollback.md](rollback.md) |
| Restore from a backup | [restore.md](restore.md) |
| Rotate keys and secrets | [rotate-keys.md](rotate-keys.md) |
| Stop all agent runs at once | [stop-agent-runs.md](stop-agent-runs.md) |
| Set up where alerts go | [alerts.md](alerts.md) |

| Alert | Page |
| --- | --- |
| More than 5% of runs failing | [error-rate.md](error-rate.md) |
| Global daily budget past 80% | [global-budget.md](global-budget.md) |
| A worker stopped reporting | [worker-silent.md](worker-silent.md) |
| A background job hasn't succeeded | [job-stale.md](job-stale.md) |
| Runs sent to the dead letters | [dead-letters.md](dead-letters.md) |
| Queue backing up | [queue-backlog.md](queue-backlog.md) |
