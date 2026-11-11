# Set up where alerts go

The alert rules are provisioned from `ops/grafana/provisioning/alerting/owl-alerts.yml` when Grafana starts. Where they're sent is the owner's choice:

1. Open Grafana (http://localhost:3002), then Alerting, then Contact points.
2. Edit the default contact point: email, Slack or a webhook.
3. Send a test notification.

Each alert's `runbook_url` opens its page here. The rules read the M4-E metrics, so Prometheus must scrape the API and every worker (`ops/prometheus/prometheus.yml`).
