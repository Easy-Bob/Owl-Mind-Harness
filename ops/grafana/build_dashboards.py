"""Writes the provisioned Grafana dashboards (M4-E). Run after changing a panel:

    uv run python ops/grafana/build_dashboards.py

The JSON files are committed, and Grafana loads them at start.
"""

import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
DATASOURCE = {"type": "prometheus", "uid": "prometheus"}


def panel(index: int, title: str, exprs: list[tuple[str, str]], unit: str = "short") -> dict:
    return {
        "id": index + 1,
        "title": title,
        "type": "timeseries",
        "datasource": DATASOURCE,
        "gridPos": {"h": 8, "w": 12, "x": 12 * (index % 2), "y": 8 * (index // 2)},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "targets": [
            {"refId": chr(65 + n), "expr": expr, "legendFormat": legend, "datasource": DATASOURCE}
            for n, (expr, legend) in enumerate(exprs)
        ],
    }


def dashboard(uid: str, title: str, panels: list[tuple]) -> dict:
    return {
        "uid": uid,
        "title": title,
        "tags": ["owl-mind"],
        "timezone": "utc",
        "schemaVersion": 39,
        "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "panels": [panel(i, *p) for i, p in enumerate(panels)],
    }


WAIT = "histogram_quantile({q}, sum by (le) (rate(owl_queue_wait_seconds_bucket[5m])))"
FINISHED = "sum by (status) (rate(owl_runs_finished_total[5m])) * 60"
DASHBOARDS = [
    dashboard(
        "owl-health",
        "Service health",
        [
            ("Runs finished per minute, by status", [(FINISHED, "{{status}}")]),
            (
                "Share of runs failed",
                [
                    (
                        'sum(rate(owl_runs_finished_total{status="failed"}[5m])) '
                        "/ clamp_min(sum(rate(owl_runs_finished_total[5m])), 1e-9)",
                        "failed",
                    )
                ],
                "percentunit",
            ),
            (
                "Seconds since each worker reported",
                [("time() - owl_worker_heartbeat_seconds", "{{instance}}")],
                "s",
            ),
            (
                "Seconds since each job last succeeded",
                [
                    (
                        "time() - max by (job_name) (label_replace(owl_job_last_success_seconds, "
                        '"job_name", "$1", "job", "(.*)"))',
                        "{{job_name}}",
                    )
                ],
                "s",
            ),
            (
                "Model call failures per minute, by component",
                [
                    (
                        'sum by (component) (rate(llm_calls_total{outcome!="success"}[5m])) * 60',
                        "{{component}}",
                    )
                ],
            ),
            (
                "Dead letters in the last hour",
                [("sum by (why) (increase(owl_dead_letters_total[1h]))", "{{why}}")],
            ),
        ],
    ),
    dashboard(
        "owl-runs",
        "Runs and queue",
        [
            (
                "Run seconds, 95th percentile, by kind",
                [
                    (
                        "histogram_quantile(0.95, sum by (le, task_kind) "
                        "(rate(owl_run_seconds_bucket[5m])))",
                        "{{task_kind}}",
                    )
                ],
                "s",
            ),
            (
                "Queue wait seconds",
                [
                    (
                        WAIT.format(q=0.5),
                        "p50",
                    ),
                    (
                        WAIT.format(q=0.95),
                        "p95",
                    ),
                ],
                "s",
            ),
            (
                "Queue depth",
                [("max(owl_queue_lag)", "not yet read"), ("max(owl_queue_pending)", "being run")],
            ),
            ("Runs running now", [("sum(owl_runs_running)", "running")]),
            (
                "Stopped runs per hour, by reason",
                [
                    (
                        'sum by (reason) (increase(owl_runs_finished_total{status="stopped"}[1h]))',
                        "{{reason}}",
                    )
                ],
            ),
            (
                "Takeovers per hour",
                [("sum by (step) (increase(owl_takeovers_total[1h]))", "{{step}}")],
            ),
        ],
    ),
    dashboard(
        "owl-spending",
        "Spending",
        [
            (
                "Tokens per minute, by component",
                [("sum by (component) (rate(llm_tokens_total[5m])) * 60", "{{component}}")],
            ),
            (
                "Model calls per minute, by component",
                [("sum by (component) (rate(llm_calls_total[5m])) * 60", "{{component}}")],
            ),
            (
                "Budget ladder steps per hour",
                [("sum by (step) (increase(owl_budget_steps_total[1h]))", "{{step}}")],
            ),
            (
                "Integrity outcomes per hour",
                [("sum by (result) (increase(owl_integrity_total[1h]))", "{{result}}")],
            ),
            (
                "Model call latency, 95th percentile, by component",
                [
                    (
                        "histogram_quantile(0.95, sum by (le, component) "
                        "(rate(llm_latency_ms_bucket[5m])))",
                        "{{component}}",
                    )
                ],
                "ms",
            ),
        ],
    ),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for board in DASHBOARDS:
        (OUT / f"{board['uid']}.json").write_text(json.dumps(board, indent=2) + "\n")


if __name__ == "__main__":
    main()
