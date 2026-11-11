"""Prometheus metrics for runs, the queue, workers, jobs and spending (M4-E).

Labels never hold student data: no emails, IDs or text, only routes, kinds, statuses, steps
and component names. Each process (the API and every worker) serves its own /metrics, and
Prometheus scrapes them all.
"""

from prometheus_client import Counter, Gauge, Histogram

_SECONDS = (0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 200)

RUNS_STARTED = Counter("owl_runs_started_total", "Runs a worker or the API started")
RUNS_FINISHED = Counter(
    "owl_runs_finished_total",
    "Runs that ended, by how they ended",
    ["route", "task_kind", "status", "reason"],
)
RUN_SECONDS = Histogram(
    "owl_run_seconds", "Seconds from a run's start to its end", ["route", "task_kind"],
    buckets=_SECONDS,
)  # fmt: skip
RUNS_RUNNING = Gauge("owl_runs_running", "Runs this process is running now")

QUEUE_WAIT = Histogram(
    "owl_queue_wait_seconds", "Seconds a run waited in the queue before a worker took it",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300),
)  # fmt: skip
QUEUE_LAG = Gauge("owl_queue_lag", "Queued runs no worker has read yet")
QUEUE_PENDING = Gauge("owl_queue_pending", "Runs read by a worker and not yet acknowledged")
DEAD_LETTERS = Counter("owl_dead_letters_total", "Runs sent to the dead letters", ["why"])
TAKEOVERS = Counter("owl_takeovers_total", "Runs taken over from a stopped worker", ["step"])

WORKER_HEARTBEAT = Gauge("owl_worker_heartbeat_seconds", "When this worker last reported")
# job_name, not job: Prometheus sets its own job label on every series it scrapes.
JOB_RUNS = Counter("owl_job_runs_total", "Background job passes", ["job_name", "outcome"])
JOB_LAST_SUCCESS = Gauge("owl_job_last_success_seconds", "When a job last succeeded", ["job_name"])

BUDGET_STEPS = Counter("owl_budget_steps_total", "Steps of the budget ladder reached", ["step"])
INTEGRITY = Counter("owl_integrity_total", "Integrity check outcomes", ["result", "layer"])


def observe_event(kind: str, data: dict) -> None:
    """Counts the run events that are also metrics. Called for every event a run sends."""
    if kind == "integrity":
        INTEGRITY.labels(str(data.get("result")), str(data.get("layer"))).inc()
    elif kind == "budget_step":
        BUDGET_STEPS.labels(str(data.get("step"))).inc()
