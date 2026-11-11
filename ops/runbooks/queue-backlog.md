# Alert: queue backing up

**Means:** over 20 runs have waited unread for 5 minutes. Students are waiting for answers.

1. Runs and queue dashboard: are workers running runs (Runs running now)? Is the queue wait growing?
2. No workers running: start them ([worker-silent.md](worker-silent.md)).
3. Workers busy: add workers, or raise `WORKER_CONCURRENCY` within the database pool (`DB_POOL_SIZE`); see the M4-E load test results in `evals/load/`.
4. Model calls slow: check DeepSeek's status, and the latency panel on Spending.
