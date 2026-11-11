# Alert: a worker stopped reporting

**Means:** no worker heartbeat for over a minute.

1. Admin, then Reliability: which workers reported, and when.
2. Is the process running? Check its log for the last error.
3. Restart it (`uv run python -m owl_mind.worker.runs`). Its unfinished run is taken over after 10 minutes and resumed from its checkpoint.
4. If all workers are down, runs queue up and no answer is written: see [queue-backlog.md](queue-backlog.md).
