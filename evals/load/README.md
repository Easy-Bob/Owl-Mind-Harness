# M4-E load test results

`ops/load/run.sh <label>` runs a fake model (800 ms per call), the API and two workers on a database and Redis of their own, then simulated students who each sign up, start chats one after another and stream every answer to the end. Each level lasts 150 s. A turn is one learn turn: the intent step, the orchestrator, the concept tutor, the reviewer and the orchestrator's answer, so about 4.2 s of model time. Run on 2026-11-11 on one Mac, while a live eval ran beside it.

| Config | Students | Turns / min | Turn p50 | Turn p95 | Mean queue wait | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| A: 2 workers × 4, pool 5 (old default) | 5 | 70.8 | 4.2 s | 4.3 s | 0.01 s | 0 |
| | 10 | 110.6 | 4.3 s | 8.5 s | 1.0 s | 0 |
| | 20 | 111.0 | 8.6 s | 12.8 s | 6.2 s | 0 |
| | 40 | 112.6 | 21.1 s | 21.4 s | 15.8 s | 0 |
| B: 2 workers × 16, pool 5 | 5 | 70.2 | 4.3 s | 4.4 s | 0.00 s | 0 |
| | 10 | 140.5 | 4.3 s | 4.4 s | 0.00 s | 0 |
| | 20 | 281.8 | 4.2 s | 4.4 s | 0.00 s | 0 |
| | 40 | 442.2 | 4.5 s | 7.8 s | 1.0 s | 0 |
| C: 2 workers × 16, pool 12 | 40 | 441.2 | 4.5 s | 8.0 s | 1.0 s | 0 |

**What it shows:** a run spends nearly all its time waiting on the model, so 4 at once per worker left the workers idle while students queued. With 8 slots in all, throughput stopped at about 111 turns a minute and waits grew with every student. With 16 per worker (32 slots), 40 students kept turns near their model time, and the only queueing came from having more students than slots. A database pool of 5 per process was enough for 32 runs at once: 12 changed nothing.

**Tuned:**
- `WORKER_CONCURRENCY` 4 → 16.
- `DB_POOL_SIZE` stays 5.
- Redis timeouts unchanged: no timeout or error in 5,928 turns across the three configs.

Capacity for more students comes from more workers; at 16 slots each, a worker serves about 16 students sending back to back.

**Limits of this test:** the fake model answers in a fixed time and never fails or rate-limits, so DeepSeek's own limits and slow calls aren't covered, and every turn is the same learn turn.
