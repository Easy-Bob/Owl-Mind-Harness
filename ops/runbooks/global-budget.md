# Alert: global daily budget past 80%

**Means:** today's estimated spend passed 80% of `GLOBAL_DAILY_USD`. At 100%, every agent run stops until midnight UTC.

1. Admin, then Users: who is spending. Admin, then Overview: runs and calls today.
2. One student far above the rest: check their runs' timelines for loops; lower `STUDENT_DAILY_TOKENS` if needed.
3. Everyone up: a release may have raised cost per turn. Compare with the last eval results on the Quality page; roll back if so.
4. The estimate uses peak prices with no cache discount, so the real cost (in `ops.model_calls`) is usually lower. Raise `GLOBAL_DAILY_USD` only if the real cost allows.
