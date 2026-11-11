# Alert: a background job hasn't succeeded

**Means:** a job (approvals, resume, memory, forum) hasn't succeeded for 30 minutes, or retention for two days.

1. Admin, then Reliability: each job's last success and failure.
2. Worker logs: `<job> job failed: <ErrorType>`.
3. **memory:** usually the model or budget; summaries resume on their own. **forum:** GitHub access (see the forum settings). **retention:** the database; a failed pass is retried next night, and each step runs on its own. **approvals / resume:** the database or Redis.
4. A job runs under a Redis lease (`lease:<job>`), so only one worker runs it. A stuck lease expires by itself after three intervals.
