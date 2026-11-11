# Alert: runs sent to the dead letters

**Means:** a run was delivered 3 times without finishing (each worker stopped while running it), or its queue entry was unreadable. The run was failed and the student asked to try again.

1. Admin, then Reliability: the dead-letter list. Open its run's timeline from there.
2. Look for what the run has in common with others: the same task kind, a crash in the logs at the same time.
3. Workers crashing on one kind of run is a bug: stop agent runs if it repeats ([stop-agent-runs.md](stop-agent-runs.md)).
4. The dead letters are kept in `stream:runs:dead` (up to 10,000). They hold the queued message, so they're never shown in Admin beyond the run ID.
