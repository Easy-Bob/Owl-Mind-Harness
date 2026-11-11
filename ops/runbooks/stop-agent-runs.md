# Stop all agent runs at once

When answers are wrong, leaking, or costing too much:

1. Set `LIVE_AGENT_RUNS_ENABLED=false` and restart the API and workers. Students then get "Tutoring answers are switched off right now" for study questions; greetings, due topics and forum handoffs still work.
2. Runs already started finish or stop at their own limits. To stop them at once too, stop the workers; their runs are failed with "This answer was interrupted" when the workers come back, or resumed if under an hour old.
3. To narrow it to some students instead, set `AGENT_COHORT` (emails, `@domain`, or `invite:<code>`, comma-separated). Only those get agent answers.
4. To turn it back on, undo the setting and restart. Check the Quality page and the evals before widening the cohort again.
