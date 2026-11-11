# Changelog

## v1.0.0 (unreleased)

The first release for real students. Live agent runs stay limited to test users (`AGENT_COHORT`) until the school's privacy approval (Q19) is in place: that's the owner's call.

### Operations (M4)
- **Evals:** a hand-written corpus (216 intent cases, follow-ups, 20 dialogue turns, 40 attacks, memory sessions), live runners, a pinned baseline, and a CI check that fails on any drop past its margin or any leak (#56).
- **Workers:** runs and background jobs run in workers fed by Redis Streams, with takeover after a crash, dead letters after 3 deliveries, and one lease per job (#57).
- **Resending and resuming:** a resent message gets the first copy's run. A run a worker left is resumed from its checkpoint, without repeating the question or finished tutor calls (#58).
- **Savings, each behind the evals:** experiments on intent tuning, joined versus rewritten answers, forced delegation, a direct route for one-tutor turns, quiz reuse and the large model, with kept and rejected changes recorded with their numbers (#59).
- **Metrics:** runs, the queue, workers, jobs, budgets and integrity outcomes in Prometheus; three Grafana dashboards; a load test with a fake model (#60).
- **Admin Quality and Reliability pages:** eval results by prompt version, live quality figures, failed and stopped runs, workers, jobs and dead letters (#61).
- **Release operations:** alerts with runbooks, nightly retention (checkpoint pruning, 180-day chats, 400-day logs, partitions made ahead), account deletion, backups with a restore drill, and the cohort allowlist (#62).

### Tutoring (M3)
- Middleware checks around every agent step: budgets, tool errors, result caps, argument checks, fan-out limits and a context block (#39).
- Token budgets per run, per student per day and for the whole system, with a degrade ladder and summarization (#43).
- An integrity check on practice and quiz answers: a code layer, a reviewer, one rewrite and a fallback hint (#40).
- Approval cards before a plan is committed or progress is reset, resumed from the checkpoint (#41).
- Long-term memory: each student's profile and past sessions, searchable by meaning, which the student can read and delete (#42).
- Stable prompt beginnings for caching, a model tier table and delegation rules (#44).
- Admin Overview, Users and Savings pages (#45).
- Answer scope, citations shown as sources, course topics, and code-made multiple-choice quizzes (#50, #51).

### Course materials and the forum (M2)
- Course material import with chunking and embeddings; hybrid keyword and vector search with citations (#25, #26).
- A handoff to the GitHub Discussions forum with the student's approval, replies shown in the chat, connected GitHub accounts, and an optional rewrite (#27, #28, #29).
- Retrieval and handoff events in Admin (#30).

### The first turn (M0, M1)
- The scaffold, config and guard rails; Postgres and Redis with the LangGraph saver and store; the DeepSeek model door with thinking off; model call telemetry and prices (#3 to #6).
- The intent step, the agent factory with tutor subagents and study tools, the virtual filesystem and skills, the runtime with SSE streaming and run limits, signup and roles, the chat UI, run routes and Admin mode (#12 to #18).
