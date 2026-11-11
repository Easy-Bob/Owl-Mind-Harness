#!/usr/bin/env bash
# The load test (M4-E): a fake model, the API and workers, and simulated students.
#
# Everything runs against a database (owl_load) and a Redis database (5) of its own, so real
# accounts, chats and today's budget counters are never touched. The fake model is free.
#
# Usage: ops/load/run.sh <label>
#   WORKERS=2 CONCURRENCY=4 POOL=5 DELAY_MS=800 LEVELS=5,10,20,40 SECONDS_PER=150
# Writes evals/load/<date>-<label>.json
set -euo pipefail
cd "$(dirname "$0")/../.."

LABEL=${1:-default}
WORKERS=${WORKERS:-2}
CONCURRENCY=${CONCURRENCY:-4}
POOL=${POOL:-5}
DELAY_MS=${DELAY_MS:-800}
LEVELS=${LEVELS:-5,10,20,40}
SECONDS_PER=${SECONDS_PER:-150}
API_PORT=8020
LOGS=$(mktemp -d)

if ! docker compose exec -T postgres psql -U owl -d owl -tAc \
  "SELECT 1 FROM pg_database WHERE datname = 'owl_load'" | grep -q 1; then
  docker compose exec -T postgres createdb -U owl owl_load
fi

export DATABASE_URL=postgresql://owl:owl_dev@localhost:5433/owl_load
export REDIS_URL=redis://localhost:6380/5
export DEEPSEEK_BASE_URL=http://127.0.0.1:8099
export DEEPSEEK_API_KEY=not-a-real-key
export LIVE_AGENT_RUNS_ENABLED=true
export COOKIE_SECURE=false
export INVITE_CODES=load-test
export GLOBAL_DAILY_USD=100000
export STUDENT_DAILY_TOKENS=1000000000
export WORKER_CONCURRENCY=$CONCURRENCY
export DB_POOL_SIZE=$POOL
export FORUM_REPO=

uv run alembic upgrade head > /dev/null
docker compose exec -T redis redis-cli -n 5 FLUSHDB > /dev/null

pids=()
cleanup() { kill "${pids[@]}" 2> /dev/null || true; wait 2> /dev/null || true; }
trap cleanup EXIT

uv run python -m owl_mind.evaluation.fake_model --port 8099 --delay-ms "$DELAY_MS" \
  > "$LOGS/fake.log" 2>&1 &
pids+=($!)
uv run uvicorn owl_mind.api.app:app --port "$API_PORT" --log-level warning \
  > "$LOGS/api.log" 2>&1 &
pids+=($!)
targets=()
for n in $(seq 1 "$WORKERS"); do
  WORKER_METRICS_PORT=$((9200 + n)) uv run python -m owl_mind.worker.runs \
    > "$LOGS/worker-$n.log" 2>&1 &
  pids+=($!)
  targets+=("http://127.0.0.1:$((9200 + n))/metrics")
done

for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$API_PORT/auth/notice" > /dev/null && break
  sleep 1
done

uv run python -m owl_mind.evaluation.load_test --api "http://127.0.0.1:$API_PORT" \
  --levels "$LEVELS" --seconds "$SECONDS_PER" --workers "${targets[@]}" \
  --label "$LABEL workers=$WORKERS concurrency=$CONCURRENCY pool=$POOL delay=$DELAY_MS" \
  --out "evals/load/$(date +%F)-$LABEL.json"
echo "Logs: $LOGS"
