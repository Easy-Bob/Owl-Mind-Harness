"""The load test (M4-E): simulated students against the API and workers, with a fake model.

Each student signs up, logs in, then starts chats one after another and streams each answer
to the end, as the web app does. Levels of students run one after the other. Run it through
ops/load/run.sh, which starts the fake model, the API and the workers first.

Usage: uv run python -m owl_mind.evaluation.load_test --levels 5,10,20,40 --seconds 150 \
           --invite load-test --out evals/load/<name>.json
"""

import argparse
import asyncio
import json
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

QUESTIONS = [
    "How does a Bloom filter decide that an item is probably in the set?",
    "Explain linear probing in hash tables.",
    "What does Chebyshev's inequality say?",
    "Explain consistent hashing.",
]


async def _student(
    api: str, invite: str, until: float, n: int, samples: list[dict], errors: list[str]
) -> None:
    email = f"load-{uuid.uuid4().hex[:10]}@example.org"
    password = "load-test-password-1"
    async with httpx.AsyncClient(base_url=api, timeout=240) as client:
        signup = await client.post(
            "/auth/signup",
            json={"email": email, "password": password, "invite_code": invite,
                  "accepted_notice": True},
        )  # fmt: skip
        login = await client.post("/auth/login", json={"email": email, "password": password})
        if signup.status_code != 201 or login.status_code != 200:
            errors.append(f"login {signup.status_code}/{login.status_code}")
            return
        turn = 0
        while time.monotonic() < until:
            question = QUESTIONS[(n + turn) % len(QUESTIONS)]
            turn += 1
            sent = time.monotonic()
            started = await client.post(
                "/threads", json={"message": question, "client_message_id": str(uuid.uuid4())}
            )
            if started.status_code != 201:
                errors.append(f"start {started.status_code}")
                await asyncio.sleep(1)
                continue
            first = None
            status = "lost"
            async with client.stream("GET", f"/runs/{started.json()['run_id']}/stream") as stream:
                kind = None
                async for line in stream.aiter_lines():
                    if line.startswith("event: "):
                        kind = line[7:]
                        if kind == "text_delta" and first is None:
                            first = time.monotonic() - sent
                    elif line.startswith("data: ") and kind in ("done", "error"):
                        status = json.loads(line[6:]).get("status", "failed")
                        break
            samples.append(
                {"seconds": time.monotonic() - sent, "first_text": first, "status": status}
            )


def _scrape(url: str) -> dict[str, float]:
    """Queue-wait sum and count from a worker's /metrics, to average the wait per level."""
    try:
        text = httpx.get(url, timeout=5).text
    except httpx.HTTPError:
        return {}
    found = {}
    for name in ("owl_queue_wait_seconds_sum", "owl_queue_wait_seconds_count"):
        match = re.search(rf"^{name} ([0-9.e+]+)$", text, re.M)
        found[name] = float(match[1]) if match else 0.0
    return found


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)


async def level(api: str, invite: str, students: int, seconds: int, workers: list[str]):
    samples: list[dict] = []
    errors: list[str] = []
    before = [_scrape(w) for w in workers]
    started = time.monotonic()
    until = started + seconds
    await asyncio.gather(
        *[_student(api, invite, until, n, samples, errors) for n in range(students)]
    )
    elapsed = time.monotonic() - started
    after = [_scrape(w) for w in workers]
    waits = sum(a.get("owl_queue_wait_seconds_sum", 0) - b.get("owl_queue_wait_seconds_sum", 0)
                for a, b in zip(after, before, strict=True))  # fmt: skip
    counted = sum(
        a.get("owl_queue_wait_seconds_count", 0) - b.get("owl_queue_wait_seconds_count", 0)
        for a, b in zip(after, before, strict=True)
    )
    done = [s for s in samples if s["status"] == "completed"]
    seconds_all = [s["seconds"] for s in done]
    firsts = [s["first_text"] for s in done if s["first_text"] is not None]
    return {
        "students": students,
        "seconds": round(elapsed),
        "turns": len(samples),
        "completed": len(done),
        "failed": len(samples) - len(done),
        "errors": len(errors),
        "error_kinds": sorted(set(errors)),
        "turns_per_minute": round(len(done) / elapsed * 60, 1),
        "turn_seconds_p50": _quantile(seconds_all, 0.5),
        "turn_seconds_p95": _quantile(seconds_all, 0.95),
        "first_text_p95": _quantile(firsts, 0.95),
        "queue_wait_mean": round(waits / counted, 3) if counted else None,
        "turn_seconds_mean": round(statistics.mean(seconds_all), 2) if seconds_all else None,
    }


async def run(args) -> dict[str, Any]:
    levels = []
    for students in [int(n) for n in args.levels.split(",")]:
        result = await level(args.api, args.invite, students, args.seconds, args.workers)
        print(json.dumps(result))
        levels.append(result)
    return {"settings": vars(args) | {"workers": len(args.workers)}, "levels": levels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8020")
    parser.add_argument("--invite", default="load-test")
    parser.add_argument("--levels", default="5,10,20,40")
    parser.add_argument("--seconds", type=int, default=150)
    parser.add_argument("--workers", nargs="*", default=["http://127.0.0.1:9101/metrics"])
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = asyncio.run(run(args))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
