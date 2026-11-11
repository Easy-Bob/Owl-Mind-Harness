"""A stand-in for DeepSeek's chat API, for the load test (M4-E). Free and repeatable.

It answers after a set delay, the way a real call takes time, and follows one fixed learn turn:
the intent step says concept_explain, the orchestrator asks the concept tutor, the tutor
explains, the reviewer says it's a hint, and the orchestrator answers. Usage is reported, so
budgets and telemetry run as they would.

Usage: uv run python -m owl_mind.evaluation.fake_model --port 8099 --delay-ms 800
Point the API and workers at it with DEEPSEEK_BASE_URL=http://127.0.0.1:8099
"""

import argparse
import asyncio
import json
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from owl_mind.core import prompts
from owl_mind.core.integrity import REVIEWER_PROMPT

ANSWER = (
    "A Bloom filter is a bit array with k hash functions. Adding an item sets k bits; checking "
    "an item looks at the same k bits, and if any is 0 the item was never added. If all are 1 "
    "it is probably there, because other items may have set those bits. [cite:load-test]"
)
USAGE = {"prompt_tokens": 1200, "completion_tokens": 120, "total_tokens": 1320}


def _system(body: dict) -> str:
    content = (body.get("messages") or [{}])[0].get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)
    return content


def reply_for(body: dict) -> dict:
    """The message this fake gives for a request, by who is asking."""
    system, messages = _system(body), body.get("messages") or []
    if system.startswith("Classify a student's message"):
        intents = {"intents": [{"intent": "concept_explain", "confidence": 0.95}]}
        return {"role": "assistant", "content": json.dumps(intents)}
    if system == REVIEWER_PROMPT:
        return {"role": "assistant", "content": '{"verdict": "hint", "reason": "none"}'}
    if system.startswith(prompts.ORCHESTRATOR[:60]):
        if not any(m.get("role") == "tool" for m in messages):
            call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": "task",
                    "arguments": json.dumps(
                        {"description": "Explain Bloom filters.", "subagent_type": "concept_tutor"}
                    ),
                },
            }
            return {"role": "assistant", "content": None, "tool_calls": [call]}
        return {"role": "assistant", "content": ANSWER}
    return {"role": "assistant", "content": ANSWER}


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def build(delay_ms: int) -> FastAPI:
    app = FastAPI()

    @app.post("/chat/completions")
    async def complete(request: Request):
        body = await request.json()
        await asyncio.sleep(delay_ms / 1000)
        message = reply_for(body)
        if not body.get("stream"):
            return JSONResponse(
                {
                    "id": "fake",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "deepseek-flash",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                    "usage": USAGE,
                }
            )

        async def events():
            deltas: list[dict] = [{"role": "assistant", "content": ""}]
            text = message.get("content") or ""
            deltas += [{"content": text[i : i + 40]} for i in range(0, len(text), 40)]
            for index, call in enumerate(message.get("tool_calls") or []):
                deltas.append({"tool_calls": [dict(call, index=index)]})
            chunks = [_chunk(d) for d in deltas] + [_chunk({}, "stop")]
            chunks.append(dict(_chunk({}), choices=[], usage=USAGE))
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--delay-ms", type=int, default=800)
    args = parser.parse_args()
    uvicorn.run(build(args.delay_ms), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
