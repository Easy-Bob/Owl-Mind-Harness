"""Stands in for DeepSeek at the network layer, so the real client and agent loop run."""

import json

import httpx
import openai

USAGE = {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15,
    "prompt_cache_hit_tokens": 0,
    "prompt_cache_miss_tokens": 10,
    "completion_tokens_details": {"reasoning_tokens": 0},
}


def completion(message: dict, finish_reason: str = "stop") -> dict:
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": USAGE,
    }


def tool_call(call_id: str, code_name: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "lookup_code", "arguments": json.dumps({"name": code_name})},
    }


class FakeDeepSeek:
    """Replays set replies and records every request body the client would send."""

    def __init__(
        self,
        replies: list[dict],
        usage: dict | None = USAGE,
        finish_reason: str = "stop",
        status: int = 200,
    ) -> None:
        self.replies = list(replies)
        self.usage = usage
        self.finish_reason = finish_reason
        self.status = status
        self.handler = self
        self.requests: list[dict] = []

    def _dispatch(self, request: httpx.Request):
        return self.handler(request)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if self.requests[-1].get("stream"):
            return self._stream(self.replies.pop(0))
        body = dict(completion(self.replies.pop(0), self.finish_reason), usage=self.usage)
        return httpx.Response(self.status, json=body)

    def attach(self, model) -> None:
        client = openai.OpenAI(
            api_key="not-a-real-key",
            base_url="https://api.deepseek.com",
            http_client=httpx.Client(transport=httpx.MockTransport(self._dispatch)),
            max_retries=0,
        )
        model.root_client = client
        model.client = client.chat.completions
        async_client = openai.AsyncOpenAI(
            api_key="not-a-real-key",
            base_url="https://api.deepseek.com",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._dispatch)),
            max_retries=0,
        )
        model.root_async_client = async_client
        model.async_client = async_client.chat.completions

    def _stream(self, message: dict) -> httpx.Response:
        """Reply in the server-sent format: text in small pieces, then usage in the last chunk."""
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "failed"}})
        deltas: list[dict] = [{"role": "assistant", "content": ""}]
        content = message.get("content") or ""
        deltas += [{"content": content[i : i + 8]} for i in range(0, len(content), 8)]
        for index, call in enumerate(message.get("tool_calls") or []):
            deltas.append({"tool_calls": [dict(call, index=index)]})
        chunks = [_chunk(delta) for delta in deltas]
        chunks.append(_chunk({}, finish_reason=self.finish_reason))
        if self.usage is not None:
            chunks.append(dict(_chunk({}), choices=[], usage=self.usage))
        lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks] + ["data: [DONE]\n\n"]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content="".join(lines)
        )


def _chunk(delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": "test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
