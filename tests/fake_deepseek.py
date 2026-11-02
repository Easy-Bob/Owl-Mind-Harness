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
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        body = dict(completion(self.replies.pop(0), self.finish_reason), usage=self.usage)
        return httpx.Response(self.status, json=body)

    def attach(self, model) -> None:
        client = openai.OpenAI(
            api_key="not-a-real-key",
            base_url="https://api.deepseek.com",
            http_client=httpx.Client(transport=httpx.MockTransport(self)),
            max_retries=0,
        )
        model.root_client = client
        model.client = client.chat.completions
        async_client = openai.AsyncOpenAI(
            api_key="not-a-real-key",
            base_url="https://api.deepseek.com",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self)),
            max_retries=0,
        )
        model.root_async_client = async_client
        model.async_client = async_client.chat.completions
