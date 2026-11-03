"""The forum: GitHub Discussions in one private repository, reached through a GitHub App.

Only three operations exist here, so nothing else in GitHub's API can be reached through this code.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
import jwt

_API = "https://api.github.com"
_TIMEOUT_SECONDS = 15
_TOKEN_MARGIN_SECONDS = 300


class ForumUnavailable(Exception):
    """GitHub couldn't be reached or refused the request."""


class ForumAuthFailed(ForumUnavailable):
    """GitHub refused the token. For a student's own token, they need to reconnect."""


class DiscussionGone(Exception):
    """The discussion was deleted, or the app can no longer see it."""


@dataclass(frozen=True)
class ForumPost:
    url: str
    node_id: str


@dataclass(frozen=True)
class SimilarPost:
    title: str
    url: str


@dataclass(frozen=True)
class Reply:
    id: str
    author: str
    body: str
    created_at: datetime
    is_answer: bool


@dataclass(frozen=True)
class Replies:
    replies: list[Reply]
    # False when GitHub has more comments than one request returns, so a missing comment
    # may just be on a later page rather than deleted.
    complete: bool


@runtime_checkable
class Forum(Protocol):
    @property
    def home_url(self) -> str: ...

    async def search_similar(self, text: str, limit: int = 3) -> list[SimilarPost]: ...

    async def post_question(
        self, category: str, title: str, body: str, user_token: str | None = None
    ) -> ForumPost: ...

    async def get_replies(self, node_id: str) -> Replies: ...


_REPOSITORY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    id
    discussionCategories(first: 50) { nodes { id name } }
  }
}
"""

_SEARCH = """
query($query: String!, $limit: Int!) {
  search(type: DISCUSSION, query: $query, first: $limit) {
    nodes { ... on Discussion { title url } }
  }
}
"""

_CREATE = """
mutation($repository: ID!, $category: ID!, $title: String!, $body: String!) {
  createDiscussion(input: {
    repositoryId: $repository, categoryId: $category, title: $title, body: $body
  }) { discussion { id url } }
}
"""

_REPLIES = """
query($id: ID!) {
  node(id: $id) {
    ... on Discussion {
      answer { id }
      comments(first: 50) {
        totalCount
        nodes {
          id body createdAt author { login }
          replies(first: 20) { totalCount nodes { id body createdAt author { login } } }
        }
      }
    }
  }
}
"""


class GitHubForum:
    def __init__(
        self,
        repository: str,
        app_id: str,
        private_key_path: Path,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._owner, self._name = repository.split("/", 1)
        self._app_id = app_id
        self._key_path = private_key_path
        self._client = httpx.AsyncClient(
            base_url=_API, timeout=_TIMEOUT_SECONDS, transport=transport
        )
        self._token: tuple[str, float] | None = None
        self._ids: tuple[str, dict[str, str]] | None = None
        self._lock = asyncio.Lock()

    @property
    def home_url(self) -> str:
        return f"https://github.com/{self._owner}/{self._name}/discussions"

    async def aclose(self) -> None:
        await self._client.aclose()

    def _app_jwt(self) -> str:
        now = int(time.time())
        claims = {"iat": now - 60, "exp": now + 540, "iss": self._app_id}
        return jwt.encode(claims, self._key_path.read_text(), algorithm="RS256")

    async def _request(self, method: str, url: str, auth: str, **kwargs) -> Any:
        headers = {
            "Authorization": auth,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            response = await self._client.request(method, url, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise ForumUnavailable(type(exc).__name__) from exc
        if response.status_code == 401:
            raise ForumAuthFailed("GitHub refused the token")
        if response.status_code >= 400:
            raise ForumUnavailable(f"GitHub answered {response.status_code}")
        return response.json()

    async def _installation_token(self) -> str:
        """Tokens last an hour. A new one is fetched shortly before the old one expires."""
        async with self._lock:
            if self._token and self._token[1] - time.time() > _TOKEN_MARGIN_SECONDS:
                return self._token[0]
            app = f"Bearer {self._app_jwt()}"
            installation = await self._request(
                "GET", f"/repos/{self._owner}/{self._name}/installation", app
            )
            granted = await self._request(
                "POST", f"/app/installations/{installation['id']}/access_tokens", app
            )
            self._token = (granted["token"], time.time() + 3600)
            return self._token[0]

    async def _graphql(
        self, document: str, user_token: str | None = None, **variables
    ) -> dict[str, Any]:
        """Runs as the app, or as a student when given their own token."""
        auth = f"Bearer {user_token}" if user_token else f"token {await self._installation_token()}"
        data = await self._request(
            "POST", "/graphql", auth, json={"query": document, "variables": variables}
        )
        if data.get("errors"):
            kinds = {error.get("type") for error in data["errors"]}
            raise ForumUnavailable(f"GitHub refused the request: {sorted(map(str, kinds))}")
        return data["data"]

    async def _repository_ids(self) -> tuple[str, dict[str, str]]:
        if self._ids is None:
            data = await self._graphql(_REPOSITORY, owner=self._owner, name=self._name)
            repository = data["repository"]
            categories = {c["name"]: c["id"] for c in repository["discussionCategories"]["nodes"]}
            self._ids = (repository["id"], categories)
        return self._ids

    async def search_similar(self, text: str, limit: int = 3) -> list[SimilarPost]:
        words = " ".join(text.split()[:12])
        query = f"repo:{self._owner}/{self._name} {words}"
        data = await self._graphql(_SEARCH, query=query, limit=limit)
        return [SimilarPost(n["title"], n["url"]) for n in data["search"]["nodes"] if n]

    async def post_question(
        self, category: str, title: str, body: str, user_token: str | None = None
    ) -> ForumPost:
        repository_id, categories = await self._repository_ids()
        if category not in categories:
            raise ForumUnavailable(f"No category named {category}")
        data = await self._graphql(
            _CREATE,
            user_token,
            repository=repository_id,
            category=categories[category],
            title=title,
            body=body,
        )
        discussion = data["createDiscussion"]["discussion"]
        return ForumPost(url=discussion["url"], node_id=discussion["id"])

    async def get_replies(self, node_id: str) -> Replies:
        try:
            node = (await self._graphql(_REPLIES, id=node_id))["node"]
        except ForumAuthFailed:
            raise
        except ForumUnavailable as exc:
            # GitHub reports a deleted node as an error with NOT_FOUND.
            if "NOT_FOUND" in str(exc):
                raise DiscussionGone from exc
            raise
        if not node:
            raise DiscussionGone
        answer = (node.get("answer") or {}).get("id")
        comments = node["comments"]
        complete = comments["totalCount"] <= len(comments["nodes"])
        replies = []
        for comment in comments["nodes"]:
            replies.append(_reply(comment, answer))
            nested = comment.get("replies") or {"totalCount": 0, "nodes": []}
            complete = complete and nested["totalCount"] <= len(nested["nodes"])
            replies.extend(_reply(child, answer) for child in nested["nodes"])
        return Replies(replies, complete)


def _reply(comment: dict[str, Any], answer: str | None) -> Reply:
    # A deleted GitHub user shows as no author.
    author = (comment.get("author") or {}).get("login") or "ghost"
    created = datetime.fromisoformat(comment["createdAt"].replace("Z", "+00:00"))
    return Reply(comment["id"], author, comment["body"], created, comment["id"] == answer)
