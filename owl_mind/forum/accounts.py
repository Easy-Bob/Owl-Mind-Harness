"""A student's own GitHub account, connected through the GitHub App's user authorization.

A connected token can act as the student on GitHub, within the app's Discussions permission, so
it is stored only encrypted and never logged, shown or returned by any route.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_TIMEOUT_SECONDS = 15
# Refresh a little early, so a token can't expire between the check and the post.
_REFRESH_MARGIN = timedelta(minutes=5)

RECONNECT_MESSAGE = (
    "Your GitHub connection stopped working, so nothing was posted. Reconnect your account,"
    " or disconnect it to post as the Owl-Mind bot."
)


class ReconnectNeeded(Exception):
    """The student's token no longer works. Posting must not fall back to the bot silently."""


class OAuthFailed(Exception):
    """GitHub refused the code or refresh token, or couldn't be reached."""


class TokenCipher:
    """Encrypts with the first key, and decrypts with any, so keys can be rotated."""

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise ValueError("No encryption key")
        self._fernet = MultiFernet([Fernet(key.encode()) for key in keys])

    @classmethod
    def from_setting(cls, value: str) -> TokenCipher:
        return cls([key.strip() for key in value.split(",") if key.strip()])

    def encrypt(self, value: str) -> bytes:
        return self._fernet.encrypt(value.encode())

    def decrypt(self, value: bytes) -> str:
        return self._fernet.decrypt(bytes(value)).decode()

    def rotate(self, value: bytes) -> bytes:
        return self._fernet.rotate(bytes(value))


@dataclass(frozen=True)
class Tokens:
    access: str
    refresh: str | None
    access_expires_at: datetime | None
    refresh_expires_at: datetime | None


@dataclass(frozen=True)
class ForumAccount:
    student_id: uuid.UUID
    github_login: str
    access_token: bytes
    refresh_token: bytes | None
    access_expires_at: datetime | None
    refresh_expires_at: datetime | None
    needs_reconnect: bool = False


@runtime_checkable
class AccountStore(Protocol):
    async def get(self, student_id: uuid.UUID) -> ForumAccount | None: ...

    async def save(self, account: ForumAccount) -> None: ...

    async def delete(self, student_id: uuid.UUID) -> None: ...

    async def all(self) -> list[ForumAccount]: ...


class GitHubOAuth:
    """The GitHub App's user authorization: the web flow, token refresh and revocation."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_url = redirect_url
        self._clock = clock
        self._client = httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    def authorize_url(self, state: str) -> str:
        query = {"client_id": self._client_id, "redirect_uri": self._redirect_url, "state": state}
        return f"https://github.com/login/oauth/authorize?{urlencode(query)}"

    async def _token_request(self, fields: dict[str, str]) -> Tokens:
        payload = {"client_id": self._client_id, "client_secret": self._client_secret, **fields}
        try:
            response = await self._client.post(
                "https://github.com/login/oauth/access_token",
                data=payload,
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise OAuthFailed(type(exc).__name__) from exc
        data: dict[str, Any] = response.json() if response.status_code == 200 else {}
        # GitHub answers 200 with an "error" field for a bad code or refresh token.
        if "access_token" not in data:
            raise OAuthFailed(str(data.get("error", response.status_code)))
        now = self._clock()

        def expiry(field: str) -> datetime | None:
            return now + timedelta(seconds=int(data[field])) if data.get(field) else None

        return Tokens(
            data["access_token"],
            data.get("refresh_token"),
            expiry("expires_in"),
            expiry("refresh_token_expires_in"),
        )

    async def exchange(self, code: str) -> Tokens:
        return await self._token_request({"code": code, "redirect_uri": self._redirect_url})

    async def refresh(self, refresh_token: str) -> Tokens:
        return await self._token_request(
            {"grant_type": "refresh_token", "refresh_token": refresh_token}
        )

    async def login(self, access_token: str) -> str:
        try:
            response = await self._client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github+json",
                },
            )
        except httpx.HTTPError as exc:
            raise OAuthFailed(type(exc).__name__) from exc
        if response.status_code != 200:
            raise OAuthFailed(str(response.status_code))
        return response.json()["login"]

    async def revoke(self, access_token: str) -> None:
        """Best effort: the token is deleted here whether or not GitHub confirms."""
        try:
            await self._client.request(
                "DELETE",
                f"https://api.github.com/applications/{self._client_id}/token",
                auth=(self._client_id, self._client_secret),
                json={"access_token": access_token},
            )
        except httpx.HTTPError:
            pass


class ForumAccounts:
    def __init__(
        self,
        store: AccountStore,
        oauth: GitHubOAuth,
        cipher: TokenCipher,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.store = store
        self.oauth = oauth
        self.cipher = cipher
        self._clock = clock

    def _account(self, student_id: uuid.UUID, login: str, tokens: Tokens) -> ForumAccount:
        return ForumAccount(
            student_id,
            login,
            self.cipher.encrypt(tokens.access),
            self.cipher.encrypt(tokens.refresh) if tokens.refresh else None,
            tokens.access_expires_at,
            tokens.refresh_expires_at,
        )

    async def connect(self, student_id: uuid.UUID, code: str) -> str:
        tokens = await self.oauth.exchange(code)
        login = await self.oauth.login(tokens.access)
        await self.store.save(self._account(student_id, login, tokens))
        return login

    async def status(self, student_id: uuid.UUID) -> dict[str, Any]:
        account = await self.store.get(student_id)
        if account is None:
            return {"connected": False, "login": None, "needs_reconnect": False}
        return {
            "connected": True,
            "login": account.github_login,
            "needs_reconnect": account.needs_reconnect,
        }

    async def token_for(self, student_id: uuid.UUID) -> tuple[str, str] | None:
        """The student's token and login, or None to post as the bot. Refreshes when needed."""
        account = await self.store.get(student_id)
        if account is None:
            return None
        if account.needs_reconnect:
            raise ReconnectNeeded
        now = self._clock()
        expires = account.access_expires_at
        try:
            if expires is None or expires - now > _REFRESH_MARGIN:
                return self.cipher.decrypt(account.access_token), account.github_login
            refresh_ends = account.refresh_expires_at
            if account.refresh_token is None or (refresh_ends is not None and refresh_ends <= now):
                raise ReconnectNeeded
            tokens = await self.oauth.refresh(self.cipher.decrypt(account.refresh_token))
        except (OAuthFailed, InvalidToken, ReconnectNeeded) as exc:
            # InvalidToken: stored under a key that is no longer configured.
            await self.mark_broken(student_id)
            raise ReconnectNeeded from exc
        await self.store.save(self._account(student_id, account.github_login, tokens))
        return tokens.access, account.github_login

    async def mark_broken(self, student_id: uuid.UUID) -> None:
        account = await self.store.get(student_id)
        if account is not None:
            await self.store.save(replace(account, needs_reconnect=True))

    async def disconnect(self, student_id: uuid.UUID) -> None:
        account = await self.store.get(student_id)
        if account is None:
            return
        await self.store.delete(student_id)
        try:
            await self.oauth.revoke(self.cipher.decrypt(account.access_token))
        except InvalidToken:
            pass  # A token under a lost key can't be revoked, but it's deleted here.

    async def rotate_keys(self) -> Rotation:
        return await rotate_keys(self.store, self.cipher)


@dataclass(frozen=True)
class Rotation:
    rotated: int
    # Tokens under a key that is no longer listed. Those students must reconnect.
    unreadable: int


async def rotate_keys(store: AccountStore, cipher: TokenCipher) -> Rotation:
    """Re-encrypt every token under the first key. Run after adding a new key in front."""
    rotated = unreadable = 0
    for account in await store.all():
        try:
            refresh = cipher.rotate(account.refresh_token) if account.refresh_token else None
            access = cipher.rotate(account.access_token)
        except InvalidToken:
            unreadable += 1
            await store.save(replace(account, needs_reconnect=True))
            continue
        await store.save(replace(account, access_token=access, refresh_token=refresh))
        rotated += 1
    return Rotation(rotated, unreadable)


class InMemoryAccountStore:
    def __init__(self) -> None:
        self.accounts: dict[uuid.UUID, ForumAccount] = {}

    async def get(self, student_id) -> ForumAccount | None:
        return self.accounts.get(student_id)

    async def save(self, account: ForumAccount) -> None:
        self.accounts[account.student_id] = account

    async def delete(self, student_id) -> None:
        self.accounts.pop(student_id, None)

    async def all(self) -> list[ForumAccount]:
        return list(self.accounts.values())


_COLUMNS = (
    "student_id, github_login, access_token, refresh_token, access_expires_at,"
    " refresh_expires_at, needs_reconnect"
)


class PostgresAccountStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def _select(self, where: str, params: dict) -> list[ForumAccount]:
        async with self._sessions() as session:
            rows = await session.execute(
                text(f"SELECT {_COLUMNS} FROM app.forum_accounts {where}"), params
            )
        return [ForumAccount(**row._mapping) for row in rows]

    async def get(self, student_id) -> ForumAccount | None:
        found = await self._select("WHERE student_id = :id", {"id": student_id})
        return found[0] if found else None

    async def save(self, account: ForumAccount) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                text(
                    f"INSERT INTO app.forum_accounts ({_COLUMNS}) VALUES (:student_id,"
                    " :github_login, :access_token, :refresh_token, :access_expires_at,"
                    " :refresh_expires_at, :needs_reconnect)"
                    " ON CONFLICT (student_id) DO UPDATE SET github_login = :github_login,"
                    " access_token = :access_token, refresh_token = :refresh_token,"
                    " access_expires_at = :access_expires_at,"
                    " refresh_expires_at = :refresh_expires_at,"
                    " needs_reconnect = :needs_reconnect, updated_at = now()"
                ),
                vars(account),
            )

    async def delete(self, student_id) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                text("DELETE FROM app.forum_accounts WHERE student_id = :id"), {"id": student_id}
            )

    async def all(self) -> list[ForumAccount]:
        return await self._select("", {})
