"""Re-encrypt every stored GitHub token under the first key in FORUM_KEY_ENCRYPTION_KEY.

Put the new key first and keep the old one after it, run this, then remove the old key:

    uv run python -m owl_mind.forum.rotate_keys
"""

import asyncio

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.core.config import get_settings
from owl_mind.forum.accounts import PostgresAccountStore, TokenCipher, rotate_keys
from owl_mind.services.models import sqlalchemy_url


async def main() -> None:
    settings = get_settings()
    if settings.forum_key_encryption_key is None:
        raise SystemExit("FORUM_KEY_ENCRYPTION_KEY isn't set.")
    cipher = TokenCipher.from_setting(settings.forum_key_encryption_key.get_secret_value())
    engine = create_async_engine(sqlalchemy_url(settings.database_url))
    try:
        store = PostgresAccountStore(async_sessionmaker(engine, expire_on_commit=False))
        result = await rotate_keys(store, cipher)
        print(f"Re-encrypted {result.rotated} accounts.")
        if result.unreadable:
            print(f"{result.unreadable} accounts were under an unknown key, and must reconnect.")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
