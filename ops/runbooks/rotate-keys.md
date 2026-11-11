# Rotate keys and secrets

| Secret | How | Effect |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | Make a new key at DeepSeek, set it, restart the API and workers, then revoke the old key. | None for students. |
| `JWT_SECRET` | Set a new random value of 32+ characters, restart the API. | Everyone is logged out. |
| `FORUM_KEY_ENCRYPTION_KEY` | Put the new key first and keep the old one after it, run `uv run python -m owl_mind.forum.rotate_keys`, then remove the old key and restart. | None. |
| GitHub App private key | Make a new key in the App's settings, point `FORUM_APP_PRIVATE_KEY_PATH` at it, restart, delete the old key in GitHub. | None. |
| `INVITE_CODES` | Replace the codes, restart the API. | New signups need a new code. Cohorts named by `invite:<code>` keep working for accounts already made. |
| Database password | `ALTER USER owl PASSWORD '...'`, update `DATABASE_URL`, restart everything. | A short outage. |

Never paste a secret into chat, an issue or a commit. `.env` is git-ignored.
