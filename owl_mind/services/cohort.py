"""Turning live agent runs on one cohort at a time (M4-G).

AGENT_COHORT lists who may get agent answers: emails, "@domain" for a whole domain, and
"invite:<code>" for everyone who signed up with that invite code. Empty means everyone. Others
get the "answers are switched off" reply, as when agent runs are off for all.
"""

import hashlib
import hmac

from owl_mind.services.accounts import Account


def invite_hash(code: str) -> str:
    """Kept on the account instead of the code, which is a secret."""
    return hashlib.sha256(code.strip().encode()).hexdigest()


def allowed(account: Account | None, cohort: str) -> bool:
    entries = [entry.strip() for entry in cohort.split(",") if entry.strip()]
    if not entries:
        return True
    if account is None:
        return False
    email = account.email.lower()
    for entry in entries:
        if entry.lower().startswith("invite:"):
            known = account.invite_hash
            if known and hmac.compare_digest(known, invite_hash(entry[len("invite:") :])):
                return True
        elif entry.startswith("@"):
            if email.endswith(entry.lower()):
                return True
        elif hmac.compare_digest(email.encode(), entry.lower().encode()):
            return True
    return False
