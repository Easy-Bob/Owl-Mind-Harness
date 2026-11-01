"""The only module that builds chat models."""

from enum import StrEnum

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from owl_mind.core.config import ConfigError, get_settings
from owl_mind.core.telemetry import TokenAccounting


class ModelTier(StrEnum):
    SMALL = "small"
    LARGE = "large"


_MODEL_IDS = {
    ModelTier.SMALL: "deepseek-flash",
    # TODO: LARGE needs a better model than Flash. It uses Flash for now to limit cost.
    ModelTier.LARGE: "deepseek-flash",
}

# Thinking is on by default at DeepSeek, so every model must turn it off explicitly (Q2).
_THINKING_OFF = {"thinking": {"type": "disabled"}}


def get_chat_model(
    tier: ModelTier | str, *, component: str, max_tokens: int | None = None
) -> BaseChatModel:
    """Build a chat model. component names who spends the tokens in ops.model_calls."""
    try:
        model_id = _MODEL_IDS[ModelTier(tier)]
    except ValueError:
        known = ", ".join(ModelTier)
        raise ValueError(f"Unknown model tier {tier!r}. Known tiers: {known}") from None

    settings = get_settings()
    if settings.deepseek_api_key is None:
        raise ConfigError("Missing or invalid settings: DEEPSEEK_API_KEY")

    # Not the openai provider: it renames max_tokens to max_completion_tokens, which DeepSeek
    # does not document, so the cap may not apply.
    return init_chat_model(
        model_id,
        model_provider="deepseek",
        api_base=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key.get_secret_value(),
        max_tokens=max_tokens,
        extra_body=_THINKING_OFF,
        callbacks=[TokenAccounting(component=component, model=model_id)],
    )
