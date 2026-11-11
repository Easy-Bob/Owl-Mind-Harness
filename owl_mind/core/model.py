"""The only module that builds chat models."""

from enum import StrEnum

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from owl_mind.core.config import ConfigError, get_settings
from owl_mind.core.telemetry import TokenAccounting


class ModelTier(StrEnum):
    SMALL = "small"
    LARGE = "large"


# Which tier each component uses. get_chat_model refuses any other, so a call can't drift to a
# costlier or weaker model unnoticed. An "agent:" component is any tutor.
COMPONENT_TIERS: dict[str, ModelTier] = {
    "intent": ModelTier.SMALL,
    "intent:layer3": ModelTier.LARGE,
    "orchestrator": ModelTier.SMALL,
    "orchestrator:composite": ModelTier.LARGE,
    "agent:": ModelTier.SMALL,
    "summarization": ModelTier.SMALL,
    "integrity_reviewer": ModelTier.SMALL,
    "memory:write": ModelTier.SMALL,
    "materials:topics": ModelTier.SMALL,
    "handoff:improve": ModelTier.SMALL,
    # Evaluation only: the judge of the adversarial set.
    "integrity_judge": ModelTier.LARGE,
}


def tier_for(component: str) -> ModelTier:
    key = "agent:" if component.startswith("agent:") else component
    try:
        return COMPONENT_TIERS[key]
    except KeyError:
        raise ValueError(f"No tier is set for component {component!r}") from None


def _model_ids() -> dict[ModelTier, str]:
    # LARGE stays Flash until the owner approves a model from the M3-F comparison.
    return {ModelTier.SMALL: "deepseek-flash", ModelTier.LARGE: get_settings().large_model}


# Thinking is on by default at DeepSeek, so every model must turn it off explicitly (Q2).
_THINKING_OFF = {"thinking": {"type": "disabled"}}


def get_chat_model(
    tier: ModelTier | str, *, component: str, max_tokens: int | None = None
) -> BaseChatModel:
    """Build a chat model. component names who spends the tokens in ops.model_calls."""
    try:
        tier = ModelTier(tier)
    except ValueError:
        known = ", ".join(ModelTier)
        raise ValueError(f"Unknown model tier {tier!r}. Known tiers: {known}") from None
    if tier is not tier_for(component):
        raise ValueError(f"{component} uses the {tier_for(component).value} tier, not {tier.value}")
    model_id = _model_ids()[tier]

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
        # Streamed replies only report usage when asked, and telemetry needs it.
        stream_usage=True,
        callbacks=[TokenAccounting(component=component, model=model_id)],
    )
