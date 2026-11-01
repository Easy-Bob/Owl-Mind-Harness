"""Create ops.model_calls and ops.model_prices, and seed DeepSeek prices.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

# USD per 1M tokens: cache hit, cache miss, output. Checked against
# https://api-docs.deepseek.com/quick_start/pricing/ on 2026-11-01.
PRICES = {
    ("deepseek-flash", "off_peak"): ("0.003", "0.15", "0.60"),
    ("deepseek-flash", "peak"): ("0.006", "0.30", "1.20"),
    ("deepseek-v4-pro", "off_peak"): ("0.022", "0.66", "1.98"),
    ("deepseek-v4-pro", "peak"): ("0.044", "1.32", "3.96"),
}
EFFECTIVE_FROM = "2026-11-01T00:00:00Z"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE ops.model_prices (
            model text NOT NULL,
            band text NOT NULL CHECK (band IN ('peak', 'off_peak')),
            effective_from timestamptz NOT NULL,
            cache_hit_usd_per_mtok numeric(12, 6) NOT NULL,
            cache_miss_usd_per_mtok numeric(12, 6) NOT NULL,
            output_usd_per_mtok numeric(12, 6) NOT NULL,
            PRIMARY KEY (model, band, effective_from)
        )
        """
    )
    for (model, band), (hit, miss, output) in PRICES.items():
        op.execute(
            "INSERT INTO ops.model_prices VALUES "
            f"('{model}', '{band}', '{EFFECTIVE_FROM}', {hit}, {miss}, {output})"
        )

    # NULL means unknown. A missing token count is never stored as 0.
    op.execute(
        """
        CREATE TABLE ops.model_calls (
            call_id uuid NOT NULL,
            called_at timestamptz NOT NULL,
            run_id uuid,
            component text NOT NULL,
            provider text NOT NULL,
            model text NOT NULL,
            input_tokens integer,
            output_tokens integer,
            cache_hit_tokens integer,
            cache_miss_tokens integer,
            reasoning_tokens integer,
            usage_complete boolean NOT NULL,
            cost_usd numeric(14, 8),
            latency_ms integer NOT NULL,
            outcome text NOT NULL CHECK (outcome IN ('success', 'error', 'truncated')),
            prompt_version text,
            PRIMARY KEY (call_id, called_at)
        ) PARTITION BY RANGE (called_at)
        """
    )
    # The default partition means an insert never fails for lack of a monthly partition.
    op.execute("CREATE TABLE ops.model_calls_default PARTITION OF ops.model_calls DEFAULT")
    op.execute(
        "CREATE TABLE ops.model_calls_2026_11 PARTITION OF ops.model_calls "
        "FOR VALUES FROM ('2026-11-01T00:00:00Z') TO ('2026-12-01T00:00:00Z')"
    )
    op.execute(
        "CREATE TABLE ops.model_calls_2026_12 PARTITION OF ops.model_calls "
        "FOR VALUES FROM ('2026-12-01T00:00:00Z') TO ('2027-01-01T00:00:00Z')"
    )
    op.execute("CREATE INDEX ix_model_calls_run_id ON ops.model_calls (run_id)")


def downgrade() -> None:
    op.execute("DROP TABLE ops.model_calls")
    op.execute("DROP TABLE ops.model_prices")
