"""Index the parts of dotted names, so a search for heappush finds heapq.heappush.

Postgres keeps a dotted name as one word. Underscored names are already split.

Revision ID: 0010
Revises: 0009
"""

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None

# One place decides the extra words, for every import and for the rows already stored.
_SPLIT_FUNCTION = r"""
CREATE FUNCTION materials.dotted_name_parts(value text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE
RETURN coalesce(
    (
        SELECT string_agg(replace(match[1], '.', ' '), ' ')
        FROM regexp_matches(value, '([[:alnum:]_]+(?:\.[[:alnum:]_]+)+)', 'g') AS match
    ),
    ''
)
"""

_OLD_COLUMN = "to_tsvector('english', heading || ' ' || text)"
_NEW_COLUMN = (
    "to_tsvector('english', heading || ' ' || text || ' '"
    " || materials.dotted_name_parts(heading || ' ' || text))"
)


def _rebuild(expression: str) -> None:
    op.execute("DROP INDEX materials.ix_chunks_search_text")
    op.execute("ALTER TABLE materials.chunks DROP COLUMN search_text")
    op.execute(
        "ALTER TABLE materials.chunks ADD COLUMN search_text tsvector"
        f" GENERATED ALWAYS AS ({expression}) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_search_text ON materials.chunks USING gin (search_text)")


def upgrade() -> None:
    op.execute(_SPLIT_FUNCTION)
    _rebuild(_NEW_COLUMN)


def downgrade() -> None:
    _rebuild(_OLD_COLUMN)
    op.execute("DROP FUNCTION materials.dotted_name_parts(text)")
