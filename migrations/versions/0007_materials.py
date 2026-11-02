"""Create the materials tables: documents, chunks with vectors and text search, import jobs.

Revision ID: 0007
Revises: 0006
"""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE materials.documents (
            id uuid PRIMARY KEY,
            title text NOT NULL,
            file_type text NOT NULL,
            content_hash text NOT NULL UNIQUE,
            source_url text,
            uploaded_by uuid NOT NULL,
            chunk_count integer NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    # Search by meaning uses the vector, and search by exact words uses the tsvector.
    op.execute(
        """
        CREATE TABLE materials.chunks (
            id uuid PRIMARY KEY,
            document_id uuid NOT NULL REFERENCES materials.documents (id) ON DELETE CASCADE,
            position integer NOT NULL,
            location text NOT NULL,
            heading text NOT NULL,
            text text NOT NULL,
            embedding vector(384) NOT NULL,
            search_text tsvector GENERATED ALWAYS AS (
                to_tsvector('english', heading || ' ' || text)
            ) STORED,
            UNIQUE (document_id, position)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_chunks_embedding ON materials.chunks"
        " USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("CREATE INDEX ix_chunks_search_text ON materials.chunks USING gin (search_text)")
    op.execute(
        """
        CREATE TABLE materials.ingest_jobs (
            id uuid PRIMARY KEY,
            filename text NOT NULL,
            status text NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
            reason text,
            document_id uuid,
            uploaded_by uuid NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE TABLE materials.corpus ("
        " id integer PRIMARY KEY CHECK (id = 1), version integer NOT NULL)"
    )


def downgrade() -> None:
    for table in ("corpus", "ingest_jobs", "chunks", "documents"):
        op.execute(f"DROP TABLE materials.{table}")
