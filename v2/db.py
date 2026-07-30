"""Database operations for conversation ingestion."""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import psycopg2
from psycopg2.extras import Json, execute_values

from .parser import ParsedConversation, ParsedMessage

logger = logging.getLogger(__name__)


class SourceFileCollision(Exception):
    """Raised when a file would overwrite a conversation ingested from a DIFFERENT file.

    This is the D1 guard. A subagent transcript inherits its parent's sessionId,
    so before the identity fix it resolved to the parent's conversation row and
    destroyed the parent's messages. Refusing loudly is always correct here:
    two distinct transcript files must never share one conversation row.
    """

# Default config — override via environment
DB_CONFIG = {
    "host": os.environ.get("CONVERSATIONS_DB_HOST", "100.127.104.75"),
    "port": int(os.environ.get("CONVERSATIONS_DB_PORT", "5432")),
    "dbname": os.environ.get("CONVERSATIONS_DB_NAME", "conversationsdb_v2"),
    "user": os.environ.get("CONVERSATIONS_DB_USER", "conversations_writer"),
    "password": os.environ.get("CONVERSATIONS_DB_PASSWORD", ""),
}

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    user_name TEXT NOT NULL,
    machine TEXT NOT NULL,
    project TEXT,
    model TEXT,
    source_file TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    last_message_at TIMESTAMPTZ,
    message_count INT DEFAULT 0,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    file_mtime TIMESTAMPTZ,
    source TEXT DEFAULT 'code',
    is_subagent BOOLEAN NOT NULL DEFAULT FALSE,
    parent_session_id TEXT
);

-- Additive migration for databases created before subagent support.
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS is_subagent BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS parent_session_id TEXT;

-- Search filters on subagent provenance; parent lookup joins on parent_session_id.
CREATE INDEX IF NOT EXISTS idx_conversations_is_subagent ON conversations(is_subagent);
CREATE INDEX IF NOT EXISTS idx_conversations_parent_session
    ON conversations(parent_session_id) WHERE parent_session_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    uuid TEXT,
    parent_uuid TEXT,
    role TEXT NOT NULL,
    content TEXT,
    thinking TEXT,
    tool_calls JSONB,
    tool_results JSONB,
    is_sidechain BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMPTZ,
    sequence_num INT
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_name);
CREATE INDEX IF NOT EXISTS idx_conversations_machine ON conversations(machine);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_conversations_unique
    ON conversations(session_id, machine, user_name);

-- Trigram index for LIKE '%keyword%' search
CREATE INDEX IF NOT EXISTS idx_messages_content_trgm
    ON messages USING gin(content gin_trgm_ops);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_messages_content_fts
    ON messages USING gin(to_tsvector('english', coalesce(content, '')));
"""


def get_connection():
    """Get a database connection."""
    return psycopg2.connect(**DB_CONFIG)


def ensure_schema(conn):
    """Create tables and indexes if they don't exist."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()
    logger.info("Schema ensured")


def _basename(path: Optional[str]) -> Optional[str]:
    return os.path.basename(str(path)) if path else None


def _assert_same_source(stored_source: str, incoming_source: str, session_id: str):
    """D1 guard: refuse to overwrite a conversation ingested from a different file.

    The destructive DELETE below is reachable only for a row whose stored
    source_file names the SAME transcript we are re-ingesting. A different
    transcript claiming the same session_id is the subagent-collision bug and
    must fail closed and loudly rather than silently replace real messages.

    A path that merely MOVED (project directory renamed, mirror root changed)
    keeps its basename, so legitimate re-ingest still succeeds.
    Non-.jsonl sources (e.g. web imports) are out of scope for this guard.
    """
    if stored_source == incoming_source:
        return

    old, new = _basename(stored_source), _basename(incoming_source)
    if old == new:
        return  # same transcript, relocated — allowed

    if not (str(stored_source).endswith(".jsonl") and str(incoming_source).endswith(".jsonl")):
        return  # not two transcript files — guard does not apply

    raise SourceFileCollision(
        f"REFUSING to overwrite conversation session_id={session_id!r}: "
        f"stored source_file={stored_source!r} but incoming file={incoming_source!r}. "
        f"Two distinct transcripts cannot share one conversation row."
    )


def upsert_conversation(
    conn,
    parsed: ParsedConversation,
    user_name: str,
    machine: str,
    source_file: str,
    file_mtime: Optional[datetime] = None,
    file_hash: Optional[str] = None,
    is_subagent: bool = False,
    parent_session_id: Optional[str] = None,
):
    """Insert or replace a conversation and all its messages.

    DELETE + INSERT in a transaction for idempotency.

    The DELETE is gated by _assert_same_source(): it can only ever remove
    messages that came from the very file being re-ingested.
    """
    with conn.cursor() as cur:
        # Find existing conversation
        cur.execute(
            "SELECT id, source_file FROM conversations WHERE session_id = %s AND machine = %s AND user_name = %s",
            (parsed.session_id, machine, user_name),
        )
        existing = cur.fetchone()

        if existing:
            conv_id, stored_source = existing
            # Fail closed BEFORE any destructive statement runs.
            _assert_same_source(stored_source, source_file, parsed.session_id)
            # Delete old messages (CASCADE would do this, but explicit is clearer)
            cur.execute("DELETE FROM messages WHERE conversation_id = %s", (conv_id,))
            # Update conversation metadata
            cur.execute(
                """UPDATE conversations SET
                    project = %s, model = %s, source_file = %s,
                    started_at = %s, last_message_at = %s,
                    message_count = %s, ingested_at = NOW(), file_mtime = %s, file_hash = %s,
                    is_subagent = %s, parent_session_id = %s
                WHERE id = %s""",
                (
                    parsed.project, parsed.model, source_file,
                    parsed.started_at, parsed.last_message_at,
                    parsed.message_count, file_mtime, file_hash,
                    is_subagent, parent_session_id, conv_id,
                ),
            )
        else:
            # Insert new conversation
            cur.execute(
                """INSERT INTO conversations
                    (session_id, user_name, machine, project, model, source_file,
                     started_at, last_message_at, message_count, file_mtime, file_hash,
                     is_subagent, parent_session_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id""",
                (
                    parsed.session_id, user_name, machine, parsed.project,
                    parsed.model, source_file, parsed.started_at,
                    parsed.last_message_at, parsed.message_count, file_mtime, file_hash,
                    is_subagent, parent_session_id,
                ),
            )
            conv_id = cur.fetchone()[0]

        # Batch insert all messages (execute_values is ~100x faster than individual INSERTs)
        rows = []
        for msg in parsed.messages:
            ch = hashlib.sha256(msg.content.encode("utf-8")).hexdigest() if msg.content else None
            rows.append((
                conv_id, msg.uuid, msg.parent_uuid, msg.role,
                msg.content, msg.thinking,
                Json(msg.tool_calls) if msg.tool_calls else None,
                Json(msg.tool_results) if msg.tool_results else None,
                msg.is_sidechain, msg.timestamp, msg.sequence_num, ch,
            ))

        if rows:
            execute_values(
                cur,
                """INSERT INTO messages
                    (conversation_id, uuid, parent_uuid, role, content,
                     thinking, tool_calls, tool_results, is_sidechain,
                     timestamp, sequence_num, content_hash)
                VALUES %s""",
                rows,
                page_size=500,
            )

    conn.commit()
    logger.info(
        f"Upserted conversation {parsed.session_id}: "
        f"{parsed.message_count} messages"
    )
    return conv_id
