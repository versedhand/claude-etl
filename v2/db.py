"""Database operations for conversation ingestion."""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import psycopg2
from psycopg2.extras import Json

from .parser import ParsedConversation, ParsedMessage

logger = logging.getLogger(__name__)

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
    source TEXT DEFAULT 'code'
);

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


def upsert_conversation(
    conn,
    parsed: ParsedConversation,
    user_name: str,
    machine: str,
    source_file: str,
    file_mtime: Optional[datetime] = None,
):
    """Insert or replace a conversation and all its messages.

    DELETE + INSERT in a transaction for idempotency.
    """
    with conn.cursor() as cur:
        # Find existing conversation
        cur.execute(
            "SELECT id FROM conversations WHERE session_id = %s AND machine = %s AND user_name = %s",
            (parsed.session_id, machine, user_name),
        )
        existing = cur.fetchone()

        if existing:
            conv_id = existing[0]
            # Delete old messages (CASCADE would do this, but explicit is clearer)
            cur.execute("DELETE FROM messages WHERE conversation_id = %s", (conv_id,))
            # Update conversation metadata
            cur.execute(
                """UPDATE conversations SET
                    project = %s, model = %s, source_file = %s,
                    started_at = %s, last_message_at = %s,
                    message_count = %s, ingested_at = NOW(), file_mtime = %s
                WHERE id = %s""",
                (
                    parsed.project, parsed.model, source_file,
                    parsed.started_at, parsed.last_message_at,
                    parsed.message_count, file_mtime, conv_id,
                ),
            )
        else:
            # Insert new conversation
            cur.execute(
                """INSERT INTO conversations
                    (session_id, user_name, machine, project, model, source_file,
                     started_at, last_message_at, message_count, file_mtime)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id""",
                (
                    parsed.session_id, user_name, machine, parsed.project,
                    parsed.model, source_file, parsed.started_at,
                    parsed.last_message_at, parsed.message_count, file_mtime,
                ),
            )
            conv_id = cur.fetchone()[0]

        # Insert all messages (with content hash for embedding lookup)
        for msg in parsed.messages:
            ch = hashlib.sha256(msg.content.encode("utf-8")).hexdigest() if msg.content else None
            cur.execute(
                """INSERT INTO messages
                    (conversation_id, uuid, parent_uuid, role, content,
                     thinking, tool_calls, tool_results, is_sidechain,
                     timestamp, sequence_num, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    conv_id, msg.uuid, msg.parent_uuid, msg.role,
                    msg.content, msg.thinking,
                    Json(msg.tool_calls) if msg.tool_calls else None,
                    Json(msg.tool_results) if msg.tool_results else None,
                    msg.is_sidechain, msg.timestamp, msg.sequence_num, ch,
                ),
            )

    conn.commit()
    logger.info(
        f"Upserted conversation {parsed.session_id}: "
        f"{parsed.message_count} messages"
    )
    return conv_id
