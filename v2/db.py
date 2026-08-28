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


def _stem(path: Optional[str]) -> Optional[str]:
    base = _basename(path)
    return base[: -len(".jsonl")] if base and base.endswith(".jsonl") else base


def _assert_same_source(stored_source: str, incoming_source: str, session_id: str):
    """D1 guard: refuse to overwrite a conversation ingested from a different file.

    upsert_conversation() DELETEs the existing row's messages before re-inserting.
    That DELETE must be reachable ONLY when we are demonstrably re-ingesting the
    same transcript. Everything else fails closed and loudly.

    The rules, in order:

    1. Identical source_file -> ALLOW. This is the ordinary idempotent re-ingest
       that runs every minute; without it nothing would ever update.

    2. Stored source is NOT a .jsonl transcript -> BLOCK, always.
       A conversation that came from a legacy import, a web export zip, or the v1
       migration was never produced by a transcript file, so no transcript file can
       legitimately claim to be a newer version of it. The previous version of this
       guard RETURNED EARLY here ("web imports are out of scope"), which meant any
       of the 84,925 non-.jsonl-sourced conversations — including all 83 rows
       migrated from v1 — could be silently overwritten by a file ingest that
       resolved to the same (session_id, machine, user_name). Nothing needed that
       permission: web_ingest.py does its own SQL and never calls this function,
       so the only writers here are file ingests. Made explicit and total.

    3. Incoming source is NOT a .jsonl transcript -> BLOCK.
       Defensive: this function only ever guards the file-ingest path.

    4. Two different .jsonl paths -> ALLOW ONLY a PROVABLE relocation:
       the basename must match AND the basename stem must equal the session_id.
       A Claude Code parent transcript is named <session-uuid>.jsonl, so a file
       named after the very session it claims is the same conversation no matter
       which project directory it now sits in (verified: 394/394 parent
       transcripts on this host satisfy stem == sessionId). This preserves the
       genuine project-directory-rename case.

       Matching basenames alone is NOT sufficient and used to be accepted:
       subagent transcripts are named 'journal.jsonl' / 'agent-<id>.jsonl' and
       that name repeats in every workflow directory, so a bare basename match
       let one file's messages replace an unrelated file's (demonstrated:
       11,739 messages deleted in a rolled-back test). Subagent conversations
       carry a path-derived 'sub:...' session_id which never equals a file stem,
       so they can no longer satisfy this branch at all.
    """
    if stored_source == incoming_source:
        return  # rule 1: idempotent re-ingest of the same file

    stored_is_jsonl = str(stored_source).endswith(".jsonl")
    incoming_is_jsonl = str(incoming_source).endswith(".jsonl")

    if not stored_is_jsonl:
        raise SourceFileCollision(
            f"REFUSING to overwrite conversation session_id={session_id!r}: "
            f"stored source_file={stored_source!r} is not a transcript file "
            f"(legacy import, web export or migration), but incoming file="
            f"{incoming_source!r} is. A file ingest may never replace a "
            f"non-transcript conversation."
        )

    if not incoming_is_jsonl:
        raise SourceFileCollision(
            f"REFUSING to overwrite conversation session_id={session_id!r}: "
            f"incoming source={incoming_source!r} is not a transcript file. "
            f"This code path ingests transcripts only."
        )

    # rule 4: both are transcripts at different paths — only a provable relocation
    if _basename(stored_source) == _basename(incoming_source) and _stem(incoming_source) == session_id:
        return

    raise SourceFileCollision(
        f"REFUSING to overwrite conversation session_id={session_id!r}: "
        f"stored source_file={stored_source!r} but incoming file={incoming_source!r}. "
        f"Two distinct transcripts cannot share one conversation row."
    )


def _content_hash(content: Optional[str]) -> Optional[str]:
    """The content_hash we store for a message. One definition, used by both
    the writer and the append-only prefix check, so they can never disagree."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None


def _append_only_from(cur, conv_id: int, parsed: ParsedConversation) -> Optional[int]:
    """Return the highest stored sequence_num IF the stored messages are a
    provable PREFIX of `parsed`, else None.

    None means "take the DELETE + reinsert path" — this function must never
    return a number on anything it has not positively verified, because the
    caller uses it to SKIP the delete.

    Why this exists: the sync cron re-ingests a live transcript every minute,
    and the DELETE + reinsert path rewrites every message of that session on
    each pass. Message rows carry tool_results that run to hundreds of KB, so
    WAL tracks BYTES OF TOAST REWRITTEN, not row count — measured 2026-08-28,
    a 2.5 GB messages table was generating 25-33 GB of WAL per day and filling
    the backup volume. Claude transcripts are append-only in the normal case,
    so appending is both correct and ~95% cheaper.

    ⛔ The abnormal case is real and is why every check below is required:
    COMPACTION REWRITES A TRANSCRIPT FROM THE HEAD. After it, the file can be
    the same length or longer while containing entirely different messages at
    the same sequence numbers. Verifying only the tail would step straight
    into that. So we pin BOTH ends of the stored range by uuid and re-derived
    content hash, and bail on anything we cannot prove.
    """
    cur.execute(
        "SELECT COUNT(*), MIN(sequence_num), MAX(sequence_num) FROM messages WHERE conversation_id = %s",
        (conv_id,),
    )
    stored_count, min_seq, max_seq = cur.fetchone()

    # Nothing stored: no prefix to extend. (The DELETE is a no-op anyway.)
    if not stored_count:
        return None

    # The file shrank, or is unchanged in length. Either way this is not an
    # append, and "unchanged" still needs the full path so an in-place edit
    # cannot survive as a stale row.
    if parsed.message_count <= stored_count:
        return None

    # Require a contiguous stored range. A gap means something already
    # diverged from a clean prefix and we should not reason about offsets.
    if min_seq is None or max_seq is None or (max_seq - min_seq + 1) != stored_count:
        return None

    # Index the incoming messages by sequence_num so the comparison does not
    # assume list position equals sequence.
    incoming = {m.sequence_num: m for m in parsed.messages}

    cur.execute(
        "SELECT sequence_num, uuid, content_hash FROM messages "
        "WHERE conversation_id = %s AND sequence_num IN (%s, %s)",
        (conv_id, min_seq, max_seq),
    )
    boundaries = cur.fetchall()
    if len(boundaries) != len({min_seq, max_seq}):
        return None

    for seq, stored_uuid, stored_hash in boundaries:
        msg = incoming.get(seq)
        if msg is None:
            return None
        # uuid is the strong identity. Without one on BOTH sides we decline —
        # a missing uuid must not read as a match.
        if not stored_uuid or not msg.uuid or stored_uuid != msg.uuid:
            return None
        # Content must be byte-identical too, so an in-place edit that kept the
        # uuid still falls back to the full rewrite.
        if stored_hash != _content_hash(msg.content):
            return None

    return max_seq


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

    Two paths, and the fast one is only taken on a PROVEN prefix:

    - APPEND: stored messages are a verified prefix of `parsed` (see
      `_append_only_from`) — insert only the new tail, no DELETE.
    - REPLACE: anything else — DELETE + INSERT in a transaction, as before.

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

        append_from = None
        if existing:
            conv_id, stored_source = existing
            # Fail closed BEFORE any destructive statement runs. Unchanged, and
            # it still runs first on the append path — a source collision must
            # be refused whether or not we were going to delete anything.
            _assert_same_source(stored_source, source_file, parsed.session_id)
            append_from = _append_only_from(cur, conv_id, parsed)
            if append_from is None:
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

        # Batch insert messages (execute_values is ~100x faster than individual INSERTs).
        # On the append path only the new tail is written; on the replace path
        # everything was just deleted, so this writes the whole conversation.
        to_write = parsed.messages
        if append_from is not None:
            to_write = [m for m in parsed.messages if m.sequence_num > append_from]

        rows = []
        for msg in to_write:
            ch = _content_hash(msg.content)
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
    # Say WHICH path ran and how many rows it actually wrote. Without this an
    # append and a full rewrite are indistinguishable in the log, and the whole
    # point of the change is the difference between them.
    mode = f"appended {len(rows)}" if append_from is not None else f"replaced {len(rows)}"
    logger.info(
        f"Upserted conversation {parsed.session_id}: "
        f"{parsed.message_count} messages ({mode})"
    )
    return conv_id
