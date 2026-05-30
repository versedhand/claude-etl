"""Embedding generation for conversation messages.

Content-addressed: embeddings are stored by content hash in a separate table.
Same content across different conversations shares one embedding.

Skip rules filter out noise (tool dumps, code blocks, paste bombs, etc.)
to keep embedding costs reasonable.
"""

import hashlib
import logging
import os
import re
from typing import Optional

import psycopg2

logger = logging.getLogger(__name__)

# Skip rules — messages that aren't worth embedding
MIN_CONTENT_LENGTH = 20   # skip one-word acks
MAX_CONTENT_LENGTH = 5000  # skip paste dumps
MAX_CODE_RATIO = 0.5       # skip if >50% non-alpha (code/JSON)

EMBEDDABLE_ROLES = {"user", "assistant"}


def content_hash(text: str) -> str:
    """SHA256 hash of content for dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def should_embed(role: str, content: Optional[str], tool_calls: Optional[list]) -> bool:
    """Determine if a message is worth embedding.

    Returns False for noise: tool-only turns, paste dumps, code blocks,
    system messages, tiny acks, etc.
    """
    if role not in EMBEDDABLE_ROLES:
        return False

    if not content or not content.strip():
        return False

    text = content.strip()
    length = len(text)

    if length < MIN_CONTENT_LENGTH:
        return False

    if length > MAX_CONTENT_LENGTH:
        return False

    # Tool-only assistant turns (no text, just tool calls)
    if role == "assistant" and tool_calls and not text:
        return False

    # High non-alpha ratio = code/JSON/data dump
    alpha_chars = sum(1 for c in text if c.isalpha() or c.isspace())
    if length > 50 and alpha_chars / length < (1 - MAX_CODE_RATIO):
        return False

    return True


def compute_hashes_for_messages(conn):
    """Populate content_hash column for all messages that don't have one yet."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, content FROM messages WHERE content_hash IS NULL AND content IS NOT NULL"
        )
        rows = cur.fetchall()

    if not rows:
        logger.info("All messages already have content hashes")
        return 0

    updated = 0
    with conn.cursor() as cur:
        for msg_id, content in rows:
            h = content_hash(content)
            cur.execute(
                "UPDATE messages SET content_hash = %s WHERE id = %s",
                (h, msg_id)
            )
            updated += 1

    conn.commit()
    logger.info(f"Computed {updated} content hashes")
    return updated


def find_embeddable_messages(conn) -> list[dict]:
    """Find messages that should be embedded but don't have embeddings yet.

    Returns list of {content_hash, content, role} dicts for unique content
    that needs embedding.
    """
    with conn.cursor() as cur:
        # Find unique content hashes that:
        # 1. Don't have an embedding yet
        # 2. Meet the should_embed criteria (checked in Python)
        cur.execute("""
            SELECT DISTINCT m.content_hash, m.content, m.role, m.tool_calls
            FROM messages m
            LEFT JOIN embeddings e ON m.content_hash = e.content_hash
            WHERE m.content_hash IS NOT NULL
              AND m.content IS NOT NULL
              AND e.content_hash IS NULL
              AND m.role IN ('user', 'assistant')
        """)
        rows = cur.fetchall()

    candidates = []
    for content_hash_val, content, role, tool_calls in rows:
        if should_embed(role, content, tool_calls):
            candidates.append({
                "content_hash": content_hash_val,
                "content": content,
                "role": role,
            })

    logger.info(
        f"Found {len(candidates)} embeddable messages "
        f"(filtered from {len(rows)} unhashed)"
    )
    return candidates


def try_reuse_old_embeddings(conn, old_db_config: Optional[dict] = None) -> int:
    """Check old conversationsdb for existing embeddings matching content hashes.

    If the old DB has embeddings for the same content, copy them instead of
    regenerating. Saves API cost.
    """
    if not old_db_config:
        old_db_config = {
            "host": "100.127.104.75",
            "port": 5432,
            "dbname": "conversationsdb",
            "user": "postgres",
            "password": os.environ.get("LIFEDB_PASSWORD", ""),
        }

    try:
        old_conn = psycopg2.connect(**old_db_config)
    except Exception as e:
        logger.warning(f"Can't connect to old DB for embedding reuse: {e}")
        return 0

    # Find content hashes in new DB that need embeddings
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT m.content_hash, m.content
            FROM messages m
            LEFT JOIN embeddings e ON m.content_hash = e.content_hash
            WHERE m.content_hash IS NOT NULL AND e.content_hash IS NULL
        """)
        need_embeddings = {row[0]: row[1] for row in cur.fetchall()}

    if not need_embeddings:
        old_conn.close()
        return 0

    # Check old DB for matching embeddings
    # Old schema may differ — try common column names
    reused = 0
    try:
        with old_conn.cursor() as old_cur:
            # Try to find embeddings in old DB by content matching
            old_cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'claude_messages' AND column_name = 'embedding'"
            )
            if old_cur.fetchone():
                for ch, content in need_embeddings.items():
                    old_cur.execute(
                        "SELECT embedding FROM claude_messages WHERE content_hash = %s AND embedding IS NOT NULL LIMIT 1",
                        (ch,)
                    )
                    row = old_cur.fetchone()
                    if row:
                        with conn.cursor() as cur:
                            cur.execute(
                                "INSERT INTO embeddings (content_hash, content_preview, embedding) "
                                "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                                (ch, content[:200], row[0])
                            )
                        reused += 1
    except Exception as e:
        logger.warning(f"Error reusing old embeddings: {e}")
    finally:
        old_conn.close()

    if reused:
        conn.commit()
        logger.info(f"Reused {reused} embeddings from old DB")

    return reused


def store_embedding(conn, ch: str, content_preview: str, embedding_vector: list[float]):
    """Store a single embedding in the embeddings table."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO embeddings (content_hash, content_preview, embedding) "
            "VALUES (%s, %s, %s::halfvec) ON CONFLICT (content_hash) DO NOTHING",
            (ch, content_preview[:200], str(embedding_vector))
        )
    conn.commit()


def get_embedding_stats(conn) -> dict:
    """Return embedding coverage stats."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM messages")
        total_msgs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM messages WHERE content_hash IS NOT NULL")
        hashed = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM embeddings")
        embedded = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(DISTINCT m.content_hash)
            FROM messages m
            LEFT JOIN embeddings e ON m.content_hash = e.content_hash
            WHERE m.content_hash IS NOT NULL AND e.content_hash IS NULL
              AND m.role IN ('user', 'assistant')
        """)
        pending = cur.fetchone()[0]

    return {
        "total_messages": total_msgs,
        "hashed": hashed,
        "embedded": embedded,
        "pending_unique_hashes": pending,
    }
