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

# Skip rules — messages that aren't worth embedding.
#
# NO MINIMUM LENGTH. Removed 2026-07-30 after direct measurement. The old
# MIN_CONTENT_LENGTH = 20 meant Isaac's terse ratifications were never embedded and
# therefore could never be returned by semantic search at all. Verified case: the user
# turn "ok let's go" (11 chars, 2026-07-27) is the Finch Harbor brand ratification cited
# across the corpus as a decided fact — it was `embedded = f`. His rulings are terse by
# design, so a byte floor preferentially deletes the decision record. Cost of removing it
# is negligible because content-addressing collapses 55,538 short user turns into 6,979
# distinct hashes (~cents).
#
# HONEST LIMIT: removing the floor makes these rows ELIGIBLE, not necessarily FINDABLE.
# A bare "ok let's go" vector matches other short affirmations, not "when did Isaac
# approve the brand." Real fix is contextual embedding — see CHUNKING-AND-CONTEXT-GAP.md.
MAX_CONTENT_LENGTH = 5000  # skip paste dumps — see F8 gap doc, 17,695 distinct contents
MAX_CODE_RATIO = 0.5       # skip if >50% non-alpha (code/JSON)

EMBEDDABLE_ROLES = {"user", "assistant"}

# Isaac's ruling (2026-07-30): "we don't want reasoning or tool blocks indexed."
# Structurally satisfied: the parser writes text blocks ONLY into messages.content
# (parser.extract_text_content keeps type == "text"), while thinking / tool_use /
# tool_result land in the separate thinking, tool_calls and tool_results columns. This
# module reads messages.content and nothing else, so the searchable corpus is prose only.
# Any future change that concatenates those columns into content BREAKS this ruling.


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
    """Populate content_hash column for all messages that don't have one yet.

    Skips empty-string content. Measured 2026-07-30: 73,175 rows had content = '' and
    content_hash IS NULL, and ZERO rows had non-empty content without a hash. So the
    entire remaining workload of this step was hashing the empty string 73,175 times —
    every one producing the same constant digest — inside a single ~22-minute
    transaction, for no benefit. An empty message is never embeddable (should_embed
    rejects it), so its hash is never read.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, content FROM messages "
            "WHERE content_hash IS NULL AND content IS NOT NULL AND content <> ''"
        )
        rows = cur.fetchall()

    if not rows:
        logger.info("All non-empty messages already have content hashes")
        return 0

    # One round trip per batch instead of one per row.
    from psycopg2.extras import execute_values
    payload = [(msg_id, content_hash(content)) for msg_id, content in rows]
    with conn.cursor() as cur:
        execute_values(
            cur,
            "UPDATE messages SET content_hash = data.h "
            "FROM (VALUES %s) AS data(id, h) WHERE messages.id = data.id::uuid",
            payload,
            page_size=1000,
        )

    conn.commit()
    logger.info(f"Computed {len(payload)} content hashes")
    return len(payload)


def find_embeddable_messages(conn) -> list[dict]:
    """Find messages that should be embedded but don't have embeddings yet.

    Returns list of {content_hash, content, role} dicts for unique content
    that needs embedding.
    """
    with conn.cursor() as cur:
        # DISTINCT ON (content_hash), not DISTINCT over 4 columns. The old form included
        # role and the tool_calls jsonb in the distinct key, so identical content that
        # appeared under both roles (or with differing tool_calls) yielded duplicate rows
        # for one hash — measured 41 duplicate hashes, i.e. paid-for duplicate API calls
        # whose second INSERT was silently dropped by ON CONFLICT DO NOTHING.
        #
        # Length bound is applied in SQL as well as in should_embed so the database does
        # not ship multi-hundred-KB paste dumps over the wire only for Python to discard
        # them. should_embed remains the single source of truth for eligibility.
        cur.execute("""
            SELECT DISTINCT ON (m.content_hash)
                   m.content_hash, m.content, m.role, m.tool_calls
            FROM messages m
            LEFT JOIN embeddings e ON m.content_hash = e.content_hash
            WHERE m.content_hash IS NOT NULL
              AND m.content IS NOT NULL
              AND m.content <> ''
              AND e.content_hash IS NULL
              AND m.role IN ('user', 'assistant')
              AND length(m.content) <= %s
            ORDER BY m.content_hash
        """, (MAX_CONTENT_LENGTH,))
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


def store_embedding(conn, ch: str, content_preview: str, embedding_vector: list[float],
                    commit: bool = True):
    """Store a single embedding. Pass commit=False to batch (see store_embeddings_batch)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO embeddings (content_hash, content_preview, embedding) "
            "VALUES (%s, %s, %s::halfvec) ON CONFLICT (content_hash) DO NOTHING",
            (ch, content_preview[:200], str(embedding_vector))
        )
    if commit:
        conn.commit()


def store_embeddings_batch(conn, items: list[tuple]) -> int:
    """Store a batch of (content_hash, preview, vector) with ONE commit.

    The per-row commit in store_embedding cost one fsync per embedding — 57k commits for
    a full backfill. One commit per API batch keeps the same crash semantics that matter
    here (a lost batch is simply re-embedded on the next run, because the driver is
    'rows lacking an embedding') while removing the fsync storm.
    """
    with conn.cursor() as cur:
        for ch, preview, vec in items:
            cur.execute(
                "INSERT INTO embeddings (content_hash, content_preview, embedding) "
                "VALUES (%s, %s, %s::halfvec) ON CONFLICT (content_hash) DO NOTHING",
                (ch, preview[:200], str(vec))
            )
    conn.commit()
    return len(items)


def get_coverage(conn, days: int = 7) -> dict:
    """Embedding coverage over the ELIGIBLE denominator for a trailing window.

    THE DENOMINATOR IS THE WHOLE POINT. Coverage measured against *all* user/assistant
    messages reads ~40-55% even in perfectly healthy months, because should_embed
    legitimately rejects empty turns, machine dumps and >5000-char pastes. A watchdog
    using that denominator could never be satisfied and would alarm forever.

    So eligibility is decided HERE by calling should_embed — the same predicate the
    embedder uses — rather than by re-implementing the rule in SQL. If the rule changes,
    the gate and the thing it gates move together. That is deliberate: a checker with its
    own private copy of the rule measures something different from what it gates, and the
    first edit to should_embed would silently break it.

    Returns eligible / embedded / pct / pending, plus scanned for transparency.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (m.content_hash)
                   m.content_hash, m.content, m.role, m.tool_calls,
                   (e.content_hash IS NOT NULL) AS has_embedding
            FROM messages m
            LEFT JOIN embeddings e ON m.content_hash = e.content_hash
            WHERE m.role IN ('user', 'assistant')
              AND m.content IS NOT NULL
              AND m.timestamp > now() - make_interval(days => %s)
            ORDER BY m.content_hash
        """, (days,))
        rows = cur.fetchall()

    eligible = embedded = 0
    for _ch, content, role, tool_calls, has_emb in rows:
        if should_embed(role, content, tool_calls):
            eligible += 1
            if has_emb:
                embedded += 1

    return {
        "window_days": days,
        "scanned_distinct": len(rows),
        "eligible": eligible,
        "embedded": embedded,
        "pending": eligible - embedded,
        "pct": round(100.0 * embedded / eligible, 2) if eligible else 100.0,
    }


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
