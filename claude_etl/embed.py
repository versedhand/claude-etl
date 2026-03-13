#!/usr/bin/env python3
"""
Embedding generation for Claude conversations.

Generates OpenAI embeddings for messages that don't have them yet.
Called by code_ingest.py and web_ingest.py after they complete.
"""

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple
import psycopg2
from psycopg2.extras import execute_values

# Try to import OpenAI
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI not installed. Run: pip install openai")
    sys.exit(1)

# Configuration
DB_CONFIG = {
    "host": "100.127.104.75",
    "port": 5432,
    "dbname": "lifedb",
    "user": "postgres",
    "password": os.environ["LIFEDB_PASSWORD"],
}

API_KEY_PATHS = [
    Path.home() / "corpus/isaac-workspace-corpus/etc/api-keys/openai.key",  # Desktop (WSL) / Server
]
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMS = 3072
BATCH_SIZE = 100  # OpenAI allows up to 2048, but smaller = safer
MAX_TOKENS_PER_TEXT = 8000  # Truncate long messages


def get_openai_client() -> OpenAI:
    """Get OpenAI client with API key."""
    api_key = None
    for path in API_KEY_PATHS:
        if path.exists():
            api_key = path.read_text().strip()
            break

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(f"No OpenAI API key found in {API_KEY_PATHS} or OPENAI_API_KEY env var")

    return OpenAI(api_key=api_key)


def content_hash(text: str) -> str:
    """Generate SHA256 hash of content for change detection."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]




def get_hash_matches(conn, messages: list) -> dict:
    """
    Check which messages have content that already has embeddings.
    Returns dict mapping uuid -> (existing_embedding, existing_hash, model)
    """
    if not messages:
        return {}
    
    # Compute hashes for all messages
    hash_to_uuid = {}
    for uuid, source_type, msg_content in messages:
        h = content_hash(msg_content or "")
        if h not in hash_to_uuid:
            hash_to_uuid[h] = []
        hash_to_uuid[h].append(uuid)
    
    # Find existing embeddings with matching hashes
    hashes = list(hash_to_uuid.keys())
    matches = {}
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content_hash, embedding, model
            FROM message_embeddings
            WHERE content_hash = ANY(%s)
        """, (hashes,))
        
        for row in cur.fetchall():
            existing_hash, embedding, model = row
            # Map back to all UUIDs with this hash
            for uuid in hash_to_uuid.get(existing_hash, []):
                matches[uuid] = (embedding, existing_hash, model)
    
    return matches


def reuse_embeddings(conn, messages: list, verbose: bool = True) -> int:
    """
    For messages with matching content_hash, copy existing embedding.
    Returns count of messages that reused embeddings.
    """
    if not messages:
        return 0
    
    matches = get_hash_matches(conn, messages)
    if not matches:
        return 0
    
    # Build message lookup
    msg_lookup = {m[0]: m for m in messages}
    
    # Insert embeddings for matched messages
    data = []
    for uuid, (embedding, existing_hash, model) in matches.items():
        msg = msg_lookup.get(uuid)
        if msg:
            source_type = msg[1]
            data.append((uuid, source_type, existing_hash, embedding, model))
    
    if data:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO message_embeddings (source_uuid, source_type, content_hash, embedding, model)
                VALUES %s
                ON CONFLICT (source_uuid) DO NOTHING
            """, data, template="(%s, %s, %s, %s::vector, %s)")
        conn.commit()
        
        if verbose:
            print(f"  Reused {len(data)} embeddings from hash matches (saved API calls)")
    
    return len(data)

def truncate_text(text: str, max_chars: int = 24000) -> str:
    """Truncate text to avoid token limits. ~6000 tokens max per text to stay under 8192 limit."""
    if len(text) > max_chars:
        return text[:max_chars] + "... [truncated]"
    return text


def chunk_text(text: str, chunk_size: int = 6000, overlap: int = 300) -> List[str]:
    """
    Split long text into overlapping chunks for embedding.

    Args:
        text: The text to chunk
        chunk_size: Target size per chunk in characters (~1500-2000 tokens, safe under 8192 limit)
        overlap: Character overlap between chunks for context continuity

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap for context continuity

        # Prevent infinite loop on very small overlap
        if start >= len(text) - overlap:
            break

    return chunks


def average_embeddings(embeddings: List[List[float]]) -> List[float]:
    """Average multiple embeddings into one vector."""
    if len(embeddings) == 1:
        return embeddings[0]

    # Element-wise average
    num_dims = len(embeddings[0])
    averaged = []
    for i in range(num_dims):
        avg = sum(emb[i] for emb in embeddings) / len(embeddings)
        averaged.append(avg)
    return averaged


def ensure_table(conn):
    """Verify embeddings table exists (don't create - use existing schema)."""
    with conn.cursor() as cur:
        # Check table exists
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'message_embeddings'
        """)
        if not cur.fetchone():
            raise RuntimeError("message_embeddings table missing. Create it first with correct dimensions.")

        # Ensure source_type index exists
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_source_type
            ON message_embeddings(source_type)
        """)
        # Note: HNSW index limited to 2000 dims, so we use IVFFlat for 3072-dim vectors
        # Index already exists on table - don't recreate
    conn.commit()
    print("Embeddings table ready")


def get_unembedded_web_messages(conn, limit: int = 1000) -> List[Tuple[str, str, str]]:
    """Get web messages that need embeddings. Returns (uuid, source_type, content)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.uuid, 'web', m.content
            FROM claude_web_messages m
            WHERE m.content IS NOT NULL
              AND m.content != ''
              AND NOT EXISTS (SELECT 1 FROM message_embeddings e WHERE e.source_uuid = m.uuid)
            LIMIT %s
        """, (limit,))
        return cur.fetchall()


def get_unembedded_code_messages(conn, limit: int = 1000) -> List[Tuple[str, str, str]]:
    """Get code messages that need embeddings. Returns (uuid, source_type, content)."""
    with conn.cursor() as cur:
        # Extract message content from JSONL data
        cur.execute("""
            SELECT
                r.record_uuid::text,
                'code',
                COALESCE(r.data->>'message', r.data->>'content', r.data::text)
            FROM claude_code_raw r
            WHERE r.record_uuid IS NOT NULL
              AND LENGTH(r.data::text) < 100000  -- Skip messages > 100KB
              AND r.data::text NOT LIKE '%%"type": "image"%%'  -- Skip base64 images
              AND r.data::text NOT LIKE '%%data:image/%%'  -- Skip inline base64
              AND r.record_type IN ('user', 'assistant')
              AND NOT EXISTS (SELECT 1 FROM message_embeddings e WHERE e.source_uuid = r.record_uuid::text)
            LIMIT %s
        """, (limit,))
        return cur.fetchall()


def get_unembedded_chatgpt_messages(conn, limit: int = 1000) -> List[Tuple[str, str, str]]:
    """Get ChatGPT messages that need embeddings. Returns (uuid, source_type, content)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.message_id, 'chatgpt', m.content
            FROM chatgpt_messages m
            WHERE m.content IS NOT NULL
              AND m.content != ''
              AND m.is_canonical = TRUE
              AND NOT EXISTS (SELECT 1 FROM message_embeddings e WHERE e.source_uuid = m.message_id)
            LIMIT %s
        """, (limit,))
        return cur.fetchall()


def generate_embeddings(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts."""
    # Truncate long texts
    texts = [truncate_text(t) for t in texts]

    # Filter empty texts
    texts = [t if t else " " for t in texts]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return [item.embedding for item in response.data]


def embed_batch(conn, client: OpenAI, messages: List[Tuple[str, str, str]]) -> int:
    """Embed a batch of messages and insert to database. Returns count inserted."""
    if not messages:
        return 0

    uuids = [m[0] for m in messages]
    source_types = [m[1] for m in messages]
    contents = [m[2] or "" for m in messages]

    # Generate embeddings
    embeddings = generate_embeddings(client, contents)

    # Prepare data for insert
    data = [
        (uuid, source_type, content_hash(content), embedding, EMBEDDING_MODEL)
        for uuid, source_type, content, embedding in zip(uuids, source_types, contents, embeddings)
    ]

    # Upsert
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO message_embeddings (source_uuid, source_type, content_hash, embedding, model)
            VALUES %s
            ON CONFLICT (source_uuid) DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                embedding = EXCLUDED.embedding,
                model = EXCLUDED.model,
                updated_at = NOW()
        """, data, template="(%s, %s, %s, %s::vector, %s)")

    conn.commit()
    return len(data)


def embed_single_long_message(conn, client: OpenAI, message: Tuple[str, str, str],
                              verbose: bool = True) -> bool:
    """
    Embed a single long message by chunking and averaging.

    Args:
        message: (uuid, source_type, content)

    Returns:
        True if successful, False if failed
    """
    uuid, source_type, content = message

    # Chunk the content (6k chars per chunk, ~1.5-2k tokens each, safe under 8192 limit)
    chunks = chunk_text(content, chunk_size=6000, overlap=300)

    if verbose:
        print(f"    Chunking message {uuid[:8]}... into {len(chunks)} chunks")

    try:
        # Embed each chunk
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[chunk]
            )
            chunk_embeddings.append(response.data[0].embedding)
            time.sleep(0.1)  # Small delay between chunk API calls

        # Average the embeddings
        final_embedding = average_embeddings(chunk_embeddings)

        # Insert to database
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO message_embeddings (source_uuid, source_type, content_hash, embedding, model)
                VALUES (%s, %s, %s, %s::vector, %s)
                ON CONFLICT (source_uuid) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    embedding = EXCLUDED.embedding,
                    model = EXCLUDED.model,
                    updated_at = NOW()
            """, (uuid, source_type, content_hash(content), final_embedding, EMBEDDING_MODEL))
        conn.commit()

        if verbose:
            print(f"    Embedded chunked message {uuid[:8]}... ({len(chunks)} chunks averaged)")
        return True

    except Exception as e:
        if verbose:
            print(f"    Failed to embed chunked message {uuid[:8]}...: {e}")
        return False


def embed_batch_with_retry(conn, client: OpenAI, messages: List[Tuple[str, str, str]],
                           verbose: bool = True) -> Tuple[int, int]:
    """
    Embed batch with retry logic for token limit errors.
    Splits batch on failure, retries smaller batches.
    For single long messages, chunks and averages embeddings.
    Returns (count_embedded, count_errors).
    """
    if not messages:
        return 0, 0

    try:
        count = embed_batch(conn, client, messages)
        return count, 0
    except Exception as e:
        error_str = str(e)
        # Check if it's a token limit error
        if "maximum context length" not in error_str and "token" not in error_str.lower():
            # Not a token error, don't retry
            raise

        # Token limit exceeded - split and retry
        if len(messages) == 1:
            # Single message too long - chunk it and average embeddings
            success = embed_single_long_message(conn, client, messages[0], verbose)
            return (1, 0) if success else (0, 1)

        # Split batch in half and retry each
        mid = len(messages) // 2
        left = messages[:mid]
        right = messages[mid:]

        if verbose:
            print(f"    Batch too large, splitting {len(messages)} -> {len(left)} + {len(right)}")

        left_count, left_errors = embed_batch_with_retry(conn, client, left, verbose)
        time.sleep(0.2)  # Small delay between retries
        right_count, right_errors = embed_batch_with_retry(conn, client, right, verbose)

        return left_count + right_count, left_errors + right_errors


def embed_new_messages(limit_per_source: int = 5000, verbose: bool = True) -> dict:
    """
    Main entry point. Find and embed messages that don't have embeddings yet.

    Returns stats dict with counts.
    """
    stats = {"web": 0, "code": 0, "chatgpt": 0, "errors": 0}

    # Connect
    conn = psycopg2.connect(**DB_CONFIG)
    client = get_openai_client()

    try:
        ensure_table(conn)

        # Process web messages
        if verbose:
            print("Checking web messages...")
        web_messages = get_unembedded_web_messages(conn, limit_per_source)
        if verbose:
            print(f"  Found {len(web_messages)} needing embeddings")

        # First, reuse embeddings for messages with matching content hash
        reused = reuse_embeddings(conn, web_messages, verbose)
        stats["web"] += reused
        # Filter out messages that were handled by hash reuse
        if reused > 0:
            reused_uuids = set(get_hash_matches(conn, web_messages).keys())
            web_messages = [m for m in web_messages if m[0] not in reused_uuids]
            if verbose:
                print(f"  {len(web_messages)} remaining after hash dedup")

        for i in range(0, len(web_messages), BATCH_SIZE):
            batch = web_messages[i:i + BATCH_SIZE]
            try:
                count, errors = embed_batch_with_retry(conn, client, batch, verbose)
                stats["web"] += count
                stats["errors"] += errors
                if verbose:
                    print(f"  Embedded {stats['web']}/{len(web_messages)} web messages")
            except Exception as e:
                stats["errors"] += 1
                print(f"  Error embedding web batch: {e}")

            # Rate limiting
            time.sleep(0.5)

        # Process code messages
        if verbose:
            print("Checking code messages...")
        code_messages = get_unembedded_code_messages(conn, limit_per_source)
        if verbose:
            print(f"  Found {len(code_messages)} needing embeddings")

        # First, reuse embeddings for messages with matching content hash
        reused = reuse_embeddings(conn, code_messages, verbose)
        stats["code"] += reused
        # Filter out messages that were handled by hash reuse
        if reused > 0:
            reused_uuids = set(get_hash_matches(conn, code_messages).keys())
            code_messages = [m for m in code_messages if m[0] not in reused_uuids]
            if verbose:
                print(f"  {len(code_messages)} remaining after hash dedup")

        for i in range(0, len(code_messages), BATCH_SIZE):
            batch = code_messages[i:i + BATCH_SIZE]
            try:
                count, errors = embed_batch_with_retry(conn, client, batch, verbose)
                stats["code"] += count
                stats["errors"] += errors
                if verbose:
                    print(f"  Embedded {stats['code']}/{len(code_messages)} code messages")
            except Exception as e:
                stats["errors"] += 1
                print(f"  Error embedding code batch: {e}")

            # Rate limiting
            time.sleep(0.5)

        # Process ChatGPT messages
        if verbose:
            print("Checking ChatGPT messages...")
        chatgpt_messages = get_unembedded_chatgpt_messages(conn, limit_per_source)
        if verbose:
            print(f"  Found {len(chatgpt_messages)} needing embeddings")

        # First, reuse embeddings for messages with matching content hash
        reused = reuse_embeddings(conn, chatgpt_messages, verbose)
        stats["chatgpt"] += reused
        # Filter out messages that were handled by hash reuse
        if reused > 0:
            reused_uuids = set(get_hash_matches(conn, chatgpt_messages).keys())
            chatgpt_messages = [m for m in chatgpt_messages if m[0] not in reused_uuids]
            if verbose:
                print(f"  {len(chatgpt_messages)} remaining after hash dedup")

        for i in range(0, len(chatgpt_messages), BATCH_SIZE):
            batch = chatgpt_messages[i:i + BATCH_SIZE]
            try:
                count, errors = embed_batch_with_retry(conn, client, batch, verbose)
                stats["chatgpt"] += count
                stats["errors"] += errors
                if verbose:
                    print(f"  Embedded {stats['chatgpt']}/{len(chatgpt_messages)} chatgpt messages")
            except Exception as e:
                stats["errors"] += 1
                print(f"  Error embedding chatgpt batch: {e}")

            # Rate limiting
            time.sleep(0.5)

    finally:
        conn.close()

    if verbose:
        print(f"\nDone. Web: {stats['web']}, Code: {stats['code']}, ChatGPT: {stats['chatgpt']}, Errors: {stats['errors']}")

    return stats


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate embeddings for Claude conversations")
    parser.add_argument("--limit", type=int, default=5000, help="Max messages per source per run")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--continuous", action="store_true", help="Loop until all messages embedded")
    args = parser.parse_args()

    total_errors = 0
    run_count = 0

    while True:
        run_count += 1
        if not args.quiet:
            print(f"\n{'='*50}")
            print(f"Run {run_count}")
            print(f"{'='*50}")

        stats = embed_new_messages(limit_per_source=args.limit, verbose=not args.quiet)
        total_errors += stats["errors"]

        # Check if we processed anything
        if stats["web"] == 0 and stats["code"] == 0 and stats["chatgpt"] == 0:
            if not args.quiet:
                print(f"\nAll messages embedded! Total runs: {run_count}, Total errors: {total_errors}")
            break

        if not args.continuous:
            break

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
