#!/usr/bin/env python3
"""
Topic-based conversation chunking.

Uses message embeddings to detect topic shifts, then:
1. Splits conversations into coherent chunks
2. Generates topic summaries (LLM)
3. Embeds each chunk for semantic search

Requires: message_embeddings table populated (run embed.py first)
"""

import os
import sys
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import execute_values

# Try imports
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

# Check multiple paths for API key (desktop vs server)
API_KEY_PATHS = [
    Path.home() / "corpus/isaac-workspace-corpus/etc/api-keys/openai.key",
]
EMBEDDING_MODEL = "text-embedding-3-large"
SUMMARY_MODEL = "gpt-4o-mini"

# Chunking parameters
SIMILARITY_THRESHOLD = 0.3    # Below this = topic boundary (tune this)
MIN_CHUNK_MESSAGES = 3        # Don't create tiny chunks
MAX_CHUNK_MESSAGES = 50       # Force split if chunk gets huge
MAX_CHUNK_CHARS = 20000       # Also force split by size

# Strip base64 blobs from chunk text (screenshots, PDFs, images)
_BASE64_RE = re.compile(r'"data"\s*:\s*"[A-Za-z0-9+/=\s\\]{1000,}"')

def _strip_base64(text: str) -> str:
    """Remove base64 data blobs from text to prevent storage/embedding waste."""
    return _BASE64_RE.sub('"data": "[base64 removed]"', text)



@dataclass
class Message:
    """A message with its embedding."""
    uuid: str
    idx: int
    role: str
    content: str
    embedding: Optional[List[float]] = None


@dataclass
class Chunk:
    """A topic chunk spanning multiple messages."""
    start_idx: int
    end_idx: int
    messages: List[Message]
    boundary_similarity: Optional[float] = None

    @property
    def text(self) -> str:
        """Concatenated message content."""
        parts = []
        for m in self.messages:
            role = m.role.upper() if m.role else "MSG"
            parts.append(f"[{role}]: {m.content}")
        return _strip_base64("\n\n".join(parts))

    @property
    def char_count(self) -> int:
        return len(self.text)


def get_openai_client() -> OpenAI:
    """Get OpenAI client with API key."""
    api_key = None

    # Check file paths
    for path in API_KEY_PATHS:
        if path.exists():
            api_key = path.read_text().strip()
            break

    # Fall back to env var
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(f"No OpenAI API key found at {API_KEY_PATHS} or in OPENAI_API_KEY env var")

    return OpenAI(api_key=api_key)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def batch_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix for all embedding pairs.
    Much faster than pairwise loop for large conversations.

    Args:
        embeddings: (n, dim) array of embeddings
    Returns:
        (n, n) similarity matrix
    """
    # Normalize each row
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms
    # Similarity matrix via dot product
    return np.dot(normalized, normalized.T)


def ensure_table(conn):
    """Create conversation_topics table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_topics (
                topic_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                conversation_id TEXT NOT NULL,
                source_type TEXT NOT NULL,

                start_msg_idx INT NOT NULL,
                end_msg_idx INT NOT NULL,
                start_msg_uuid TEXT,
                end_msg_uuid TEXT,

                topic_summary TEXT,
                chunk_text TEXT NOT NULL,
                message_count INT NOT NULL,

                embedding vector(3072),
                model TEXT,

                avg_internal_sim FLOAT,
                boundary_sim FLOAT,

                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_topics_conversation
            ON conversation_topics(conversation_id);

            CREATE INDEX IF NOT EXISTS idx_topics_source_type
            ON conversation_topics(source_type);

        """)
    conn.commit()
    print("conversation_topics table ready")


def get_conversation_messages_web(conn, conversation_id: str) -> List[Message]:
    """Get messages for a web conversation with their embeddings."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                m.uuid,
                m.sender,
                m.content,
                e.embedding::text
            FROM claude_web_messages m
            LEFT JOIN message_embeddings e ON e.source_uuid = m.uuid
            WHERE m.conversation_uuid = %s
              AND m.content IS NOT NULL
              AND m.content != ''
            ORDER BY m.created_at
        """, (conversation_id,))

        messages = []
        for idx, (uuid, sender, content, emb_str) in enumerate(cur.fetchall()):
            embedding = None
            if emb_str:
                # Parse vector string like "[0.1,0.2,...]"
                embedding = json.loads(emb_str)
            messages.append(Message(
                uuid=uuid,
                idx=idx,
                role=sender,
                content=content,
                embedding=embedding
            ))
        return messages


def get_conversation_messages_code(conn, conversation_id: str) -> List[Message]:
    """Get messages for a code conversation with their embeddings."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                r.record_uuid::text,
                r.record_type,
                COALESCE(r.data->>'message', r.data->>'content', ''),
                e.embedding::text
            FROM claude_code_raw r
            LEFT JOIN message_embeddings e ON e.source_uuid = r.record_uuid::text
            WHERE r.conversation_id = %s
              AND r.record_type IN ('user', 'assistant')
            ORDER BY r.timestamp
        """, (conversation_id,))

        messages = []
        for idx, (uuid, role, content, emb_str) in enumerate(cur.fetchall()):
            if not content or not content.strip():
                continue
            embedding = None
            if emb_str:
                embedding = json.loads(emb_str)
            messages.append(Message(
                uuid=uuid,
                idx=idx,
                role=role,
                content=content,
                embedding=embedding
            ))
        return messages


def detect_topic_boundaries(messages: List[Message],
                           threshold: float = SIMILARITY_THRESHOLD) -> List[int]:
    """
    Find indices where topic shifts occur based on embedding similarity drops.
    Returns list of boundary indices (the index AFTER which a new topic starts).

    Optimized: Uses batch similarity computation for speed.
    """
    if len(messages) < 2:
        return []

    # Build embedding matrix for messages that have embeddings
    # Track which indices have embeddings
    indices_with_emb = []
    emb_list = []
    for i, m in enumerate(messages):
        if m.embedding is not None:
            indices_with_emb.append(i)
            emb_list.append(m.embedding)

    if len(emb_list) < 2:
        return []

    # Compute similarity matrix
    emb_array = np.array(emb_list)
    sim_matrix = batch_cosine_similarity(emb_array)

    # Find boundaries: where adjacent (in original order) embedded messages have low similarity
    boundaries = []
    for j in range(len(indices_with_emb) - 1):
        orig_idx = indices_with_emb[j]
        next_orig_idx = indices_with_emb[j + 1]

        # Only consider as boundary if they're actually adjacent in original
        if next_orig_idx == orig_idx + 1:
            sim = sim_matrix[j, j + 1]
            if sim < threshold:
                boundaries.append(orig_idx)

    return boundaries


def create_chunks(messages: List[Message], boundaries: List[int]) -> List[Chunk]:
    """
    Split messages into chunks at boundary points.
    Enforces min/max chunk sizes.
    """
    if not messages:
        return []

    # Add implicit boundaries at start/end
    all_boundaries = [-1] + sorted(set(boundaries)) + [len(messages) - 1]

    chunks = []
    i = 0

    while i < len(all_boundaries) - 1:
        start_idx = all_boundaries[i] + 1
        end_idx = all_boundaries[i + 1]

        # Get messages for this chunk
        chunk_messages = messages[start_idx:end_idx + 1]

        # Skip empty chunks
        if not chunk_messages:
            i += 1
            continue

        # Check size constraints
        chunk = Chunk(
            start_idx=start_idx,
            end_idx=end_idx,
            messages=chunk_messages
        )

        # If chunk is too small, merge with next (but not if that would cause split-merge loop)
        if len(chunk_messages) < MIN_CHUNK_MESSAGES and i < len(all_boundaries) - 2:
            # Check if merging would create a chunk that's too big to keep but too small to split
            next_end = all_boundaries[i + 2]
            merged_msgs = messages[start_idx:next_end + 1]
            merged_chars = sum(len(m.content or '') for m in merged_msgs)
            # Only merge if the result won't be oversized, OR if it would be splittable
            if merged_chars <= MAX_CHUNK_CHARS or len(merged_msgs) >= MIN_CHUNK_MESSAGES * 2:
                all_boundaries.pop(i + 1)
                continue
            # Otherwise, accept the small chunk as-is

        # If chunk is too big, force split (but only if result won't be too small)
        if (len(chunk_messages) > MAX_CHUNK_MESSAGES or chunk.char_count > MAX_CHUNK_CHARS) and len(chunk_messages) >= MIN_CHUNK_MESSAGES * 2:
            # Split in half
            mid = start_idx + len(chunk_messages) // 2
            all_boundaries.insert(i + 1, mid - 1)
            continue
        # If too big but can't split without making tiny chunks, accept as-is

        # Record boundary similarity if available
        if end_idx < len(messages) - 1:
            m_end = messages[end_idx]
            m_next = messages[end_idx + 1]
            if m_end.embedding and m_next.embedding:
                chunk.boundary_similarity = cosine_similarity(m_end.embedding, m_next.embedding)

        chunks.append(chunk)
        i += 1

    return chunks


def compute_internal_similarity(chunk: Chunk) -> Optional[float]:
    """Compute average pairwise similarity within a chunk (coherence measure).

    Optimized: Uses batch similarity for O(n) instead of O(n²) loop.
    """
    embeddings = [m.embedding for m in chunk.messages if m.embedding]
    if len(embeddings) < 2:
        return None

    # Use batch computation
    emb_array = np.array(embeddings)
    sim_matrix = batch_cosine_similarity(emb_array)

    # Get upper triangle (excluding diagonal) for unique pairs
    n = len(embeddings)
    upper_tri_indices = np.triu_indices(n, k=1)
    sims = sim_matrix[upper_tri_indices]

    return float(np.mean(sims)) if len(sims) > 0 else None


def generate_summary(client: OpenAI, chunk_text: str) -> str:
    """Generate a brief topic summary for a chunk."""
    # Truncate if too long
    if len(chunk_text) > 12000:
        chunk_text = chunk_text[:12000] + "\n... [truncated]"

    try:
        response = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this conversation segment in 1-2 sentences. Focus on the main topic or task being discussed. Be specific and concrete. Start directly with the topic, no preamble."
                },
                {
                    "role": "user",
                    "content": chunk_text
                }
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    Summary generation failed: {e}")
        return ""


def generate_embedding(client: OpenAI, text: str) -> List[float]:
    """Generate embedding for chunk text."""
    # Truncate if needed - 8192 token limit, ~4 chars/token for code = 6000 char safe limit
    if len(text) > 6000:
        text = text[:6000] + "... [truncated]"

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def get_conversations_to_process(conn, source_type: str, limit: int = 100,
                                  worker_id: int = 0, num_workers: int = 1) -> List[str]:
    """Get conversation IDs that haven't been chunked yet.

    Filters to conversations with at least MIN_CHUNK_MESSAGES embedded messages.
    When num_workers > 1, partitions work using modulo on conversation_id hash.
    """
    with conn.cursor() as cur:
        if source_type == 'web':
            # Web: Join through claude_web_messages to count embedded messages
            base_query = """
                WITH eligible AS (
                    SELECT m.conversation_uuid as conversation_id,
                           COUNT(*) as embedded_count
                    FROM claude_web_messages m
                    INNER JOIN message_embeddings e ON e.source_uuid = m.uuid
                    WHERE m.content IS NOT NULL AND m.content != ''
                    GROUP BY m.conversation_uuid
                    HAVING COUNT(*) >= %s
                )
                SELECT e.conversation_id
                FROM eligible e
                LEFT JOIN conversation_topics t ON t.conversation_id = e.conversation_id::text
                WHERE t.topic_id IS NULL
            """
        else:
            # Code: Join through claude_code_raw to count embedded messages
            base_query = """
                WITH eligible AS (
                    SELECT r.conversation_id,
                           COUNT(*) as embedded_count
                    FROM claude_code_raw r
                    INNER JOIN message_embeddings e ON e.source_uuid = r.record_uuid::text
                    WHERE r.record_type IN ('user', 'assistant')
                      AND (r.data->>'message' IS NOT NULL OR r.data->>'content' IS NOT NULL)
                      AND COALESCE(r.data->>'message', r.data->>'content', '') != ''
                    GROUP BY r.conversation_id
                    HAVING COUNT(*) >= %s
                )
                SELECT e.conversation_id
                FROM eligible e
                LEFT JOIN conversation_topics t ON t.conversation_id = e.conversation_id::text
                WHERE t.topic_id IS NULL
            """

        if num_workers > 1:
            query = base_query + """
                  AND MOD(ABS(HASHTEXT(e.conversation_id::text)), %s) = %s
                ORDER BY e.embedded_count DESC
                LIMIT %s
            """
            cur.execute(query, (MIN_CHUNK_MESSAGES, num_workers, worker_id, limit))
        else:
            query = base_query + """
                ORDER BY e.embedded_count DESC
                LIMIT %s
            """
            cur.execute(query, (MIN_CHUNK_MESSAGES, limit))

        return [row[0] for row in cur.fetchall()]


def process_conversation(conn, client: OpenAI, conversation_id: str,
                        source_type: str, verbose: bool = True) -> int:
    """
    Process a single conversation: detect topics, create chunks, summarize, embed.
    Returns number of chunks created.
    """
    # Get messages with embeddings
    if source_type == 'web':
        messages = get_conversation_messages_web(conn, conversation_id)
    else:
        messages = get_conversation_messages_code(conn, conversation_id)

    if len(messages) < MIN_CHUNK_MESSAGES:
        # Too short to chunk meaningfully
        return 0

    # Check if we have enough embeddings
    embedded_count = sum(1 for m in messages if m.embedding)
    if embedded_count < len(messages) * 0.5:
        # Less than 50% embedded, skip for now
        if verbose:
            print(f"    Skipping {conversation_id[:8]}... - only {embedded_count}/{len(messages)} embedded")
        return 0

    # Detect topic boundaries
    boundaries = detect_topic_boundaries(messages)

    # Create chunks
    chunks = create_chunks(messages, boundaries)

    if not chunks:
        return 0

    if verbose:
        print(f"    {len(messages)} messages -> {len(chunks)} chunks")

    # Process each chunk
    data = []
    for chunk in chunks:
        # Generate summary
        summary = generate_summary(client, chunk.text)

        # Generate embedding
        try:
            embedding = generate_embedding(client, chunk.text)
        except Exception as e:
            print(f"    Embedding failed for chunk: {e}")
            continue

        # Compute internal similarity
        internal_sim = compute_internal_similarity(chunk)

        data.append((
            conversation_id,
            source_type,
            chunk.start_idx,
            chunk.end_idx,
            chunk.messages[0].uuid if chunk.messages else None,
            chunk.messages[-1].uuid if chunk.messages else None,
            summary,
            chunk.text,
            len(chunk.messages),
            embedding,
            EMBEDDING_MODEL,
            internal_sim,
            chunk.boundary_similarity
        ))

    # Insert chunks
    if data:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO conversation_topics (
                    conversation_id, source_type, start_msg_idx, end_msg_idx,
                    start_msg_uuid, end_msg_uuid, topic_summary, chunk_text,
                    message_count, embedding, model, avg_internal_sim, boundary_sim
                ) VALUES %s
                ON CONFLICT (conversation_id, start_msg_idx, end_msg_idx) DO NOTHING
            """, data, template="""(
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s
            )""")
        conn.commit()

    return len(data)


def chunk_conversations(source_type: str = 'web', limit: int = 100,
                       verbose: bool = True, worker_id: int = 0,
                       num_workers: int = 1) -> dict:
    """
    Main entry point. Process conversations that haven't been chunked yet.

    Args:
        worker_id: This worker's ID (0 to num_workers-1)
        num_workers: Total number of parallel workers
    """
    stats = {"conversations": 0, "chunks": 0, "errors": 0}

    conn = psycopg2.connect(**DB_CONFIG)
    client = get_openai_client()

    try:
        ensure_table(conn)

        # Get conversations to process (partitioned if using workers)
        conversation_ids = get_conversations_to_process(
            conn, source_type, limit, worker_id, num_workers
        )
        if verbose:
            print(f"Found {len(conversation_ids)} {source_type} conversations to process")

        for conv_id in conversation_ids:
            try:
                if verbose:
                    print(f"  Processing {conv_id[:8] if conv_id else 'unknown'}...")

                chunks_created = process_conversation(conn, client, conv_id, source_type, verbose)

                if chunks_created > 0:
                    stats["conversations"] += 1
                    stats["chunks"] += chunks_created

            except Exception as e:
                stats["errors"] += 1
                print(f"  Error processing {conv_id}: {e}")

    finally:
        conn.close()

    if verbose:
        print(f"\nDone. Conversations: {stats['conversations']}, Chunks: {stats['chunks']}, Errors: {stats['errors']}")

    return stats


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Chunk conversations by topic")
    parser.add_argument("--source", choices=['web', 'code', 'both'], default='both',
                       help="Which source to process")
    parser.add_argument("--limit", type=int, default=100,
                       help="Max conversations per source")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD,
                       help=f"Similarity threshold for topic boundaries (default: {SIMILARITY_THRESHOLD})")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--continuous", action="store_true",
                       help="Loop until all conversations processed")
    parser.add_argument("--worker", type=int, default=0,
                       help="Worker ID (0 to num-workers-1)")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Total number of parallel workers")
    args = parser.parse_args()

    # Update threshold if specified (passed to functions via default args)
    threshold = args.threshold

    sources = ['web', 'code'] if args.source == 'both' else [args.source]

    if args.num_workers > 1 and not args.quiet:
        print(f"Worker {args.worker}/{args.num_workers}")

    while True:
        total_chunks = 0

        for source in sources:
            if not args.quiet:
                print(f"\n{'='*50}")
                print(f"Processing {source} conversations")
                print(f"{'='*50}")

            stats = chunk_conversations(
                source_type=source,
                limit=args.limit,
                verbose=not args.quiet,
                worker_id=args.worker,
                num_workers=args.num_workers
            )
            total_chunks += stats["chunks"]

        if not args.continuous or total_chunks == 0:
            break

        if not args.quiet:
            print(f"\nContinuing... (processed {total_chunks} chunks this round)")


if __name__ == "__main__":
    main()
