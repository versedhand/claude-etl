#!/usr/bin/env python3
"""Generate embeddings for all embeddable messages that don't have them yet.

Batches requests to OpenAI API. Tracks progress. Safe to interrupt and resume.
"""

import logging
import os
import sys
import time

import psycopg2
from openai import OpenAI

from .embed import (
    should_embed,
    content_hash,
    find_embeddable_messages,
    store_embedding,
    get_embedding_stats,
    compute_hashes_for_messages,
)
from .db import get_connection, DB_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    open(os.path.expanduser("~/corpus/isaac-workspace-corpus/etc/api-keys/openai.key")).read().strip()
)
MODEL = "text-embedding-3-large"
BATCH_SIZE = 100  # OpenAI supports up to 2048 inputs per request
MAX_TOKENS_PER_BATCH = 500_000  # stay well under rate limits


def generate_embeddings_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts."""
    response = client.embeddings.create(
        model=MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def main():
    conn = get_connection()
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Step 1: Compute content hashes for any messages missing them
    logger.info("Computing content hashes...")
    compute_hashes_for_messages(conn)

    # Step 2: Find messages that need embedding
    logger.info("Finding embeddable messages...")
    candidates = find_embeddable_messages(conn)

    if not candidates:
        stats = get_embedding_stats(conn)
        logger.info(f"Nothing to embed. Stats: {stats}")
        conn.close()
        return

    logger.info(f"{len(candidates)} messages to embed")

    # Step 3: Batch embed
    total_embedded = 0
    total_tokens = 0
    start_time = time.time()

    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i:i + BATCH_SIZE]
        texts = [c["content"] for c in batch]
        hashes = [c["content_hash"] for c in batch]

        try:
            embeddings = generate_embeddings_batch(client, texts)

            for ch, text, emb in zip(hashes, texts, embeddings):
                store_embedding(conn, ch, text[:200], emb)
                total_embedded += 1

            # Rough token estimate (4 chars per token)
            batch_chars = sum(len(t) for t in texts)
            batch_tokens = batch_chars // 4
            total_tokens += batch_tokens

            elapsed = time.time() - start_time
            rate = total_embedded / elapsed if elapsed > 0 else 0
            cost_estimate = total_tokens / 1_000_000 * 0.13

            logger.info(
                f"Batch {i // BATCH_SIZE + 1}: "
                f"{total_embedded}/{len(candidates)} embedded "
                f"(~{total_tokens:,} tokens, ~${cost_estimate:.3f}, "
                f"{rate:.0f}/sec)"
            )

        except Exception as e:
            logger.error(f"Batch failed at offset {i}: {e}")
            # Continue with next batch — already-stored embeddings are safe
            continue

    elapsed = time.time() - start_time
    cost_estimate = total_tokens / 1_000_000 * 0.13

    logger.info(
        f"Done: {total_embedded} embedded in {elapsed:.1f}s, "
        f"~{total_tokens:,} tokens, ~${cost_estimate:.3f}"
    )

    stats = get_embedding_stats(conn)
    logger.info(f"Final stats: {stats}")
    conn.close()


if __name__ == "__main__":
    main()
