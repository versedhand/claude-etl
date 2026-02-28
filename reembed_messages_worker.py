#!/usr/bin/env python3
"""
Parallel worker for re-embedding message_embeddings with text-embedding-3-large.
Usage: python3 reembed_messages_worker.py --worker 0 --num-workers 24
"""

import argparse
import os
import sys
from pathlib import Path
import psycopg2

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI not installed. Run: pip install openai")
    sys.exit(1)

# Configuration
DB_CONFIG = {
    "host": "100.86.218.108",
    "port": 5432,
    "dbname": "personal_data",
    "user": "postgres",
    "password": os.environ.get("LIFEDB_PASSWORD", "StrongPassword123"),
}

API_KEY_PATHS = [
    Path("/mnt/d/obs/life-etc/api-keys/openai.key"),
    Path("/srv/obs/life-etc/api-keys/openai.key"),
]

EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 50


def get_openai_client() -> OpenAI:
    for path in API_KEY_PATHS:
        if path.exists():
            return OpenAI(api_key=path.read_text().strip())
    raise ValueError("No OpenAI API key found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True, help='Worker ID (0 to num-workers-1)')
    parser.add_argument('--num-workers', type=int, default=24, help='Total workers')
    args = parser.parse_args()

    client = get_openai_client()
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get messages for this worker using modulo partitioning on row number
    cur.execute("""
        WITH numbered AS (
            SELECT source_uuid, source_type,
                   ROW_NUMBER() OVER (ORDER BY source_uuid) as rn
            FROM message_embeddings
        )
        SELECT n.source_uuid, n.source_type,
               COALESCE(c.content, w.content) as content
        FROM numbered n
        LEFT JOIN claude_code_messages c ON n.source_type = 'code' AND n.source_uuid = c.uuid
        LEFT JOIN claude_web_messages w ON n.source_type = 'web' AND n.source_uuid = w.uuid
        WHERE (n.rn - 1) %% %s = %s
        ORDER BY n.rn
    """, (args.num_workers, args.worker))

    messages = cur.fetchall()
    print(f"Worker {args.worker}: {len(messages)} messages to process")

    # Process in batches
    total_done = 0
    for i in range(0, len(messages), BATCH_SIZE):
        batch = messages[i:i+BATCH_SIZE]
        uuids = [m[0] for m in batch]
        contents = [m[2] or "" for m in batch]

        # Skip empty content
        valid_batch = [(u, c) for u, c in zip(uuids, contents) if c.strip()]
        if not valid_batch:
            continue

        uuids = [v[0] for v in valid_batch]
        contents = [v[1] for v in valid_batch]

        # Get embeddings
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=contents)
            embeddings = [e.embedding for e in response.data]
        except Exception as e:
            print(f"Worker {args.worker}: Error embedding batch: {e}")
            continue

        # Update database
        for uuid, embedding in zip(uuids, embeddings):
            cur.execute(
                "UPDATE message_embeddings SET embedding = %s, model = %s WHERE source_uuid = %s",
                (embedding, EMBEDDING_MODEL, uuid)
            )

        conn.commit()
        total_done += len(valid_batch)
        if total_done % 500 == 0:
            print(f"Worker {args.worker}: {total_done}/{len(messages)} complete")

    print(f"Worker {args.worker}: Done! ({total_done} embedded)")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
