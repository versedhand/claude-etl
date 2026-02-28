#!/usr/bin/env python3
"""
Re-embed conversation_topics with text-embedding-3-large (3072 dims).
One-time migration script.
"""

import os
import sys
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values

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
BATCH_SIZE = 50  # Smaller batches for large embeddings


def get_openai_client() -> OpenAI:
    for path in API_KEY_PATHS:
        if path.exists():
            return OpenAI(api_key=path.read_text().strip())
    raise ValueError("No OpenAI API key found")


def main():
    client = get_openai_client()
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get all topics (topic_summary is what we embed)
    cur.execute("SELECT topic_id, topic_summary FROM conversation_topics WHERE topic_summary IS NOT NULL ORDER BY topic_id")
    topics = cur.fetchall()
    print(f"Found {len(topics)} topics to re-embed")

    # Process in batches
    total_done = 0
    for i in range(0, len(topics), BATCH_SIZE):
        batch = topics[i:i+BATCH_SIZE]
        ids = [t[0] for t in batch]
        summaries = [t[1] for t in batch]

        # Get embeddings
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=summaries)
        embeddings = [e.embedding for e in response.data]

        # Update database
        for topic_id, embedding in zip(ids, embeddings):
            cur.execute(
                "UPDATE conversation_topics SET embedding = %s, model = %s WHERE topic_id = %s",
                (embedding, EMBEDDING_MODEL, topic_id)
            )

        conn.commit()
        total_done += len(batch)
        print(f"  {total_done}/{len(topics)} complete")

    cur.close()
    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
