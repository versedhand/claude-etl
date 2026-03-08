#!/usr/bin/env python3
"""
Full Claude ETL pipeline - run this after dropping a new web export.

Usage:
    python3 process_all.py           # Full pipeline
    python3 process_all.py --dry-run # Show what would be processed
    python3 process_all.py --status  # Just show current stats

Steps:
1. Ingest web exports (moves processed to processed/)
2. Generate embeddings for new messages
3. Chunk conversations into topics with summaries
"""

import subprocess
import sys
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run_step(name: str, cmd: list, dry_run: bool = False):
    """Run a pipeline step."""
    print(f"\n{'='*60}")
    print(f"Step: {name}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  Would run: {' '.join(cmd)}")
        return True

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        return False
    return True


def get_stats():
    """Get current pipeline stats."""
    import psycopg2

    DB_CONFIG = {
        "host": "100.127.104.75",
        "port": 5432,
        "dbname": "lifedb",
        "user": "postgres",
        "password": os.environ.get("LIFEDB_PASSWORD", "StrongPassword123"),
    }

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    stats = {}

    # Message counts
    cur.execute("SELECT COUNT(*) FROM claude_web_messages")
    stats['web_messages'] = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM claude_code_raw WHERE record_type IN ('user', 'assistant')")
    stats['code_messages'] = cur.fetchone()[0]

    # Embedding counts
    cur.execute("SELECT source_type, COUNT(*) FROM message_embeddings GROUP BY source_type")
    for row in cur.fetchall():
        stats[f'{row[0]}_embeddings'] = row[1]

    # Topic counts
    cur.execute("SELECT COUNT(*) FROM conversation_topics")
    stats['topics'] = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT conversation_id) FROM conversation_topics")
    stats['conversations_chunked'] = cur.fetchone()[0]

    # Conversation counts
    cur.execute("SELECT COUNT(*) FROM claude_web_conversations")
    stats['web_conversations'] = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM claude_code_conversations")
    stats['code_conversations'] = cur.fetchone()[0]

    conn.close()
    return stats


def print_stats():
    """Print current pipeline stats."""
    stats = get_stats()

    print("\n" + "="*60)
    print("Current Pipeline Stats")
    print("="*60)

    web_emb = stats.get('web_embeddings', 0)
    code_emb = stats.get('code_embeddings', 0)

    print(f"\nMessages:")
    print(f"  Web:  {stats['web_messages']:,} messages, {web_emb:,} embedded ({100*web_emb/max(1,stats['web_messages']):.0f}%)")
    print(f"  Code: {stats['code_messages']:,} messages, {code_emb:,} embedded ({100*code_emb/max(1,stats['code_messages']):.0f}%)")

    total_convs = stats['web_conversations'] + stats['code_conversations']
    print(f"\nConversations:")
    print(f"  Total: {total_convs:,} ({stats['web_conversations']:,} web + {stats['code_conversations']:,} code)")
    print(f"  Chunked: {stats['conversations_chunked']:,} ({100*stats['conversations_chunked']/max(1,total_convs):.0f}%)")
    print(f"  Topics: {stats['topics']:,}")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run full Claude ETL pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--status", action="store_true", help="Just show stats")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip web ingestion")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding")
    parser.add_argument("--skip-chunk", action="store_true", help="Skip topic chunking")
    args = parser.parse_args()

    if args.status:
        print_stats()
        return

    print("Claude ETL Pipeline")
    print("="*60)

    if args.dry_run:
        print("DRY RUN - no changes will be made")

    # Step 1: Ingest web exports
    if not args.skip_ingest:
        if not run_step("Ingest Web Exports", ["python3", "web_ingest.py"], args.dry_run):
            print("\nIngestion failed, but continuing...")

    # Step 2: Generate embeddings
    if not args.skip_embed:
        run_step("Generate Embeddings", ["python3", "embed.py", "--continuous"], args.dry_run)

    # Step 3: Chunk conversations
    if not args.skip_chunk:
        run_step("Chunk Conversations", ["python3", "chunk_conversations.py", "--continuous"], args.dry_run)

    # Final stats
    if not args.dry_run:
        print_stats()

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
