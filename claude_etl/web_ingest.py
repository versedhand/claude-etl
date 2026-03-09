#!/usr/bin/env python3
"""
Claude Web conversation ingestion script.

Processes ZIP exports from /life-var/inbox/claude-web-exports/ and loads into LifeDB.
Branching logic adapted from claude-chats/parsers/claude_export.py.
"""

import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import psycopg2
from psycopg2.extras import Json

# Configuration
# Auto-detect inbox path based on environment
if Path("/mnt/d/obs").exists():  # WSL/Windows
    INBOX_PATH = Path("/mnt/d/obs/life-var/inbox/claude-web-exports")
elif Path("/srv/obs").exists():  # Linux server
    INBOX_PATH = Path("/srv/obs/life-var/inbox/claude-web-exports")
else:  # Fallback to home
    INBOX_PATH = Path.home() / "obs/life-var/inbox/claude-web-exports"
PROCESSED_PATH = INBOX_PATH / "processed"
# CUTOFF_DATE removed - all conversations go to claude_web_raw now (2026-01-21)
# Archive table is deprecated, data migrated to raw

DB_CONFIG = {
    "host": "100.127.104.75",
    "port": 5432,
    "dbname": "lifedb",
    "user": "postgres",
    "password": os.environ.get("LIFEDB_PASSWORD", "StrongPassword123"),
}


def sanitize_null_bytes(obj):
    """Recursively remove null bytes from strings in a JSON object."""
    if isinstance(obj, str):
        return obj.replace('\x00', '')
    elif isinstance(obj, dict):
        return {k: sanitize_null_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_null_bytes(item) for item in obj]
    return obj


def parse_datetime(dt_string: str) -> Optional[datetime]:
    """Parse ISO 8601 datetime string."""
    if not dt_string:
        return None
    try:
        return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
    except Exception:
        return None


def extract_content(content_blocks: List[Dict[str, Any]]) -> str:
    """
    Extract text content from content blocks.

    Handles: text, thinking, tool_use, tool_result
    """
    if not content_blocks:
        return ""

    text_parts = []
    for block in content_blocks:
        block_type = block.get('type', '')
        text = block.get('text', '')

        if text:
            if block_type and block_type != 'text':
                text_parts.append(f"[{block_type}] {text}")
            else:
                text_parts.append(text)

    return '\n\n'.join(text_parts)


def parse_message(msg_data: dict, conversation_uuid: str) -> dict:
    """Parse a single message from JSON data."""
    uuid = msg_data.get('uuid')
    sender = msg_data.get('sender')
    created_at = parse_datetime(msg_data.get('created_at'))
    updated_at = parse_datetime(msg_data.get('updated_at'))
    parent_message_uuid = msg_data.get('parent_message_uuid')

    # Extract content from content blocks
    content = extract_content(msg_data.get('content', []))

    # Fallback to 'text' field if content is empty
    if not content and msg_data.get('text'):
        content = msg_data['text']

    # Build metadata
    metadata = {}
    if msg_data.get('attachments'):
        metadata['attachments'] = msg_data['attachments']
    if msg_data.get('files'):
        metadata['files'] = msg_data['files']

    return {
        'uuid': uuid,
        'conversation_uuid': conversation_uuid,
        'sender': sender,
        'content': content,
        'created_at': created_at,
        'updated_at': updated_at,
        'parent_message_uuid': parent_message_uuid,
        'metadata': metadata,
        'raw_data': msg_data,
    }


def parse_conversation(conv_data: dict) -> dict:
    """Parse a single conversation from JSON data."""
    uuid = conv_data.get('uuid')
    name = conv_data.get('name') or conv_data.get('summary') or 'Untitled'
    created_at = parse_datetime(conv_data.get('created_at'))
    updated_at = parse_datetime(conv_data.get('updated_at'))

    # Parse messages
    messages = []
    for msg_data in conv_data.get('chat_messages', []):
        try:
            message = parse_message(msg_data, uuid)
            messages.append(message)
        except Exception as e:
            print(f"  Warning: Skipping message in {uuid}: {e}")
            continue

    # Build metadata
    metadata = {
        'account_uuid': conv_data.get('account', {}).get('uuid'),
        'summary': conv_data.get('summary', ''),
    }

    return {
        'uuid': uuid,
        'name': name,
        'created_at': created_at,
        'updated_at': updated_at,
        'messages': messages,
        'metadata': metadata,
        'raw_data': conv_data,
    }


def ingest_conversation(cur, conv: dict, source_file: str) -> tuple[int, int, int, int]:
    """
    Ingest a single conversation.

    Returns (conv_inserted, conv_updated, msgs_inserted, msgs_updated).

    Logic:
    - If conversation not found → INSERT
    - If found and new updated_at > existing → UPDATE
    - If found and new updated_at <= existing → SKIP (return 0s)
    """
    conv = sanitize_null_bytes(conv)

    # All conversations go to claude_web_raw (archive distinction removed 2026-01-21)
    table = 'claude_web_raw'

    # Check if conversation exists and compare updated_at
    cur.execute(f"""
        SELECT updated_at FROM {table} WHERE conversation_id = %s::uuid
    """, (conv['uuid'],))
    existing = cur.fetchone()

    if existing:
        existing_updated = existing[0]
        # Skip if existing data is same or newer
        if existing_updated and conv['updated_at'] and existing_updated >= conv['updated_at']:
            return 0, 0, 0, 0  # Skip - existing is newer or same

    # Extract account info from raw_data
    account = conv['raw_data'].get('account', {})
    account_id = account.get('uuid', '')
    account_email = account.get('email_address', '')

    if existing:
        # UPDATE existing row with newer data
        cur.execute(f"""
            UPDATE {table} SET
                name = %s,
                updated_at = %s,
                data = %s,
                source_file = %s,
                ingested_at = NOW()
            WHERE conversation_id = %s::uuid
        """, (
            conv['name'],
            conv['updated_at'],
            Json(conv['raw_data']),
            source_file,
            conv['uuid'],
        ))
        conv_inserted = 0
        conv_updated = 1
    else:
        # INSERT new conversation
        cur.execute(f"""
            INSERT INTO {table}
            (conversation_id, account_id, account_email, export_date, name, created_at, updated_at, data, source_file, ingested_at)
            VALUES (%s::uuid, %s, %s, CURRENT_DATE, %s, %s, %s, %s, %s, NOW())
        """, (
            conv['uuid'],
            account_id,
            account_email,
            conv['name'],
            conv['created_at'],
            conv['updated_at'],
            Json(conv['raw_data']),
            source_file,
        ))
        conv_inserted = 1
        conv_updated = 0

    # Insert messages
    msgs_inserted = 0
    msgs_updated = 0

    for msg in conv['messages']:
        msg = sanitize_null_bytes(msg)

        cur.execute("""
            INSERT INTO claude_web_messages
            (uuid, conversation_uuid, sender, content, created_at, updated_at,
             parent_message_uuid, metadata, raw_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (uuid) DO UPDATE SET
                content = EXCLUDED.content,
                updated_at = EXCLUDED.updated_at,
                metadata = EXCLUDED.metadata,
                raw_data = EXCLUDED.raw_data
            RETURNING (xmax = 0) AS inserted
        """, (
            msg['uuid'],
            msg['conversation_uuid'],
            msg['sender'],
            msg['content'],
            msg['created_at'],
            msg['updated_at'],
            msg['parent_message_uuid'],
            Json(msg['metadata']),
            Json(msg['raw_data']),
        ))

        result = cur.fetchone()
        if result:
            if result[0]:
                msgs_inserted += 1
            else:
                msgs_updated += 1

    return conv_inserted, conv_updated, msgs_inserted, msgs_updated


def process_zip(conn, zip_path: Path, batch_size: int = 100) -> dict:
    """Process a single ZIP file with periodic commits."""
    stats = {
        'conversations': 0,
        'conv_inserted': 0,
        'conv_updated': 0,
        'conv_skipped': 0,
        'msgs_inserted': 0,
        'msgs_updated': 0,
        'errors': 0
    }

    try:
        with zipfile.ZipFile(zip_path) as z:
            convos_json = z.read('conversations.json')
            data = json.loads(convos_json)
    except Exception as e:
        print(f"  Error reading {zip_path.name}: {e}")
        stats['errors'] += 1
        return stats

    total_convos = len(data)

    for i, conv_data in enumerate(data):
        try:
            with conn.cursor() as cur:
                conv = parse_conversation(conv_data)
                ci, cu, mi, mu = ingest_conversation(cur, conv, zip_path.name)

                stats['conversations'] += 1
                stats['conv_inserted'] += ci
                stats['conv_updated'] += cu
                stats['msgs_inserted'] += mi
                stats['msgs_updated'] += mu

                # Track skipped (neither inserted nor updated)
                if ci == 0 and cu == 0:
                    stats['conv_skipped'] += 1

            # Commit after each conversation to avoid memory buildup
            conn.commit()

            # Progress indicator every batch_size conversations
            if (i + 1) % batch_size == 0:
                print(f"    Progress: {i + 1}/{total_convos} ({100 * (i + 1) // total_convos}%)")

        except Exception as e:
            stats['errors'] += 1
            conv_uuid = conv_data.get('uuid', 'unknown')[:8]
            print(f"  Error processing {conv_uuid}: {e}")
            # Try to recover connection
            try:
                conn.rollback()
            except Exception:
                pass
            continue

    return stats


def ensure_tables(conn):
    """Verify tables exist and create indexes."""
    with conn.cursor() as cur:
        # Check that required tables exist (don't create - use existing schema)
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name IN ('claude_web_raw', 'claude_web_archive', 'claude_web_messages')
        """)
        existing = {row[0] for row in cur.fetchall()}

        missing = {'claude_web_raw', 'claude_web_messages'} - existing
        if missing:
            raise RuntimeError(f"Required tables missing: {missing}. Create them first.")
        # Note: claude_web_archive exists but is deprecated (data migrated to raw 2026-01-21)

        # Add unique constraint if not exists (for UPSERT)
        cur.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'claude_web_raw_conversation_id_key') THEN
                    ALTER TABLE claude_web_raw ADD CONSTRAINT claude_web_raw_conversation_id_key UNIQUE (conversation_id);
                END IF;
            END $$;
        """)
        # Archive constraint check removed - table deprecated 2026-01-21

        # Ensure messages table has primary key
        cur.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'claude_web_messages_pkey') THEN
                    ALTER TABLE claude_web_messages ADD PRIMARY KEY (uuid);
                END IF;
            END $$;
        """)

        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cwm_conversation ON claude_web_messages(conversation_uuid)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cwm_parent ON claude_web_messages(parent_message_uuid)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cwm_created ON claude_web_messages(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cwr_created ON claude_web_raw(created_at)")
        # Archive index removed - table deprecated 2026-01-21

    conn.commit()
    print("Tables verified, constraints and indexes ensured")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Claude Web exports into LifeDB")
    parser.add_argument("--limit", type=int, help="Limit number of ZIP files to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--file", type=str, help="Process specific ZIP file")
    args = parser.parse_args()

    print(f"Claude Web ETL - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Inbox: {INBOX_PATH}")
    print()

    # Find ZIP files
    if args.file:
        zip_files = [INBOX_PATH / args.file]
    else:
        zip_files = sorted(INBOX_PATH.glob("*.zip"))

    if args.limit:
        zip_files = zip_files[:args.limit]

    print(f"Found {len(zip_files)} ZIP file(s)")

    if args.dry_run:
        print("\nDRY RUN - counting contents only")
        for zf in zip_files:
            try:
                with zipfile.ZipFile(zf) as z:
                    convos = json.loads(z.read('conversations.json'))
                    msg_count = sum(len(c.get('chat_messages', [])) for c in convos)
                    print(f"  {zf.name}: {len(convos)} conversations, {msg_count} messages")
            except Exception as e:
                print(f"  {zf.name}: Error - {e}")
        return

    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected to LifeDB")
    except Exception as e:
        print(f"Failed to connect to LifeDB: {e}")
        sys.exit(1)

    try:
        ensure_tables(conn)

        total = {
            'conversations': 0,
            'conv_inserted': 0,
            'conv_updated': 0,
            'conv_skipped': 0,
            'msgs_inserted': 0,
            'msgs_updated': 0,
            'errors': 0
        }

        processed_files = []

        for zf in zip_files:
            print(f"\nProcessing {zf.name}...")
            stats = process_zip(conn, zf)

            for key in total:
                total[key] += stats[key]

            print(f"  {stats['conversations']} convos: "
                  f"{stats['conv_inserted']} new, {stats['conv_updated']} updated, {stats['conv_skipped']} skipped | "
                  f"msgs: {stats['msgs_inserted']} new, {stats['msgs_updated']} updated | "
                  f"{stats['errors']} errors")

            # Track successfully processed files (0 errors)
            if stats['errors'] == 0:
                processed_files.append(zf)

        print(f"\n{'='*60}")
        print(f"Total: {total['conversations']} conversations processed")
        print(f"  Conversations: {total['conv_inserted']} new, {total['conv_updated']} updated, {total['conv_skipped']} skipped")
        print(f"  Messages: {total['msgs_inserted']} new, {total['msgs_updated']} updated")
        print(f"  Errors: {total['errors']}")

        # Move successfully processed files to processed folder
        if processed_files and not args.dry_run:
            PROCESSED_PATH.mkdir(exist_ok=True)
            print(f"\nMoving {len(processed_files)} file(s) to {PROCESSED_PATH}/")
            for zf in processed_files:
                dest = PROCESSED_PATH / zf.name
                zf.rename(dest)
                print(f"  {zf.name} → processed/")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
