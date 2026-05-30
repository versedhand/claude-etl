"""
Claude Web conversation ingestion — v2.

Processes ZIP exports from ~/corpus/isaac-workspace-corpus/var/inbox/claude-web-exports/
and loads directly into conversationsdb_v2.

Usage:
    python3 -m v2.web_ingest              # Process all ZIPs in inbox
    python3 -m v2.web_ingest --file X.zip # Process specific file
    python3 -m v2.web_ingest --dry-run    # Preview without ingesting
"""

import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values

INBOX_PATH = Path.home() / "corpus/isaac-workspace-corpus/var/inbox/claude-web-exports"
PROCESSED_PATH = INBOX_PATH / "processed"

DB_CONFIG = {
    "host": os.environ.get("CONVERSATIONS_DB_HOST", "100.127.104.75"),
    "port": int(os.environ.get("CONVERSATIONS_DB_PORT", "5432")),
    "dbname": os.environ.get("CONVERSATIONS_DB_NAME", "conversationsdb_v2"),
    "user": os.environ.get("CONVERSATIONS_DB_USER", "conversations_writer"),
    "password": os.environ.get("CONVERSATIONS_DB_PASSWORD", ""),
}


def sanitize(obj):
    """Remove null bytes from strings recursively."""
    if isinstance(obj, str):
        return obj.replace('\x00', '')
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(i) for i in obj]
    return obj


def parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except Exception:
        return None


def extract_content(blocks) -> str:
    """Extract text from Claude web content blocks."""
    if not blocks:
        return ""
    parts = []
    for b in blocks:
        text = b.get('text', '')
        if text:
            btype = b.get('type', '')
            if btype and btype != 'text':
                parts.append(f"[{btype}] {text}")
            else:
                parts.append(text)
    return '\n\n'.join(parts)


def process_zip(conn, zip_path: Path) -> dict:
    """Process one ZIP export file into conversationsdb_v2."""
    stats = {'conversations': 0, 'inserted': 0, 'updated': 0, 'skipped': 0,
             'messages': 0, 'errors': 0}

    try:
        with zipfile.ZipFile(zip_path) as z:
            data = json.loads(z.read('conversations.json'))
    except Exception as e:
        print(f"  Error reading {zip_path.name}: {e}")
        stats['errors'] += 1
        return stats

    for conv_data in data:
        conv_data = sanitize(conv_data)
        conv_uuid = conv_data.get('uuid')
        if not conv_uuid:
            stats['errors'] += 1
            continue

        name = conv_data.get('name') or conv_data.get('summary') or 'Untitled'
        created_at = parse_dt(conv_data.get('created_at'))
        updated_at = parse_dt(conv_data.get('updated_at'))
        model = conv_data.get('model')

        # Parse messages
        chat_messages = conv_data.get('chat_messages', [])
        msgs = []
        for i, m in enumerate(chat_messages):
            m = sanitize(m)
            sender = m.get('sender', 'unknown')
            role = 'assistant' if sender == 'assistant' else 'user' if sender == 'human' else sender
            content = extract_content(m.get('content', []))
            if not content:
                content = m.get('text', '')
            ts = parse_dt(m.get('created_at'))
            ch = hashlib.sha256(content.encode('utf-8')).hexdigest() if content else None
            msgs.append((
                m.get('uuid'), m.get('parent_message_uuid'),
                role, content, ts, i, ch
            ))

        stats['conversations'] += 1

        with conn.cursor() as cur:
            # Check existing
            cur.execute(
                "SELECT id, last_message_at FROM conversations WHERE session_id = %s AND machine = %s AND user_name = %s",
                (conv_uuid, 'web', 'isaac')
            )
            existing = cur.fetchone()

            if existing:
                conv_id, existing_updated = existing
                # Skip if existing is same or newer
                if existing_updated and updated_at and existing_updated >= updated_at:
                    stats['skipped'] += 1
                    continue
                # Update
                cur.execute("""
                    UPDATE conversations SET
                        project = %s, model = %s, started_at = %s, last_message_at = %s,
                        message_count = %s, ingested_at = NOW(), source_file = %s
                    WHERE id = %s
                """, (name, model, created_at, updated_at, len(msgs), zip_path.name, conv_id))
                cur.execute("DELETE FROM messages WHERE conversation_id = %s", (conv_id,))
                stats['updated'] += 1
            else:
                cur.execute("""
                    INSERT INTO conversations
                        (session_id, user_name, machine, project, model, source_file,
                         started_at, last_message_at, message_count, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (conv_uuid, 'isaac', 'web', name, model, zip_path.name,
                      created_at, updated_at, len(msgs), 'web'))
                conv_id = cur.fetchone()[0]
                stats['inserted'] += 1

            # Batch insert messages
            if msgs:
                rows = [(conv_id, uuid, parent, role, content, None, None, None, False, ts, seq, ch)
                        for uuid, parent, role, content, ts, seq, ch in msgs]
                execute_values(cur, """
                    INSERT INTO messages
                        (conversation_id, uuid, parent_uuid, role, content,
                         thinking, tool_calls, tool_results, is_sidechain,
                         timestamp, sequence_num, content_hash)
                    VALUES %s
                """, rows, page_size=500)
                stats['messages'] += len(rows)

        conn.commit()

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Claude Web exports into conversationsdb_v2")
    parser.add_argument("--file", help="Process specific ZIP file")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--limit", type=int, help="Limit files to process")
    args = parser.parse_args()

    print(f"Claude Web ETL v2 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Inbox: {INBOX_PATH}")

    if args.file:
        zips = [INBOX_PATH / args.file]
    else:
        zips = sorted(INBOX_PATH.glob("*.zip"))

    if args.limit:
        zips = zips[:args.limit]

    print(f"Found {len(zips)} ZIP file(s)")

    if not zips:
        print("Nothing to process.")
        return

    if args.dry_run:
        for zf in zips:
            try:
                with zipfile.ZipFile(zf) as z:
                    convos = json.loads(z.read('conversations.json'))
                    mc = sum(len(c.get('chat_messages', [])) for c in convos)
                    print(f"  {zf.name}: {len(convos)} conversations, {mc} messages")
            except Exception as e:
                print(f"  {zf.name}: Error — {e}")
        return

    conn = psycopg2.connect(**DB_CONFIG)

    totals = {'conversations': 0, 'inserted': 0, 'updated': 0, 'skipped': 0, 'messages': 0, 'errors': 0}
    processed = []

    for zf in zips:
        print(f"\nProcessing {zf.name}...")
        s = process_zip(conn, zf)
        for k in totals:
            totals[k] += s[k]
        print(f"  {s['conversations']} convos: {s['inserted']} new, {s['updated']} updated, "
              f"{s['skipped']} skipped | {s['messages']} msgs | {s['errors']} errors")
        if s['errors'] == 0:
            processed.append(zf)

    print(f"\nTotal: {totals['conversations']} convos, {totals['messages']} msgs, {totals['errors']} errors")

    # Move processed files
    if processed:
        PROCESSED_PATH.mkdir(exist_ok=True)
        for zf in processed:
            dest = PROCESSED_PATH / zf.name
            zf.rename(dest)
            print(f"  {zf.name} → processed/")

    conn.close()


if __name__ == "__main__":
    main()
