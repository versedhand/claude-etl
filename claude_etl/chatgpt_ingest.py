#!/usr/bin/env python3
"""
ChatGPT conversation ingestion script.

Processes ZIP exports and loads into LifeDB with attachment extraction.
Integrates with existing Claude ETL pipeline.
"""

import json
import os
import sys
import zipfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import psycopg2
from psycopg2.extras import Json

# Configuration - auto-detect paths based on environment
if Path("/mnt/d/obs").exists():  # WSL/Windows
    BASE = Path("/mnt/d/obs")
elif Path("/srv/obs").exists():  # Linux server
    BASE = Path("/srv/obs")
else:
    BASE = Path.home() / "obs"

# Source paths
DROPBOX_SOURCE = Path("/mnt/d/gdrive/var/exports-chatgpt/zipped")
INBOX_PATH = BASE / "life-var/inbox/chatgpt-exports"
PROCESSED_PATH = INBOX_PATH / "processed"
ATTACHMENTS_PATH = BASE / "life-var/cache/chatgpt-attachments"

DB_CONFIG = {
    "host": "100.127.104.75",
    "port": 5432,
    "dbname": "lifedb",
    "user": "postgres",
    "password": os.environ["LIFEDB_PASSWORD"],
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


def unix_to_datetime(ts: Optional[float]) -> Optional[datetime]:
    """Convert Unix timestamp to datetime."""
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None


def extract_text_content(parts: List[Any]) -> str:
    """Extract text content from message parts, handling multimodal."""
    text_parts = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict):
            # Skip attachment pointers, extract any nested text
            if part.get('content_type') in ('image_asset_pointer', 'file_asset_pointer'):
                continue
            if 'text' in part:
                text_parts.append(part['text'])
    return '\n'.join(text_parts)


def find_canonical_path(mapping: Dict[str, Any]) -> List[str]:
    """
    Find the canonical message path through the conversation tree.

    ChatGPT exports have explicit parent/children pointers.
    Canonical path = follow children, taking last child at each branch.
    """
    if not mapping:
        return []

    # Find root node
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get('parent')
        if parent is None or parent == 'client-created-root':
            if node_id != 'client-created-root':
                root_id = node_id
                break

    # If no clear root, look for client-created-root's first child
    if not root_id and 'client-created-root' in mapping:
        children = mapping['client-created-root'].get('children', [])
        if children:
            root_id = children[0]

    if not root_id:
        return []

    # Walk the tree, taking last child at each branch
    path = []
    current = root_id
    visited = set()

    while current and current not in visited:
        visited.add(current)
        node = mapping.get(current)
        if not node:
            break

        # Only include nodes with actual messages
        if node.get('message'):
            path.append(current)

        children = node.get('children', [])
        if children:
            # Take last child (most recent branch)
            current = children[-1]
        else:
            break

    return path


def extract_attachments_from_message(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract attachment metadata from a message."""
    attachments = []
    content = msg.get('content', {})
    parts = content.get('parts', [])

    for part in parts:
        if not isinstance(part, dict):
            continue

        content_type = part.get('content_type', '')
        if content_type in ('image_asset_pointer', 'file_asset_pointer'):
            asset_pointer = part.get('asset_pointer', '')
            attachments.append({
                'asset_pointer': asset_pointer,
                'size_bytes': part.get('size_bytes'),
                'width': part.get('width'),
                'height': part.get('height'),
                'metadata': part.get('metadata', {}),
            })

    return attachments


def asset_pointer_to_filename(asset_pointer: str, zip_contents: List[str]) -> Optional[str]:
    """
    Map asset_pointer to actual filename in ZIP.

    Handles multiple formats:
    - sediment://file_X -> file_X-sanitized.png
    - file-service://file-XXX -> file-XXX-uuid.png
    - Nested paths like file-XXX_data/file-YYY-0.jpg
    """
    # Extract file ID from various pointer formats
    file_id = None
    if asset_pointer.startswith('sediment://'):
        file_id = asset_pointer.replace('sediment://', '')
    elif asset_pointer.startswith('file-service://'):
        file_id = asset_pointer.replace('file-service://', '')
    else:
        # Unknown format - try using it directly
        file_id = asset_pointer

    if not file_id:
        return None

    # Look for matching files - check both startswith and contains
    # Priority: direct match first, then nested/contained match
    for name in zip_contents:
        if name.startswith(file_id):
            return name

    # Check for nested paths (e.g., file-XXX_data/file-YYY-0.jpg)
    for name in zip_contents:
        if '/' in name and file_id in name:
            return name

    return None


def extract_attachment_file(
    z: zipfile.ZipFile,
    filename: str,
    conv_id: str,
    attachments_path: Path
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract attachment file to cache directory.

    Returns (file_path, file_hash) or (None, None) on failure.
    """
    try:
        # Create conversation directory
        conv_dir = attachments_path / conv_id[:8]  # Use first 8 chars of conv_id
        conv_dir.mkdir(parents=True, exist_ok=True)

        # Extract file
        data = z.read(filename)
        file_hash = hashlib.sha256(data).hexdigest()

        # Use hash prefix + original name to avoid collisions
        safe_filename = f"{file_hash[:8]}_{Path(filename).name}"
        file_path = conv_dir / safe_filename

        # Skip if already exists with same hash
        if file_path.exists():
            existing_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if existing_hash == file_hash:
                return str(file_path), file_hash

        file_path.write_bytes(data)
        return str(file_path), file_hash

    except Exception as e:
        print(f"    Warning: Failed to extract {filename}: {e}")
        return None, None


def guess_content_type(filename: str) -> str:
    """Guess content type from filename extension."""
    ext = Path(filename).suffix.lower()
    types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.pdf': 'application/pdf',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.txt': 'text/plain',
    }
    return types.get(ext, 'application/octet-stream')


def parse_conversation(conv_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a ChatGPT conversation from JSON."""
    # Generate a stable conversation ID from the data
    # ChatGPT doesn't provide explicit UUIDs, use content hash
    conv_str = json.dumps(conv_data.get('mapping', {}), sort_keys=True)
    conv_id = hashlib.sha256(conv_str.encode()).hexdigest()[:32]

    # If there's an existing ID field, prefer that
    if 'id' in conv_data:
        conv_id = conv_data['id']

    return {
        'conversation_id': conv_id,
        'title': conv_data.get('title', 'Untitled'),
        'created_at': unix_to_datetime(conv_data.get('create_time')),
        'updated_at': unix_to_datetime(conv_data.get('update_time')),
        'mapping': conv_data.get('mapping', {}),
        'raw_data': conv_data,
    }


def ingest_conversation(
    cur,
    conv: Dict[str, Any],
    source_file: str,
    z: zipfile.ZipFile,
    zip_contents: List[str],
    attachments_path: Path,
    extract_files: bool = True,
    force_attachments: bool = False
) -> Dict[str, int]:
    """
    Ingest a single conversation with messages and attachments.

    Returns stats dict.
    """
    stats = {
        'conv_inserted': 0,
        'conv_updated': 0,
        'conv_skipped': 0,
        'msgs_inserted': 0,
        'msgs_updated': 0,
        'attachments': 0,
    }

    conv = sanitize_null_bytes(conv)
    conv_id = conv['conversation_id']

    # Check if conversation exists
    cur.execute("""
        SELECT updated_at FROM chatgpt_raw WHERE conversation_id = %s
    """, (conv_id,))
    existing = cur.fetchone()

    skip_conversation = False
    if existing:
        existing_updated = existing[0]
        if existing_updated and conv['updated_at'] and existing_updated >= conv['updated_at']:
            if not force_attachments:
                stats['conv_skipped'] = 1
                return stats
            # If force_attachments, continue to process attachments only
            skip_conversation = True

    # Upsert conversation (skip if only reprocessing attachments)
    if not skip_conversation:
        if existing:
            cur.execute("""
                UPDATE chatgpt_raw SET
                    title = %s,
                    updated_at = %s,
                    data = %s,
                    source_file = %s,
                    ingested_at = NOW()
                WHERE conversation_id = %s
            """, (
                conv['title'],
                conv['updated_at'],
                Json(conv['raw_data']),
                source_file,
                conv_id,
            ))
            stats['conv_updated'] = 1
        else:
            cur.execute("""
                INSERT INTO chatgpt_raw
                (conversation_id, title, created_at, updated_at, data, source_file, ingested_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (
                conv_id,
                conv['title'],
                conv['created_at'],
                conv['updated_at'],
                Json(conv['raw_data']),
                source_file,
            ))
            stats['conv_inserted'] = 1
    else:
        stats['conv_skipped'] = 1

    # Find canonical path
    mapping = conv['mapping']
    canonical_ids = set(find_canonical_path(mapping))

    # Process messages
    for node_id, node in mapping.items():
        msg = node.get('message')
        if not msg:
            continue

        # Skip system context messages (user profile, etc.)
        content = msg.get('content', {})
        if content.get('content_type') == 'user_editable_context':
            continue

        author = msg.get('author', {})
        sender = author.get('role', 'unknown')

        # Extract text content
        parts = content.get('parts', [])
        text_content = extract_text_content(parts)

        # Skip empty messages
        if not text_content and sender == 'system':
            continue

        is_canonical = node_id in canonical_ids
        msg_created = unix_to_datetime(msg.get('create_time'))

        # Upsert message (skip if only reprocessing attachments)
        if not skip_conversation:
            cur.execute("""
                INSERT INTO chatgpt_messages
                (message_id, conversation_id, parent_id, sender, content, content_type, created_at, is_canonical, raw_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (message_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    is_canonical = EXCLUDED.is_canonical,
                    raw_data = EXCLUDED.raw_data
                RETURNING (xmax = 0) AS inserted
            """, (
                node_id,
                conv_id,
                node.get('parent'),
                sender,
                text_content,
                content.get('content_type', 'text'),
                msg_created,
                is_canonical,
                Json(msg),
            ))

            result = cur.fetchone()
            if result and result[0]:
                stats['msgs_inserted'] += 1
            else:
                stats['msgs_updated'] += 1

        # Extract attachments
        if extract_files:
            attachments = extract_attachments_from_message(msg)
            for attach in attachments:
                asset_pointer = attach['asset_pointer']
                filename = asset_pointer_to_filename(asset_pointer, zip_contents)

                if not filename:
                    continue

                # Extract file
                file_path, file_hash = extract_attachment_file(
                    z, filename, conv_id, attachments_path
                )

                if not file_path:
                    continue

                # Insert attachment record
                cur.execute("""
                    INSERT INTO conversation_attachments
                    (source, conversation_id, message_id, asset_pointer, filename, original_filename,
                     content_type, size_bytes, width, height, file_path, file_hash, metadata, created_at)
                    VALUES ('chatgpt', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source, conversation_id, filename) DO UPDATE SET
                        file_path = EXCLUDED.file_path,
                        file_hash = EXCLUDED.file_hash
                """, (
                    conv_id,
                    node_id,
                    asset_pointer,
                    Path(filename).name,
                    filename,
                    guess_content_type(filename),
                    attach.get('size_bytes'),
                    attach.get('width'),
                    attach.get('height'),
                    file_path,
                    file_hash,
                    Json(attach.get('metadata', {})),
                    msg_created,
                ))
                stats['attachments'] += 1

    return stats


def process_zip(
    conn,
    zip_path: Path,
    attachments_path: Path,
    extract_files: bool = True,
    force_attachments: bool = False,
    batch_size: int = 50
) -> Dict[str, int]:
    """Process a single ZIP file."""
    stats = {
        'conversations': 0,
        'conv_inserted': 0,
        'conv_updated': 0,
        'conv_skipped': 0,
        'msgs_inserted': 0,
        'msgs_updated': 0,
        'attachments': 0,
        'errors': 0,
    }

    try:
        z = zipfile.ZipFile(zip_path)
        convos_json = z.read('conversations.json')
        data = json.loads(convos_json)
        zip_contents = z.namelist()
    except Exception as e:
        print(f"  Error reading {zip_path.name}: {e}")
        stats['errors'] += 1
        return stats

    total_convos = len(data)

    for i, conv_data in enumerate(data):
        try:
            with conn.cursor() as cur:
                conv = parse_conversation(conv_data)
                conv_stats = ingest_conversation(
                    cur, conv, zip_path.name, z, zip_contents,
                    attachments_path, extract_files, force_attachments
                )

                stats['conversations'] += 1
                for key in ['conv_inserted', 'conv_updated', 'conv_skipped',
                           'msgs_inserted', 'msgs_updated', 'attachments']:
                    stats[key] += conv_stats.get(key, 0)

            conn.commit()

            if (i + 1) % batch_size == 0:
                print(f"    Progress: {i + 1}/{total_convos}")

        except Exception as e:
            stats['errors'] += 1
            title = conv_data.get('title', 'unknown')[:30]
            print(f"  Error processing '{title}': {e}")
            try:
                conn.rollback()
            except Exception:
                pass

    z.close()
    return stats


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest ChatGPT exports into LifeDB")
    parser.add_argument("--limit", type=int, help="Limit number of ZIP files")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no changes")
    parser.add_argument("--no-attachments", action="store_true", help="Skip attachment extraction")
    parser.add_argument("--reprocess-attachments", action="store_true",
                       help="Extract attachments even for existing conversations")
    parser.add_argument("--source", choices=['inbox', 'dropbox', 'both'], default='both',
                       help="Which source to process")
    parser.add_argument("--file", type=str, help="Process specific ZIP file")
    args = parser.parse_args()

    print(f"ChatGPT ETL - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Collect ZIP files from sources
    zip_files = []

    if args.file:
        zip_files = [Path(args.file)]
    else:
        if args.source in ('inbox', 'both') and INBOX_PATH.exists():
            zip_files.extend(sorted(INBOX_PATH.glob("*.zip")))

        if args.source in ('dropbox', 'both') and DROPBOX_SOURCE.exists():
            # Only add gdrive files not already processed
            for zf in sorted(DROPBOX_SOURCE.glob("*.zip")):
                if zf not in zip_files:
                    zip_files.append(zf)

    if args.limit:
        zip_files = zip_files[:args.limit]

    print(f"Found {len(zip_files)} ZIP file(s)")

    if args.dry_run:
        print("\nDRY RUN - counting contents only")
        for zf in zip_files:
            try:
                with zipfile.ZipFile(zf) as z:
                    convos = json.loads(z.read('conversations.json'))
                    attachments = [n for n in z.namelist()
                                 if any(n.endswith(ext) for ext in
                                       ['.png', '.jpg', '.jpeg', '.webp', '.pdf', '.csv'])]
                    print(f"  {zf.name}:")
                    print(f"    {len(convos)} conversations, {len(attachments)} attachments")
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

    # Ensure attachments directory exists
    ATTACHMENTS_PATH.mkdir(parents=True, exist_ok=True)

    try:
        total = {
            'conversations': 0,
            'conv_inserted': 0,
            'conv_updated': 0,
            'conv_skipped': 0,
            'msgs_inserted': 0,
            'msgs_updated': 0,
            'attachments': 0,
            'errors': 0,
        }

        inbox_processed = []

        for zf in zip_files:
            print(f"\nProcessing {zf.name}...")
            stats = process_zip(
                conn, zf, ATTACHMENTS_PATH,
                extract_files=not args.no_attachments,
                force_attachments=args.reprocess_attachments
            )

            for key in total:
                total[key] += stats[key]

            print(f"  {stats['conversations']} convos: "
                  f"{stats['conv_inserted']} new, {stats['conv_updated']} updated, "
                  f"{stats['conv_skipped']} skipped")
            print(f"  msgs: {stats['msgs_inserted']} new, {stats['msgs_updated']} updated | "
                  f"attachments: {stats['attachments']} | errors: {stats['errors']}")

            # Track inbox files for moving
            if stats['errors'] == 0 and zf.parent == INBOX_PATH:
                inbox_processed.append(zf)

        print(f"\n{'='*60}")
        print(f"Total: {total['conversations']} conversations")
        print(f"  Conversations: {total['conv_inserted']} new, "
              f"{total['conv_updated']} updated, {total['conv_skipped']} skipped")
        print(f"  Messages: {total['msgs_inserted']} new, {total['msgs_updated']} updated")
        print(f"  Attachments: {total['attachments']}")
        print(f"  Errors: {total['errors']}")

        # Move processed inbox files
        if inbox_processed:
            PROCESSED_PATH.mkdir(exist_ok=True)
            print(f"\nMoving {len(inbox_processed)} inbox file(s) to processed/")
            for zf in inbox_processed:
                dest = PROCESSED_PATH / zf.name
                zf.rename(dest)
                print(f"  {zf.name} → processed/")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
