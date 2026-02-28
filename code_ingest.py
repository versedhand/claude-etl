#!/usr/bin/env python3
"""
Claude Code conversation ingestion script.

Scans synced .claude/projects/ directories and loads conversations into LifeDB.
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Iterator
import psycopg2
from psycopg2.extras import Json

# Namespace for generating deterministic UUIDs from non-UUID conversation IDs
CLAUDE_CODE_NAMESPACE = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')


def normalize_conversation_id(conv_id: str) -> str:
    """Convert conversation ID to valid UUID format.

    If already a valid UUID, returns as-is.
    Otherwise, generates a deterministic UUID5 from the string.
    """
    try:
        # Check if already valid UUID
        uuid.UUID(conv_id)
        return conv_id
    except ValueError:
        # Generate deterministic UUID from the string (e.g., "agent-a04b965")
        return str(uuid.uuid5(CLAUDE_CODE_NAMESPACE, conv_id))

# Configuration - check multiple base paths
HOMES_PATHS = [
    Path("/mnt/d/obs/life-var/homes"),  # Desktop (WSL)
    Path("/srv/obs/life-var/homes"),    # Server
]
HOMES_BASE = next((p for p in HOMES_PATHS if p.exists()), HOMES_PATHS[0])
DEVICES = ["blue", "black", "red", "recovered-nov", "magenta"]
DB_CONFIG = {
    "host": "100.86.218.108",
    "port": 5432,
    "dbname": "personal_data",
    "user": "postgres",
    "password": os.environ.get("LIFEDB_PASSWORD", "StrongPassword123"),
}


def sanitize_null_bytes(obj):
    """Recursively remove null bytes from strings in a JSON object.

    PostgreSQL text/JSONB doesn't allow \u0000 (null byte).
    """
    if isinstance(obj, str):
        return obj.replace('\x00', '')
    elif isinstance(obj, dict):
        return {k: sanitize_null_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_null_bytes(item) for item in obj]
    return obj


def parse_jsonl_file(filepath: Path) -> Iterator[dict]:
    """Parse a JSONL file, yielding each line as a dict."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Sanitize null bytes that PostgreSQL can't handle
                record = sanitize_null_bytes(record)
                yield line_num, record
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping line {line_num} in {filepath.name}: {e}")
                continue


def extract_fields(record: dict) -> dict:
    """Extract indexed fields from a record."""
    return {
        "record_type": record.get("type"),
        "record_uuid": record.get("uuid"),
        "parent_uuid": record.get("parentUuid"),
        "session_id": record.get("sessionId"),
        "timestamp": parse_timestamp(record.get("timestamp")),
    }


def parse_timestamp(ts) -> Optional[datetime]:
    """Parse various timestamp formats."""
    if not ts:
        return None

    if isinstance(ts, (int, float)):
        # Unix timestamp in milliseconds
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

    if isinstance(ts, str):
        try:
            # ISO format
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except ValueError:
            return None

    return None


def get_file_mtime(filepath: Path) -> datetime:
    """Get file modification time as datetime."""
    return datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)


def extract_project_path(folder_name: str) -> str:
    """Convert folder name back to path (e.g., '-mnt-d-obs-life' -> '/mnt/d/obs/life')."""
    if folder_name.startswith('-'):
        return folder_name.replace('-', '/', 1).replace('-', '/')
    return folder_name


def ingest_conversation(conn, device: str, project_path: str,
                        conversation_id: str, filepath: Path,
                        file_mtime: datetime) -> tuple[int, int]:
    """Ingest a single conversation file. Returns (inserted, updated) counts."""
    inserted = 0
    updated = 0

    with conn.cursor() as cur:
        for line_num, record in parse_jsonl_file(filepath):
            fields = extract_fields(record)

            cur.execute("""
                INSERT INTO claude_code_raw
                (device, project_path, conversation_id, line_number,
                 record_type, record_uuid, parent_uuid, session_id, timestamp,
                 data, file_mtime)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (conversation_id, line_number)
                DO UPDATE SET
                    record_type = EXCLUDED.record_type,
                    record_uuid = EXCLUDED.record_uuid,
                    parent_uuid = EXCLUDED.parent_uuid,
                    session_id = EXCLUDED.session_id,
                    timestamp = EXCLUDED.timestamp,
                    data = EXCLUDED.data,
                    file_mtime = EXCLUDED.file_mtime,
                    ingested_at = NOW()
                WHERE claude_code_raw.file_mtime < EXCLUDED.file_mtime
                RETURNING (xmax = 0) AS inserted
            """, (
                device,
                project_path,
                conversation_id,
                line_num,
                fields["record_type"],
                fields["record_uuid"],
                fields["parent_uuid"],
                fields["session_id"],
                fields["timestamp"],
                Json(record),
                file_mtime,
            ))

            result = cur.fetchone()
            if result:
                if result[0]:
                    inserted += 1
                else:
                    updated += 1

    return inserted, updated


def scan_device(conn, device: str, limit: Optional[int] = None) -> dict:
    """Scan a device's .claude/projects/ directory."""
    stats = {"conversations": 0, "inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

    device_path = HOMES_BASE / device / ".claude" / "projects"
    if not device_path.exists():
        print(f"  Device path not found: {device_path}")
        return stats

    # Iterate through project folders
    for project_folder in device_path.iterdir():
        if not project_folder.is_dir():
            continue

        project_path = extract_project_path(project_folder.name)

        # Iterate through conversation files
        for conv_file in project_folder.glob("*.jsonl"):
            if limit and stats["conversations"] >= limit:
                return stats

            conversation_id = normalize_conversation_id(conv_file.stem)
            file_mtime = get_file_mtime(conv_file)

            try:
                inserted, updated = ingest_conversation(
                    conn, device, project_path, conversation_id, conv_file, file_mtime
                )

                if inserted > 0 or updated > 0:
                    stats["conversations"] += 1
                    stats["inserted"] += inserted
                    stats["updated"] += updated
                    if inserted > 0:
                        print(f"  + {conv_file.name}: {inserted} records")
                else:
                    stats["skipped"] += 1

                conn.commit()

            except Exception as e:
                stats["errors"] += 1
                print(f"  Error processing {conv_file.name}: {e}")
                conn.rollback()

    return stats


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Claude Code conversations into LifeDB")
    parser.add_argument("--device", choices=DEVICES, help="Only process specific device")
    parser.add_argument("--limit", type=int, help="Limit number of conversations per device")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    args = parser.parse_args()

    devices = [args.device] if args.device else DEVICES

    print(f"Claude Code ETL - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Devices: {', '.join(devices)}")
    print()

    if args.dry_run:
        print("DRY RUN - counting files only")
        for device in devices:
            device_path = HOMES_BASE / device / ".claude" / "projects"
            if device_path.exists():
                count = sum(1 for _ in device_path.rglob("*.jsonl"))
                print(f"  {device}: {count} conversation files")
        return

    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected to LifeDB")
    except Exception as e:
        print(f"Failed to connect to LifeDB: {e}")
        print("Set LIFEDB_PASSWORD environment variable if needed")
        sys.exit(1)

    total_stats = {"conversations": 0, "inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

    try:
        for device in devices:
            print(f"\nProcessing {device}...")
            stats = scan_device(conn, device, limit=args.limit)

            for key in total_stats:
                total_stats[key] += stats[key]

            print(f"  {device}: {stats['conversations']} conversations, "
                  f"{stats['inserted']} inserted, {stats['updated']} updated, "
                  f"{stats['skipped']} skipped, {stats['errors']} errors")

    finally:
        conn.close()

    print(f"\nTotal: {total_stats['conversations']} conversations, "
          f"{total_stats['inserted']} records inserted, "
          f"{total_stats['updated']} updated, "
          f"{total_stats['skipped']} skipped, "
          f"{total_stats['errors']} errors")


if __name__ == "__main__":
    main()
