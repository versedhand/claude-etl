#!/usr/bin/env python3
"""
Claude Conversation Sync & Ingest

Single script run by cron every minute:
1. rsync ~/.claude/projects/ → /var/lib/versedhand/claude-mirror/
2. Scan mirror for changed files (mtime comparison)
3. Parse changed files and upsert into conversationsdb

Usage:
    claude-conversation-sync              # Normal run
    claude-conversation-sync --dry-run    # Show what would be processed
    claude-conversation-sync --status     # Show stats
    claude-conversation-sync --init-db    # Create schema only
"""

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .parser import parse_conversation
from .db import get_connection, ensure_schema, upsert_conversation
from .subagent import is_subagent_path, subagent_identity

# Configuration
# Subagent transcripts are safe to ingest (they get their own identity — see
# v2/subagent.py), but there are ~2,100 of them and backfilling is a bulk load
# with embedding cost. Ingest of NEW subagent files is opt-in until that
# backfill is scheduled deliberately: CLAUDE_ETL_INGEST_SUBAGENTS=1
INGEST_SUBAGENTS = os.environ.get("CLAUDE_ETL_INGEST_SUBAGENTS", "0") == "1"

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
MIRROR_DIR = Path("/var/lib/versedhand/claude-mirror")
STATE_FILE = Path("/var/lib/versedhand/ingest-state.json")
LOG_FILE = Path("/var/log/versedhand/claude-etl.log")

logger = logging.getLogger("claude-etl")


def setup_logging():
    """Configure logging to file and stderr."""
    log_dir = LOG_FILE.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # The modules this script drives log to their own "v2.*" loggers, which had
    # no handler and no level — so v2.db's per-conversation line had NEVER once
    # reached this file (measured 2026-08-28: 0 occurrences of "Upserted
    # conversation" in the whole log). The sync's own "N ingested" line is a
    # COUNT OF FILES and says nothing about what was written per conversation,
    # so the one place the append-vs-replace path is visible was being discarded.
    # Same handler, same level, so the two interleave in one timeline.
    v2_logger = logging.getLogger("v2")
    for h in logger.handlers:
        v2_logger.addHandler(h)
    v2_logger.setLevel(logging.INFO)


def sync_mirror() -> bool:
    """rsync live .claude/projects/ to mirror directory."""
    if not CLAUDE_PROJECTS.exists():
        logger.warning(f"Source directory does not exist: {CLAUDE_PROJECTS}")
        return False

    MIRROR_DIR.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "rsync", "-a", "--delete",
            "--exclude=*.lock", "--exclude=*.tmp",
            str(CLAUDE_PROJECTS) + "/",
            str(MIRROR_DIR) + "/",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"rsync failed: {result.stderr}")
        return False

    return True


def load_state() -> dict:
    """Load the mtime state file."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("State file corrupt, starting fresh")
    return {"files": {}}


def save_state(state: dict):
    """Write the mtime state file atomically."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(STATE_FILE)


def find_changed_files(state: dict) -> list[Path]:
    """Scan mirror directory and return files that changed since last run."""
    changed = []
    current_files = {}

    for jsonl in MIRROR_DIR.rglob("*.jsonl"):
        rel_path = str(jsonl.relative_to(MIRROR_DIR))
        # Subagent transcripts are NOT skipped any more. They used to be excluded
        # because they inherit the parent's sessionId and so overwrote the parent's
        # conversation row (D1). They now get their own path-derived identity —
        # see v2/subagent.py — so they are ingested and searchable.
        if is_subagent_path(rel_path) and not INGEST_SUBAGENTS:
            continue
        stat = jsonl.stat()
        mtime = stat.st_mtime
        size = stat.st_size

        current_files[rel_path] = {"mtime": mtime, "size": size}

        prev = state.get("files", {}).get(rel_path)
        if not prev or prev.get("mtime") != mtime or prev.get("size") != size:
            changed.append(jsonl)

    return changed


def run_ingest(dry_run: bool = False):
    """Main sync + ingest pipeline."""
    start_time = time.time()
    user_name = os.environ.get("CLAUDE_ETL_USER", os.getlogin())
    machine = os.environ.get("CLAUDE_ETL_MACHINE", socket.gethostname())

    # Step 1: Sync
    logger.info("Starting sync...")
    if not sync_mirror():
        logger.error("Sync failed, aborting")
        return

    # Step 2: Find changed files
    state = load_state()
    changed = find_changed_files(state)

    if not changed:
        duration = time.time() - start_time
        logger.info(f"No changes detected ({duration:.2f}s)")
        return

    logger.info(f"Found {len(changed)} changed file(s)")

    if dry_run:
        for f in changed:
            print(f"  Would ingest: {f}")
        return

    # Step 3: Connect to DB (skip schema creation on routine runs — tables already exist)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM conversations LIMIT 1")
    except Exception:
        conn.rollback()
        ensure_schema(conn)

    # Step 4: Parse and ingest each changed file
    ingested = 0
    errors = 0

    for jsonl_path in changed:
        rel_path = str(jsonl_path.relative_to(MIRROR_DIR))
        try:
            # Compute file content hash BEFORE parsing — skip if DB already has this exact content
            import hashlib
            file_hash = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()

            parsed = parse_conversation(jsonl_path)

            # A subagent transcript carries the PARENT's sessionId in every record.
            # Replace it with a path-derived identity so it can never resolve to —
            # and therefore never destroy — the parent's conversation row.
            is_sub = is_subagent_path(rel_path)
            parent_sid = None
            if is_sub:
                parsed.session_id, parent_sid = subagent_identity(rel_path)

            if not parsed.messages:
                logger.info(f"Skipping empty: {rel_path}")
                state.setdefault("files", {})[rel_path] = {
                    "mtime": jsonl_path.stat().st_mtime,
                    "size": jsonl_path.stat().st_size,
                    "last_ingested": datetime.now(timezone.utc).isoformat(),
                }
                continue

            # Check if this exact file content was already ingested (dedup across paths)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM conversations WHERE session_id = %s AND machine = %s AND user_name = %s AND file_hash = %s",
                    (parsed.session_id, machine, user_name, file_hash),
                )
                if cur.fetchone():
                    # Identical content already in DB — skip
                    state.setdefault("files", {})[rel_path] = {
                        "mtime": jsonl_path.stat().st_mtime,
                        "size": jsonl_path.stat().st_size,
                        "last_ingested": datetime.now(timezone.utc).isoformat(),
                    }
                    continue

            file_mtime = datetime.fromtimestamp(
                jsonl_path.stat().st_mtime, tz=timezone.utc
            )

            upsert_conversation(
                conn, parsed, user_name, machine, rel_path, file_mtime,
                file_hash=file_hash,
                is_subagent=is_sub, parent_session_id=parent_sid,
            )

            # Update state for this file
            state.setdefault("files", {})[rel_path] = {
                "mtime": jsonl_path.stat().st_mtime,
                "size": jsonl_path.stat().st_size,
                "last_ingested": datetime.now(timezone.utc).isoformat(),
            }
            ingested += 1

        except Exception as e:
            # Clear the aborted transaction so ONE bad file can't poison every file after it
            # (the "current transaction is aborted" cascade that silently stalled ingest).
            try:
                conn.rollback()
            except Exception:
                pass
            logger.error(f"Failed to ingest {rel_path}: {e}")
            errors += 1
            # Do NOT update state for failed files — they'll retry next run

    conn.close()
    save_state(state)

    duration = time.time() - start_time
    logger.info(
        f"Done: {ingested} ingested, {errors} errors, "
        f"{len(changed)} changed ({duration:.2f}s)"
    )


def show_status():
    """Show current ingestion stats."""
    state = load_state()
    n_files = len(state.get("files", {}))

    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM conversations")
            n_convos = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM messages")
            n_msgs = cur.fetchone()[0]
            cur.execute(
                "SELECT user_name, COUNT(*) FROM conversations GROUP BY user_name"
            )
            by_user = dict(cur.fetchall())
        conn.close()
    except Exception as e:
        print(f"DB connection failed: {e}")
        n_convos = n_msgs = 0
        by_user = {}

    print(f"State file: {n_files} files tracked")
    print(f"DB: {n_convos} conversations, {n_msgs} messages")
    if by_user:
        print("By user:")
        for user, count in sorted(by_user.items()):
            print(f"  {user}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Claude Conversation Sync & Ingest")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--status", action="store_true", help="Show stats")
    parser.add_argument("--init-db", action="store_true", help="Create schema only")
    args = parser.parse_args()

    setup_logging()

    if args.status:
        show_status()
    elif args.init_db:
        conn = get_connection()
        ensure_schema(conn)
        conn.close()
        print("Schema created")
    else:
        run_ingest(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
