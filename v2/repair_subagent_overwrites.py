#!/usr/bin/env python3
"""Repair conversations whose messages were overwritten by a subagent transcript (D1).

Background
----------
Subagent transcripts carry the PARENT session's `sessionId` in every record.
Before the identity fix (v2/subagent.py) ingesting one resolved to the parent's
conversation row and ran `DELETE FROM messages WHERE conversation_id = ...`,
replacing a real conversation's messages with the subagent's.

This script finds conversations whose `source_file` is a subagent path but whose
`session_id` names a parent transcript that still exists on disk, and re-ingests
the real parent file.

Method
------
The poisoned conversation row is DELETED (messages cascade) and the parent file
is ingested fresh. That deliberately takes the INSERT branch of
upsert_conversation, so the new source-collision guard is never asked to make an
exception for us. Embeddings are content-addressed (embeddings.content_hash, no
FK to messages) so nothing embedded is lost by this.

Re-ingest is done by calling the parser + upsert DIRECTLY rather than by waiting
for the cron: the ingest state file already holds the parent's path with a
matching mtime/size, so change-detection would skip it and the repair would
silently do nothing.

Verification is by a DIFFERENT path than the ingest: the expected message count
comes from `jq` (a separate C JSON parser) counting top-level records of type
user/assistant/system, not from parser.py.

Usage:
    python3 -m v2.repair_subagent_overwrites --dry-run
    python3 -m v2.repair_subagent_overwrites --apply
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .db import get_connection, upsert_conversation
from .parser import parse_conversation
from .subagent import is_subagent_path

MIRROR_DIR = Path("/var/lib/versedhand/claude-mirror")
LIVE_DIR = Path.home() / ".claude" / "projects"


def expected_message_count(path: Path) -> int | None:
    """Count ingestable records using jq — independent of parser.py.

    parse_conversation() keeps records whose top-level "type" is user,
    assistant or system. jq is a separate parser implementation, so agreement
    between the two is real evidence rather than the same code agreeing itself.
    """
    try:
        out = subprocess.run(
            ["jq", "-r", 'select(.type=="user" or .type=="assistant" or .type=="system") | .type'],
            stdin=path.open("rb"), capture_output=True, text=True, timeout=300,
        )
        if out.returncode != 0:
            return None
        return sum(1 for line in out.stdout.splitlines() if line.strip())
    except Exception:
        return None


def find_damaged(conn):
    """Conversations sourced from a subagent file whose parent transcript exists."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, session_id, machine, user_name, source_file, message_count
               FROM conversations
               WHERE source_file LIKE %s
               ORDER BY session_id""",
            ("%subagents/%",),
        )
        rows = cur.fetchall()

    damaged, orphaned = [], []
    for conv_id, sid, machine, user, src, mc in rows:
        if not is_subagent_path(src):
            continue
        project = src.split("/")[0]
        rel = f"{project}/{sid}.jsonl"
        parent = MIRROR_DIR / rel
        if not parent.exists():
            parent = LIVE_DIR / rel
        rec = {
            "conv_id": conv_id, "session_id": sid, "machine": machine,
            "user_name": user, "bad_source": src, "before_count": mc,
            "parent_rel": rel, "parent_path": parent,
        }
        (damaged if parent.exists() else orphaned).append(rec)
    return damaged, orphaned


def repair_one(conn, rec, apply: bool) -> dict:
    path = rec["parent_path"]
    expected = expected_message_count(path)
    rec["expected"] = expected

    if not apply:
        rec["after_count"] = None
        rec["status"] = "DRY-RUN"
        return rec

    with conn.cursor() as cur:
        # Remove the poisoned row entirely; messages cascade.
        cur.execute("DELETE FROM conversations WHERE id = %s", (rec["conv_id"],))
    conn.commit()

    parsed = parse_conversation(path)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    file_mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    if parsed.session_id != rec["session_id"]:
        rec["status"] = f"ABORT: parsed session_id {parsed.session_id!r} != expected"
        rec["after_count"] = 0
        return rec

    upsert_conversation(
        conn, parsed, rec["user_name"], rec["machine"], rec["parent_rel"],
        file_mtime, file_hash=file_hash, is_subagent=False, parent_session_id=None,
    )

    # Read the result back from the DB — not from the object we just wrote.
    with conn.cursor() as cur:
        cur.execute(
            """SELECT count(*) FROM messages m JOIN conversations c ON m.conversation_id = c.id
               WHERE c.session_id=%s AND c.machine=%s AND c.user_name=%s""",
            (rec["session_id"], rec["machine"], rec["user_name"]),
        )
        rec["after_count"] = cur.fetchone()[0]

    if expected is None:
        rec["status"] = "REPAIRED (jq unavailable — unverified)"
    elif rec["after_count"] == expected:
        rec["status"] = "REPAIRED+VERIFIED"
    else:
        rec["status"] = f"MISMATCH (db={rec['after_count']} jq={expected})"
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    conn = get_connection()
    damaged, orphaned = find_damaged(conn)

    print(f"Subagent-sourced conversations with a recoverable parent : {len(damaged)}")
    print(f"Subagent-sourced conversations with NO parent on disk     : {len(orphaned)}")
    print()

    results = [repair_one(conn, r, args.apply) for r in damaged]

    hdr = f"{'session_id':38} {'before':>7} {'after':>7} {'jq':>7}  status"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['session_id']:38} {r['before_count']:>7} "
              f"{str(r['after_count']) if r['after_count'] is not None else '-':>7} "
              f"{str(r['expected']) if r['expected'] is not None else '?':>7}  {r['status']}")

    if args.apply:
        ok = sum(1 for r in results if r["status"] == "REPAIRED+VERIFIED")
        print(f"\nverified: {ok}/{len(results)}")
        gained = sum((r["after_count"] or 0) - r["before_count"] for r in results)
        print(f"net messages restored: {gained:+,}")
        if ok != len(results):
            conn.close()
            sys.exit(1)

    conn.close()


if __name__ == "__main__":
    main()
