#!/usr/bin/env python3
"""
Migrate v1-only conversations from conversationsdb (v1) into conversationsdb_v2.

DESIGN RULES
  - INSERT ONLY. Never DELETE, never UPDATE an existing row.
  - Idempotent: safe to re-run. Conversations are matched on the v2 unique
    key (session_id, machine, user_name); messages are anti-joined on uuid.
  - Isaac's ruling 2026-07-30: "we don't want reasoning or tool blocks indexed".
    Only v1 message_type='message' is migrated. tool_use / tool_result rows are
    deliberately dropped. thinking/tool_calls/tool_results are left NULL.
  - content_hash = sha256(RAW content) when content is non-empty, else NULL.
    This matches v2/db.py exactly and is what embeddings key on.

FIELD MAPPING (v1 -> v2)
  conversations.conversation_id -> conversations.session_id
  conversations.device_id       -> conversations.machine
  conversations.project_path    -> conversations.project
  conversations.source_id       -> conversations.source
  conversations.model           -> conversations.model
  conversations.created_at      -> conversations.started_at
  max(messages.created_at)      -> conversations.last_message_at
  (tenant 'isaac')              -> conversations.user_name = 'isaac'
  (synthesised)                 -> conversations.source_file = MARKER

  messages.message_id           -> messages.uuid
  messages.parent_message_id    -> messages.parent_uuid
  messages.sender               -> messages.role   (human->user, else identity)
  messages.content              -> messages.content
  messages.created_at           -> messages.timestamp
  messages.ordinal              -> messages.sequence_num
"""

import hashlib
import os
import sys
import psycopg2
from psycopg2.extras import execute_values

MARKER = "legacy-import-v1mig-20260730"
REMNANT_MARKER = "legacy-import-v1mig-20260730-remnant"
USER_NAME = "isaac"          # v1 tenant_id is 'isaac' for every row
ROLE_MAP = {"human": "user", "assistant": "assistant", "system": "system"}

V1 = dict(host="100.127.104.75", dbname="conversationsdb",
          user=os.environ.get("LIFEDB_USER", "claude"),
          password=os.environ["LIFEDB_PASSWORD"])
V2 = dict(host="100.127.104.75", dbname="conversationsdb_v2",
          user=os.environ["CONVERSATIONS_DB_USER"],
          password=os.environ["CONVERSATIONS_DB_PASSWORD"])


def chash(content):
    return hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None


def fetch_v1_conv(c1, conv_id):
    with c1.cursor() as cur:
        cur.execute("""SELECT conversation_id, source_id, device_id, project_path,
                              model, created_at, updated_at
                       FROM conversations WHERE conversation_id=%s""", (conv_id,))
        return cur.fetchone()


def fetch_v1_messages(c1, conv_id, only_ids=None):
    """message_type='message' only — tool blocks excluded per Isaac's ruling."""
    with c1.cursor() as cur:
        if only_ids:
            cur.execute("""SELECT message_id, parent_message_id, sender, content,
                                  created_at, ordinal
                           FROM messages
                           WHERE conversation_id=%s AND message_type='message'
                             AND message_id = ANY(%s)
                           ORDER BY ordinal, created_at""", (conv_id, list(only_ids)))
        else:
            cur.execute("""SELECT message_id, parent_message_id, sender, content,
                                  created_at, ordinal
                           FROM messages
                           WHERE conversation_id=%s AND message_type='message'
                           ORDER BY ordinal, created_at""", (conv_id,))
        return cur.fetchall()


def get_or_create_conv(c2, session_id, machine, project, model, source,
                       started_at, last_message_at, msg_count, source_file):
    """Returns (conv_uuid, created_bool). Never updates an existing row."""
    with c2.cursor() as cur:
        cur.execute("""SELECT id, source_file FROM conversations
                       WHERE session_id=%s AND machine=%s AND user_name=%s""",
                    (session_id, machine, USER_NAME))
        row = cur.fetchone()
        if row:
            return row[0], False
        cur.execute("""INSERT INTO conversations
              (session_id, user_name, machine, project, model, source_file,
               started_at, last_message_at, message_count, source,
               is_subagent, parent_session_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FALSE,NULL)
            RETURNING id""",
                    (session_id, USER_NAME, machine, project, model, source_file,
                     started_at, last_message_at, msg_count, source))
        return cur.fetchone()[0], True


def insert_messages(c2, conv_uuid, rows):
    """Insert only messages whose uuid is not already in this conversation."""
    with c2.cursor() as cur:
        cur.execute("SELECT uuid FROM messages WHERE conversation_id=%s AND uuid IS NOT NULL",
                    (conv_uuid,))
        have = {r[0] for r in cur.fetchall()}

        payload = []
        for mid, pmid, sender, content, created_at, ordinal in rows:
            if mid in have:
                continue
            role = ROLE_MAP.get(sender)
            if role is None:
                raise ValueError(f"unmapped sender {sender!r} on message {mid}")
            payload.append((conv_uuid, mid, pmid, role, content, None, None, None,
                            False, created_at, ordinal, chash(content)))

        if payload:
            execute_values(cur, """INSERT INTO messages
                (conversation_id, uuid, parent_uuid, role, content, thinking,
                 tool_calls, tool_results, is_sidechain, timestamp,
                 sequence_num, content_hash)
                VALUES %s""", payload, page_size=500)
        return len(payload)


def main():
    whole = [l.strip() for l in open("/tmp/v1mig/v1_only_convids.txt") if l.strip()]

    # the single partially-absent shared conversation + the uuids v2 lacks
    remnants = {}
    for line in open("/tmp/v1mig/v1_only_msgs.tsv"):
        cid, mid = line.rstrip("\n").split("\t")
        if cid not in set(whole):
            remnants.setdefault(cid, []).append(mid)

    c1 = psycopg2.connect(**V1)
    c2 = psycopg2.connect(**V2)
    stats = {"conv": 0, "msg": 0, "skipped": 0}
    failed = []

    def process(cid, only_ids=None, remnant=False):
        """One conversation, isolated. Rolls back on error so a bad row
        cannot poison the rows after it (the ingest-cascade lesson)."""
        v1c = fetch_v1_conv(c1, cid)
        if not v1c:
            print(f"SKIP {cid}: absent from v1", file=sys.stderr)
            stats["skipped"] += 1
            return
        _, source_id, device_id, project_path, model, created_at, updated_at = v1c
        msgs = fetch_v1_messages(c1, cid, only_ids=only_ids)
        if not msgs:
            print(f"SKIP {cid}: no message-type rows", file=sys.stderr)
            stats["skipped"] += 1
            return

        stamps = [m[4] for m in msgs if m[4]]
        last_at = max(stamps) if stamps else updated_at
        first_at = (min(stamps) if stamps else created_at) if remnant else created_at
        sid = f"{cid}-v1remnant" if remnant else cid
        marker = REMNANT_MARKER if remnant else MARKER

        conv_uuid, created = get_or_create_conv(
            c2, sid, device_id, project_path, model, source_id,
            first_at, last_at, len(msgs), marker)
        n = insert_messages(c2, conv_uuid, msgs)
        c2.commit()
        stats["conv"] += 1 if created else 0
        stats["msg"] += n
        tag = "REMNANT " if remnant else ""
        print(f"{'NEW ' if created else 'HAVE'} {tag}{cid} machine={device_id} +{n} msgs")

    try:
        for cid in whole:
            try:
                process(cid)
            except Exception as e:
                c2.rollback()
                failed.append((cid, str(e)))
                print(f"FAIL {cid}: {e}", file=sys.stderr)

        # Partial conversations get their OWN conversation row, never an
        # injection into the live one: v1 ordinals collide with the existing
        # row's sequence_num space, and mixing two provenances in one row is
        # exactly the D1 mistake.
        for cid, mids in remnants.items():
            try:
                process(cid, only_ids=mids, remnant=True)
            except Exception as e:
                c2.rollback()
                failed.append((cid, str(e)))
                print(f"FAIL remnant {cid}: {e}", file=sys.stderr)
    finally:
        c1.close(); c2.close()

    print(f"\nconversations created: {stats['conv']}")
    print(f"messages inserted: {stats['msg']}")
    print(f"skipped: {stats['skipped']}")
    print(f"failed: {len(failed)}")
    for cid, err in failed:
        print(f"  FAILED {cid}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
