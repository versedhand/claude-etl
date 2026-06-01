"""Migrate Claude Web conversations from old DB raw tables to conversationsdb_v2."""
import os
import psycopg2
from psycopg2.extras import execute_values
import time
import hashlib

OLD_DB = {"host": "100.127.104.75", "port": 5432, "dbname": "conversationsdb",
          "user": "postgres", "password": os.environ["POSTGRES_PASSWORD"]}
NEW_DB = {"host": "100.127.104.75", "port": 5432, "dbname": "conversationsdb_v2",
          "user": "conversations_writer", "password": os.environ["CONVERSATIONS_DB_PASSWORD"]}

old = psycopg2.connect(**OLD_DB)
new = psycopg2.connect(**NEW_DB)
start = time.time()

# --- Phase 1: Insert missing conversations from claude_web_raw ---
print("Phase 1: Inserting missing web conversations...")
with old.cursor() as cur:
    cur.execute("""
        SELECT conversation_id, account_email, name, title, created_at, updated_at, data->>'model' as model
        FROM claude_web_raw
    """)
    raw_convos = cur.fetchall()

print(f"Found {len(raw_convos)} raw web conversations")

with new.cursor() as cur:
    cur.execute("SELECT session_id FROM conversations WHERE source = 'web'")
    existing = {r[0] for r in cur.fetchall()}

print(f"Already have {len(existing)} web conversations in v2")

inserted = 0
for conv_id, email, name, title, created_at, updated_at, model in raw_convos:
    if conv_id in existing:
        continue
    with new.cursor() as cur:
        cur.execute("""
            INSERT INTO conversations (session_id, user_name, machine, project, model, source_file, 
                                       started_at, last_message_at, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id, machine, user_name) DO NOTHING
        """, (conv_id, "isaac", "web", name or title, model, "legacy-web-import",
              created_at, updated_at, "web"))
    inserted += 1

new.commit()
print(f"Inserted {inserted} new web conversations")

# --- Phase 2: Backfill messages from claude_web_messages ---
print("\nPhase 2: Backfilling messages from claude_web_messages...")

# Get all web conversations that need messages
with new.cursor() as cur:
    cur.execute("""
        SELECT c.id, c.session_id FROM conversations c
        LEFT JOIN (SELECT conversation_id, count(*) as cnt FROM messages GROUP BY conversation_id) m
        ON c.id = m.conversation_id
        WHERE c.source = 'web' AND (m.cnt IS NULL OR m.cnt = 0)
    """)
    empty_convos = cur.fetchall()

print(f"Found {len(empty_convos)} web conversations with 0 messages")

msg_count = 0
for v2_id, session_id in empty_convos:
    with old.cursor() as ocur:
        ocur.execute("""
            SELECT uuid, parent_message_uuid, sender, content, created_at
            FROM claude_web_messages
            WHERE conversation_uuid = %s
            ORDER BY created_at
        """, (session_id,))
        msgs = ocur.fetchall()
    
    if not msgs:
        continue
    
    rows = []
    for i, (uuid, parent_uuid, sender, content, ts) in enumerate(msgs):
        role = "assistant" if sender == "assistant" else "user" if sender == "human" else sender or "unknown"
        ch = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None
        rows.append((v2_id, uuid, parent_uuid, role, content, None, None, None, False, ts, i, ch))
    
    with new.cursor() as ncur:
        ncur.execute("DELETE FROM messages WHERE conversation_id = %s", (v2_id,))
        execute_values(ncur, """
            INSERT INTO messages (conversation_id, uuid, parent_uuid, role, content,
                                  thinking, tool_calls, tool_results, is_sidechain,
                                  timestamp, sequence_num, content_hash)
            VALUES %s
        """, rows, page_size=500)
    
    msg_count += len(rows)
    
    if msg_count % 10000 == 0:
        new.commit()
        print(f"  {msg_count} messages inserted ({time.time()-start:.0f}s)")

new.commit()

# Update message counts
with new.cursor() as cur:
    cur.execute("""
        UPDATE conversations SET message_count = (
            SELECT COUNT(*) FROM messages WHERE conversation_id = conversations.id
        ) WHERE source = 'web'
    """)
new.commit()

elapsed = time.time() - start
print(f"\nDone: {inserted} new conversations, {msg_count} messages backfilled in {elapsed:.0f}s")

old.close()
new.close()
