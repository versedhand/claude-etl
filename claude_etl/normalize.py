#!/usr/bin/env python3
"""
Normalize Claude data from raw tables into normalized + unified layers.

Raw → Normalized → Unified (canonical only)
"""

import os
import sys
import psycopg2
from psycopg2.extras import execute_values

DB_CONFIG = {
    "host": "100.127.104.75",
    "port": 5432,
    "dbname": "lifedb",
    "user": "postgres",
    "password": os.environ["LIFEDB_PASSWORD"],
}


def populate_web_accounts(cur):
    """Extract unique accounts from raw web data."""
    print("Populating claude_web_accounts...")

    # Note: claude_web_archive deprecated 2026-01-21, data migrated to raw
    cur.execute("""
        INSERT INTO claude_web_accounts (account_id, email, first_seen, last_seen)
        SELECT DISTINCT
            account_id,
            account_email,
            MIN(created_at) as first_seen,
            MAX(updated_at) as last_seen
        FROM claude_web_raw
        WHERE account_id IS NOT NULL AND account_id != ''
        GROUP BY account_id, account_email
        ON CONFLICT (account_id) DO UPDATE SET
            last_seen = GREATEST(claude_web_accounts.last_seen, EXCLUDED.last_seen)
    """)
    print(f"  {cur.rowcount} accounts")


def populate_web_conversations(cur):
    """Extract conversations metadata from raw."""
    print("Populating claude_web_conversations...")

    # Note: claude_web_archive deprecated 2026-01-21, data migrated to raw
    cur.execute("""
        INSERT INTO claude_web_conversations
            (conversation_id, account_id, name, summary, created_at, updated_at)
        SELECT
            conversation_id::text,
            NULLIF(account_id, ''),
            name,
            data->>'summary',
            created_at,
            updated_at
        FROM claude_web_raw
        ON CONFLICT (conversation_id) DO UPDATE SET
            name = EXCLUDED.name,
            summary = EXCLUDED.summary,
            updated_at = GREATEST(claude_web_conversations.updated_at, EXCLUDED.updated_at)
    """)
    print(f"  {cur.rowcount} conversations")


def compute_canonical_web(cur):
    """
    Mark canonical messages in web conversations using array position.

    Claude exports store messages in chronological array order. When a user edits
    a message, the edit appears as a NEW message with a later timestamp - creating
    consecutive same-sender messages in the array.

    Algorithm: A message is canonical if it's the LAST of its sender type in a
    consecutive run. This filters out intermediate edits while keeping the final
    version that the conversation continued from.

    Example:
        [human, assistant, human, human, human, assistant]
         ^      ^                        ^      ^
         canonical (conversation flow:   1 → 2 → 5 → 6, edits 3-4 filtered)
    """
    print("Computing canonical paths for web messages...")

    # Reset all to non-canonical
    cur.execute("UPDATE claude_web_messages SET is_canonical = FALSE")

    # Get all conversations
    cur.execute("SELECT DISTINCT conversation_uuid FROM claude_web_messages")
    conv_uuids = [row[0] for row in cur.fetchall()]
    print(f"  Processing {len(conv_uuids)} conversations...")

    total_canonical = 0
    total_filtered = 0
    branched_count = 0

    for conv_uuid in conv_uuids:
        # Get messages in chronological order (matches export array order)
        cur.execute("""
            SELECT uuid, sender
            FROM claude_web_messages
            WHERE conversation_uuid = %s
            ORDER BY created_at, uuid
        """, (conv_uuid,))
        msgs = cur.fetchall()

        if not msgs:
            continue

        # Find canonical messages: last in each same-sender run
        canonical_uuids = []
        i = 0
        while i < len(msgs):
            curr_sender = msgs[i][1]

            # Find the end of this same-sender run
            run_end = i
            while run_end + 1 < len(msgs) and msgs[run_end + 1][1] == curr_sender:
                run_end += 1

            # The last message in this run is canonical
            canonical_uuids.append(msgs[run_end][0])
            i = run_end + 1

        # Track if this conversation had branches (filtered any messages)
        if len(canonical_uuids) < len(msgs):
            branched_count += 1

        # Mark canonical messages
        if canonical_uuids:
            cur.execute("""
                UPDATE claude_web_messages SET is_canonical = TRUE
                WHERE uuid = ANY(%s)
            """, (canonical_uuids,))

        total_canonical += len(canonical_uuids)
        total_filtered += len(msgs) - len(canonical_uuids)

    print(f"  {len(conv_uuids) - branched_count} linear conversations (all messages canonical)")
    print(f"  {branched_count} branched conversations (filtered to canonical path)")

    # Count results
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE is_canonical = TRUE) as canonical,
            COUNT(*) FILTER (WHERE is_canonical = FALSE) as non_canonical,
            COUNT(*) as total
        FROM claude_web_messages
    """)
    row = cur.fetchone()
    print(f"  {row[0]} canonical, {row[1]} non-canonical ({row[2]} total)")


def populate_code_conversations(cur):
    """Extract code conversations from raw."""
    print("Populating claude_code_conversations...")

    cur.execute("""
        INSERT INTO claude_code_conversations
            (conversation_id, device, project_path, created_at, updated_at)
        SELECT
            conversation_id::text,
            device,
            project_path,
            MIN(timestamp) as created_at,
            MAX(timestamp) as updated_at
        FROM claude_code_raw
        GROUP BY conversation_id, device, project_path
        ON CONFLICT (conversation_id) DO UPDATE SET
            updated_at = GREATEST(claude_code_conversations.updated_at, EXCLUDED.updated_at)
    """)
    print(f"  {cur.rowcount} conversations")


def populate_code_messages(cur):
    """Extract code messages from raw."""
    print("Populating claude_code_messages...")

    cur.execute("""
        INSERT INTO claude_code_messages
            (uuid, conversation_id, session_id, sender, content, created_at, record_type)
        SELECT
            record_uuid::text,
            conversation_id::text,
            session_id::text,
            CASE
                WHEN record_type = 'user' THEN 'human'
                WHEN record_type = 'assistant' THEN 'assistant'
                ELSE record_type
            END,
            COALESCE(
                data->>'message',
                data->'content'->0->>'text',
                data->>'text',
                ''
            ),
            timestamp,
            record_type
        FROM claude_code_raw
        WHERE record_type IN ('user', 'assistant', 'summary')
          AND record_uuid IS NOT NULL
        ON CONFLICT (uuid) DO NOTHING
    """)
    print(f"  {cur.rowcount} messages")


def populate_unified(cur):
    """Populate unified canonical tables."""
    print("\nPopulating unified tables (canonical only)...")

    # Accounts (web only for now - ChatGPT doesn't expose account IDs)
    cur.execute("""
        INSERT INTO claude_accounts (account_id, email, source, first_seen, last_seen)
        SELECT account_id, email, 'web', first_seen, last_seen
        FROM claude_web_accounts
        ON CONFLICT (account_id) DO UPDATE SET
            last_seen = GREATEST(claude_accounts.last_seen, EXCLUDED.last_seen)
    """)
    print(f"  {cur.rowcount} accounts")

    # Conversations (all sources: web, code, chatgpt)
    cur.execute("""
        INSERT INTO claude_conversations
            (conversation_id, account_id, name, summary, created_at, updated_at, source)
        SELECT
            conversation_id, account_id, name, summary, created_at, updated_at, 'web'
        FROM claude_web_conversations
        ON CONFLICT (conversation_id) DO NOTHING
    """)
    web_convos = cur.rowcount

    cur.execute("""
        INSERT INTO claude_conversations
            (conversation_id, name, created_at, updated_at, source, device, project_path)
        SELECT
            conversation_id, name, created_at, updated_at, 'code', device, project_path
        FROM claude_code_conversations
        ON CONFLICT (conversation_id) DO NOTHING
    """)
    code_convos = cur.rowcount

    # ChatGPT conversations
    cur.execute("""
        INSERT INTO claude_conversations
            (conversation_id, name, created_at, updated_at, source)
        SELECT
            conversation_id, title, created_at, updated_at, 'chatgpt'
        FROM chatgpt_raw
        ON CONFLICT (conversation_id) DO NOTHING
    """)
    chatgpt_convos = cur.rowcount
    print(f"  {web_convos + code_convos + chatgpt_convos} conversations "
          f"({web_convos} web, {code_convos} code, {chatgpt_convos} chatgpt)")

    # Messages (canonical only from web/chatgpt, all from code)
    cur.execute("""
        INSERT INTO claude_messages
            (uuid, conversation_id, sender, content, created_at, parent_uuid)
        SELECT
            uuid, conversation_uuid, sender, content, created_at, parent_message_uuid
        FROM claude_web_messages
        WHERE is_canonical = TRUE
        ON CONFLICT (uuid) DO NOTHING
    """)
    web_msgs = cur.rowcount

    cur.execute("""
        INSERT INTO claude_messages
            (uuid, conversation_id, sender, content, created_at, parent_uuid)
        SELECT
            m.uuid, m.conversation_id, m.sender, m.content, m.created_at, r.parent_uuid::text
        FROM claude_code_messages m
        LEFT JOIN claude_code_raw r ON m.uuid = r.record_uuid::text
        ON CONFLICT (uuid) DO NOTHING
    """)
    code_msgs = cur.rowcount

    # ChatGPT messages (canonical only)
    cur.execute("""
        INSERT INTO claude_messages
            (uuid, conversation_id, sender, content, created_at, parent_uuid)
        SELECT
            message_id, conversation_id,
            CASE sender
                WHEN 'user' THEN 'human'
                WHEN 'assistant' THEN 'assistant'
                ELSE sender
            END,
            content, created_at, parent_id
        FROM chatgpt_messages
        WHERE is_canonical = TRUE
        ON CONFLICT (uuid) DO NOTHING
    """)
    chatgpt_msgs = cur.rowcount
    print(f"  {web_msgs + code_msgs + chatgpt_msgs} messages "
          f"({web_msgs} web, {code_msgs} code, {chatgpt_msgs} chatgpt)")


def main():
    print("Claude Data Normalization")
    print("=" * 40)

    conn = psycopg2.connect(**DB_CONFIG)

    try:
        with conn.cursor() as cur:
            # Layer 2: Normalized
            print("\n--- Layer 2: Normalized ---")
            populate_web_accounts(cur)
            populate_web_conversations(cur)
            compute_canonical_web(cur)
            populate_code_conversations(cur)
            populate_code_messages(cur)

            # Layer 3: Unified
            print("\n--- Layer 3: Unified ---")
            populate_unified(cur)

        conn.commit()
        print("\n" + "=" * 40)
        print("Done!")

        # Summary
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    (SELECT COUNT(*) FROM claude_conversations) as convos,
                    (SELECT COUNT(*) FROM claude_messages) as msgs,
                    (SELECT COUNT(*) FROM claude_accounts) as accounts
            """)
            row = cur.fetchone()
            print(f"\nUnified totals: {row[0]} conversations, {row[1]} messages, {row[2]} accounts")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
