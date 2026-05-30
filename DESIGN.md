# Claude Conversation ETL — Design Document

**Author:** Eden Lux Ellington + Isaac Robinson
**Date:** 2026-05-29
**Status:** Design complete, implementation not started

## Problem

Claude Code sessions produce JSONL conversation files in `~/.claude/projects/` on each machine. Multiple users (Isaac, Racquel, Cesar, Jovelyn) across multiple machines need their conversations ingested into a searchable central database so agents can search past sessions.

The previous system (built Jan 2026) broke in May 2026 when the database was recreated with SQL_ASCII encoding during a decomposition. The hook-based real-time capture (PostToolUse + Stop → API server on port 8900) was fragile and is being replaced.

## Architecture

```
~/.claude/projects/*.jsonl       (live, Claude Code owns — never touch directly)
    ↓ rsync (local copy)
/var/lib/versedhand/claude-mirror/   (safe snapshot for reading)
    ↓ ingester (same script, runs sequentially after rsync)
conversationsdb @ 100.127.104.75     (central PostgreSQL, UTF8)
```

**One script, one cron.** The sync and ingest are sequential steps in a single script — never separate crons. This prevents the race condition where rsync is mid-copy while the ingester reads the mirror.

**Rsync invocation:**
```bash
rsync -a --delete --exclude='*.lock' --exclude='*.tmp' \
  ~/.claude/projects/ /var/lib/versedhand/claude-mirror/
```
`--delete` removes files from mirror that were deleted from source. Deleted sessions remain in the DB (retention is forever) — the mirror is just a snapshot for safe reading.

**Execution context:** Cron runs as the local Claude user (per-user crontab, NOT root). `user_name` derived from `whoami`, `machine` from `hostname`. Mirror and state file owned by the same user.

**Cadence:** Every 1 minute via cron. Cost when nothing changed: <100ms (stat all files, compare mtimes, exit). Cost when a conversation is active: one full file re-parse + DB upsert per changed file. Typical steady state: 1 conversation per minute per active session.

## Change Detection

The ingester maintains a local state file at `/var/lib/versedhand/ingest-state.json`:

```json
{
  "files": {
    "path/to/session.jsonl": {
      "mtime": 1748567890.123,
      "size": 456789,
      "last_ingested": "2026-05-29T19:00:00Z"
    }
  }
}
```

On each run:
1. Scan `/var/lib/versedhand/claude-mirror/` for all `.jsonl` files
2. Compare each file's mtime AND size against stored values
3. Optionally compare SHA256 content hash (catches mtime-only changes with no real content change)
4. Changed files → full re-parse entire file, upsert into DB
5. Unchanged files → skip
6. Update state file only for successfully ingested files — failures leave state unchanged so they retry next run

**Files are always fully re-ingested** when changed. JSONL structure contains branches, sidechains, parent-child relationships, and interleaved tool results that require full-file parsing to reconstruct the conversation tree. Byte-offset resumption is not possible.

## Idempotency

Re-ingesting the same file produces the same DB state. Per conversation:
1. DELETE all messages for this conversation_id
2. INSERT all messages from the parsed file
3. UPSERT conversation metadata

Wrapped in a transaction per conversation.

## Error Handling & Observability

- **Per-file error isolation:** If a single file fails to parse or ingest, log the error and continue processing other files. Do NOT update state for the failed file — it retries next run.
- **Logging:** Structured logging to `/var/log/versedhand/claude-etl.log` (or journalctl via systemd timer if we switch from cron).
- **Metrics:** Each run logs: files scanned, files changed, files ingested, errors, duration. Simple but sufficient.
- **Malformed JSONL:** Graceful skip of unparseable lines with warning. Future-proof: detect format version changes if Claude Code adds a version field.

## Database

**Name:** `conversationsdb`
**Host:** 100.127.104.75 (rir-lifedb container, CT 103)
**Encoding:** UTF8 (the old DB used SQL_ASCII — this broke everything)
**Create fresh.** Dump old DB first for insurance.

### Schema (Phase 1 — text search only)

```sql
CREATE DATABASE conversationsdb WITH ENCODING 'UTF8';

-- Conversations (one row per JSONL file / session)
-- session_id is unique per machine+user (composite constraint handles edge cases)
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    user_name TEXT NOT NULL,          -- isaac, racquel, cesar, jovelyn
    machine TEXT NOT NULL,            -- rir-claude, vh-racquel-laptop, etc.
    project TEXT,                     -- project path from .claude/projects/
    model TEXT,                       -- claude-opus-4-6, etc.
    source_file TEXT NOT NULL,        -- path in mirror
    started_at TIMESTAMPTZ,
    last_message_at TIMESTAMPTZ,
    message_count INT DEFAULT 0,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    file_mtime TIMESTAMPTZ           -- mtime of source file at ingest time
);

-- Messages (one row per user or assistant turn)
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    uuid TEXT,                        -- message UUID from JSONL
    parent_uuid TEXT,                 -- for branch reconstruction
    role TEXT NOT NULL,               -- user, assistant, system
    content TEXT,                     -- message text (user text or assistant response)
    thinking TEXT,                    -- assistant thinking blocks (if present)
    tool_calls JSONB,                -- tool use content blocks
    tool_results JSONB,              -- tool result content blocks
    is_sidechain BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMPTZ,
    sequence_num INT,                 -- order within conversation
    CONSTRAINT fk_conversation FOREIGN KEY (conversation_id)
        REFERENCES conversations(id) ON DELETE CASCADE
);

-- Indexes for Phase 1 (text search)
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_role ON messages(role);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);
CREATE INDEX idx_conversations_user ON conversations(user_name);
CREATE INDEX idx_conversations_machine ON conversations(machine);
CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE UNIQUE INDEX idx_conversations_unique ON conversations(session_id, machine, user_name);

-- Trigram index for LIKE '%keyword%' search
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_messages_content_trgm ON messages USING gin(content gin_trgm_ops);

-- Full-text search
CREATE INDEX idx_messages_content_fts ON messages USING gin(to_tsvector('english', coalesce(content, '')));
```

### Schema additions for Phase 2 (embeddings)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE messages ADD COLUMN embedding vector(3072);  -- text-embedding-3-large

CREATE INDEX idx_messages_embedding ON messages
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

### Schema additions for Phase 3 (web + ChatGPT)

```sql
-- Source type to distinguish Code vs Web vs ChatGPT
ALTER TABLE conversations ADD COLUMN source TEXT DEFAULT 'code';
-- 'code' = Claude Code JSONL
-- 'web' = claude.ai web export
-- 'chatgpt' = ChatGPT export

-- Web exports have different metadata
ALTER TABLE conversations ADD COLUMN account TEXT;  -- for multi-account web exports
```

## JSONL Parser

The parser must handle these JSONL record types:
- `type: "user"` — user messages (extract `message.content`)
- `type: "assistant"` — assistant responses (extract text, thinking, tool_use, tool_result from `message.content[]`)
- `type: "queue-operation"` — session metadata (enqueue has the initial prompt)
- `type: "summary"` — compaction summaries
- Records with `isSidechain: true` — mark but still store
- Records with `parentUuid` — for branch reconstruction

**Content extraction:** Assistant messages have content as an array of blocks (text, thinking, tool_use, tool_result). The parser extracts:
- `content` = concatenated text blocks
- `thinking` = concatenated thinking blocks
- `tool_calls` = array of tool_use blocks (as JSONB)
- `tool_results` = array of tool_result blocks (as JSONB)

## Deployment

**Ansible role:** `roles/claude-conversation-sync` (new role)

Deploys to every machine in the `workstations` group:
1. Creates `/var/lib/versedhand/claude-mirror/` (owned by the local user)
2. Installs the sync+ingest script to `/usr/local/bin/claude-conversation-sync`
3. Installs pip dependencies (psycopg2-binary, minimal)
4. Creates cron: `* * * * * /usr/local/bin/claude-conversation-sync`
5. Configures DB credentials via environment file at `/etc/versedhand/conversationsdb.env`

**DB credentials:** Stored in Ansible vault, deployed as env file. Each machine gets read-write access to conversationsdb.

## Phases

### Phase 1: Code JSONL ingestion + text search
- Fresh conversationsdb (UTF8)
- JSONL parser for Claude Code format
- Mirror + ingest script
- Trigram + FTS indexes
- Ansible deployment
- Tests: parser tests, idempotency tests, change detection tests
- **Deliverable:** Agents can `SELECT content FROM messages WHERE content LIKE '%keyword%'` across all users

### Phase 2: Embeddings
- OpenAI text-embedding-3-large (3072 dims)
- pgvector HNSW index
- Embed on ingest (new/changed messages only, track by per-message content hash to avoid re-embedding unchanged rows)
- Semantic search function
- **Deliverable:** Agents can do similarity search across conversations

### Phase 3: Web + ChatGPT exports
- Integrate web export ingester (from old claude-etl code)
- ChatGPT export ingester (from old claude-etl code)
- Unified schema (source column distinguishes origin)
- Import historical data from old DB dump
- **Deliverable:** All conversation sources in one searchable DB

### Phase 4: MCP server
- Lightweight MCP server wrapping conversationsdb queries
- Tools: search_conversations, get_conversation, search_messages
- Installable on any machine via Ansible
- **Deliverable:** Agents search conversations via MCP tools instead of raw SQL

## Testing Strategy

TDD throughout. Test categories:

1. **Parser tests** — Given a JSONL file, verify extracted conversations and messages match expected output. Use real JSONL files (anonymized if needed) as fixtures.
2. **Idempotency tests** — Ingest same file twice, verify DB state is identical.
3. **Change detection tests** — Verify only changed files are re-processed.
4. **Schema tests** — Verify search indexes work (trigram, FTS, vector similarity).
5. **Integration tests** — Full pipeline: write JSONL → rsync → ingest → query → verify results.

## Migration from Old System

1. Dump old `conversationsdb`: `pg_dump conversationsdb > /var/backups/conversationsdb-old-$(date +%Y%m%d).sql`
2. **DO NOT drop old DB yet.** Rename to `conversationsdb_legacy` or leave in place. It has 77K conversations including web + ChatGPT data that doesn't exist elsewhere.
3. Create fresh DB as `conversationsdb_v2` (or new name TBD) with UTF8 encoding
4. Run Phase 1 ingester against all existing JSONL files (full historical ingest)
5. Verify message counts are reasonable against old DB stats (77K conversations, 655K messages)
6. Web + ChatGPT data restored in Phase 3 from dump + original export files at `~/corpus/isaac-workspace-corpus/var/inbox/claude-web-exports/`

## Open Questions

## Resolved Questions

1. **DB roles:** Two roles. `conversations_writer` (used by ingest cron on each machine, full INSERT/DELETE/UPDATE). Per-user read roles (`conversations_reader_{user}`) with row-level security — employees can only read their own conversations. Isaac gets superuser read access to all.
2. **Retention:** Forever. No aging.
3. **Access control:** Row-level security. `user_name = current_user` filter enforced at DB level. Isaac exempt.
4. **Compaction:** Not an issue. JSONL files preserve ALL messages including pre-compaction ones. Compaction externalizes file attachments as `compact_file_reference` records but doesn't delete or replace messages. Full file re-ingest captures everything.
