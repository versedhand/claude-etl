# Claude ETL

ETL scripts for loading Claude conversations into LifeDB.

## Contents

| File | Purpose |
|------|---------|
| `README.md` | Documentation and usage instructions |
| `code_ingest.py` | Claude Code conversation ingestion from synced directories |
| `web_ingest.py` | Claude Web conversation ingestion from ZIP exports |
| `chatgpt_ingest.py` | ChatGPT conversation ingestion with attachment extraction |
| `normalize.py` | Transform raw data into normalized + unified layers |
| `embed.py` | Generate OpenAI embeddings for semantic search |
| `chunk_conversations.py` | Topic-based conversation chunking with LLM summaries |
| `process_all.py` | Full pipeline orchestration script |
| `reembed_topics.py` | One-time migration script for re-embedding topics |
| `cron-ingest.sh` | Hourly cron job for server deployment |
| `venv/` | Python virtual environment |

## Status

| Source | Status | Records | Last Run |
|--------|--------|---------|----------|
| Claude Code | Complete | 142,394 messages / 27,252 conversations | 2026-01-19 |
| Claude Web | Complete | 102,742 canonical / 108,755 total messages / 4,648 conversations | 2026-01-19 |
| ChatGPT | Complete | 9,677 canonical messages / 705 conversations / 381 attachments | 2026-01-22 |

## Setup

```bash
pip install psycopg2-binary
export LIFEDB_PASSWORD='your-password'
```

## Scripts

### code_ingest.py

Ingests Claude Code conversations from synced `.claude/projects/` directories.

```bash
python3 code_ingest.py --dry-run     # Count files
python3 code_ingest.py --device blue # Single device
python3 code_ingest.py               # Full ingestion
```

### web_ingest.py

Ingests Claude Web exports from ZIP files in inbox.

```bash
# Drop ZIP exports in inbox
cp ~/Downloads/data-*.zip /mnt/d/obs/life-var/inbox/claude-web-exports/

# Dry run - count conversations/messages
python3 web_ingest.py --dry-run

# Full ingestion (auto-moves to processed/ on success)
python3 web_ingest.py
```

**Features:**
- Skip logic: only updates if new data has newer `updated_at`
- Per-conversation commits (prevents server memory issues)
- Auto-moves successful imports to `processed/` folder
- Stores all messages (branching filtered later in normalize.py)

### normalize.py

Transforms raw data into normalized + unified layers.

```bash
python3 normalize.py
```

**What it does:**
1. Extracts accounts from raw → `claude_web_accounts`
2. Extracts conversations → `claude_web_conversations`, `claude_code_conversations`
3. Computes `is_canonical` on web messages (filters abandoned edit branches)
4. Populates unified tables: `claude_conversations`, `claude_messages`, `claude_accounts`

**Branch detection algorithm:**
- Claude exports don't include `parent_message_uuid`, but messages are in chronological array order
- When user edits a message, edit appears as NEW message with later timestamp
- Abandoned edits = consecutive same-sender messages (user edited multiple times without getting response)
- Canonical message = LAST in each same-sender run (the one conversation continued from)
- Result: ~5.5% of web messages filtered as non-canonical (abandoned edits)

**Example:**
```
[human, assistant, human, human, human, assistant]
   ✓       ✓                         ✓       ✓
Edits 3-4 filtered; canonical path is: 1 → 2 → 5 → 6
```

**Known limitation:** When each edit gets its own response (strict human/assistant alternation), we can't detect which are abandoned edit branches. Claude exports lack `parent_message_uuid` field needed for true tree reconstruction. Only consecutive same-sender messages are filterable.

### embed.py

Generates OpenAI embeddings for semantic search.

```bash
# Test with limit
python3 embed.py --limit 100

# Full run (~$3 for 211k messages)
python3 embed.py

# Quiet mode
python3 embed.py --quiet
```

**Features:**
- Keyed by source UUID (survives re-ingestion)
- Content hash for change detection
- Batches of 100 with rate limiting
- HNSW index for fast vector search

**Requires:** `OPENAI_API_KEY` env var or `/life-etc/api-keys/openai.key`

## Schema Architecture

Three-layer design: Raw → Normalized → Unified

### Layer 1: Raw (current)

Preserves full export data, nothing lost.

| Table | Purpose |
|-------|---------|
| `claude_code_raw` | Code JSONL records (line-by-line) |
| `claude_web_raw` | Web conversations ≥2025-10-01 (full JSONB) |
| `claude_web_archive` | Web conversations <2025-10-01 (full JSONB) |
| `claude_web_messages` | Web messages denormalized (all branches) |

### Layer 2: Normalized (planned)

Per-source normalized tables, full fidelity.

```
Web:
  claude_web_imports        (import tracking)
  claude_web_accounts       (account metadata)
  claude_web_conversations  (conversation metadata)
  claude_web_messages       (all messages, is_canonical flag)

Code:
  claude_code_imports
  claude_code_conversations
  claude_code_messages
```

### Layer 3: Unified (planned)

Canonical branches only, both sources merged.

```
  claude_accounts       (all accounts - nothing lost)
  claude_conversations  (all conversations - nothing lost)
  claude_messages       (canonical messages only - abandoned edits filtered)
```

## Configuration

| Setting | Value |
|---------|-------|
| HOMES_BASE | `/mnt/d/obs/life-var/homes` |
| INBOX | `/mnt/d/obs/life-var/inbox/claude-web-exports` |
| DEVICES | blue, black, red, magenta |
| DB_HOST | 100.86.218.108 (Tailscale) |
| CUTOFF_DATE | 2025-10-01 (archive threshold) |

## Notes

**Null byte handling:** PostgreSQL JSONB rejects `\u0000`. Scripts include `sanitize_null_bytes()` to strip them.

**Branching:** Web conversations can have branches (edited messages). Claude exports don't include `parent_message_uuid`, so `normalize.py` uses timestamp-based detection: conversations with out-of-order timestamps or >2 messages at same timestamp are branched. Canonical path = trace from latest message backward, alternating human/assistant sender.
