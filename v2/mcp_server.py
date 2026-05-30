#!/usr/bin/env python3
"""
MCP server for searching Claude conversations.

Tools:
  - search_text: Full-text search across messages (requires 'after' date)
  - search_semantic: Semantic/similarity search using embeddings (requires 'after' date)
  - search_sql: Run arbitrary read-only SQL against the conversations database
  - list_recent: Browse recent conversations
  - get_conversation: Full conversation by session ID
  - get_messages: Filtered messages from a conversation

Runs as: python3 -m v2.mcp_server
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": os.environ.get("CONVERSATIONS_DB_HOST", "100.127.104.75"),
    "port": int(os.environ.get("CONVERSATIONS_DB_PORT", "5432")),
    "dbname": os.environ.get("CONVERSATIONS_DB_NAME", "conversationsdb_v2"),
    "user": os.environ.get("CONVERSATIONS_DB_USER", "conversations_writer"),
    "password": os.environ.get("CONVERSATIONS_DB_PASSWORD", ""),
}

TOOL_DEFINITIONS = [
    {
        "name": "search_text",
        "description": (
            "Full-text search across conversation messages. Uses PostgreSQL FTS. "
            "The 'after' parameter is REQUIRED to scope the search window."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["query", "after"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text search query — matched using PostgreSQL full-text search",
                },
                "after": {
                    "type": "string",
                    "description": "Only search messages after this date (ISO 8601, e.g. '2026-05-01'). REQUIRED.",
                },
                "before": {
                    "type": "string",
                    "description": "Only search messages before this date (ISO 8601)",
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant"],
                    "description": "Filter by message role. Default: both.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return. Default: 20.",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "search_semantic",
        "description": (
            "Semantic similarity search across conversation messages using embeddings. "
            "Finds messages with similar meaning, not just matching words. "
            "The 'after' parameter is REQUIRED to scope the search window."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["query", "after"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query — finds semantically similar messages",
                },
                "after": {
                    "type": "string",
                    "description": "Only search messages after this date (ISO 8601). REQUIRED.",
                },
                "before": {
                    "type": "string",
                    "description": "Only search messages before this date (ISO 8601)",
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant"],
                    "description": "Filter by message role. Default: both.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return. Default: 20.",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "search_sql",
        "description": (
            "Run a read-only SQL query against the conversations database. "
            "Tables: conversations (session_id, user_name, machine, model, started_at, last_message_at, message_count), "
            "messages (conversation_id, role, content, thinking, tool_calls, timestamp, sequence_num, content_hash). "
            "SELECT only."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["sql"],
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL query (SELECT/WITH only). No mutations.",
                },
            },
        },
    },
    {
        "name": "list_recent",
        "description": "List recent conversations with metadata (session ID, date, message count, first message preview).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max conversations to return. Default: 10.",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "get_conversation",
        "description": "Get full conversation metadata and message summary by session ID.",
        "inputSchema": {
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The conversation session ID (UUID)",
                },
            },
        },
    },
    {
        "name": "get_messages",
        "description": "Get messages from a specific conversation, optionally filtered.",
        "inputSchema": {
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The conversation session ID (UUID)",
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant"],
                    "description": "Filter by role",
                },
                "after": {
                    "type": "string",
                    "description": "Messages after this timestamp",
                },
                "before": {
                    "type": "string",
                    "description": "Messages before this timestamp",
                },
            },
        },
    },
]


def get_conn():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


def _build_message_result(row) -> dict:
    """Format a message row for API output."""
    content = row["content"] or ""
    return {
        "session_id": row["session_id"],
        "role": row["role"],
        "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
        "user": row["user_name"],
        "model": row["model"],
        "conversation_started": row["started_at"].isoformat() if row["started_at"] else None,
        "content_preview": content[:500] + ("..." if len(content) > 500 else ""),
    }


def handle_search_text(params: dict) -> str:
    query = params["query"]
    after = params["after"]
    before = params.get("before")
    role = params.get("role")
    limit = params.get("limit", 20)

    conn = get_conn()
    cur = conn.cursor()

    conditions = ["m.timestamp >= %s"]
    values = [after]

    if before:
        conditions.append("m.timestamp <= %s")
        values.append(before)
    if role:
        conditions.append("m.role = %s")
        values.append(role)

    conditions.append(
        "to_tsvector('english', coalesce(m.content, '')) @@ plainto_tsquery('english', %s)"
    )
    values.append(query)

    where = " AND ".join(conditions)

    sql = f"""
        SELECT m.content, m.role, m.timestamp, m.sequence_num,
               c.session_id, c.user_name, c.started_at, c.model,
               ts_rank(to_tsvector('english', coalesce(m.content, '')),
                       plainto_tsquery('english', %s)) as rank
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE {where}
        ORDER BY rank DESC
        LIMIT %s
    """
    all_values = [query] + values + [limit]

    try:
        cur.execute(sql, all_values)
        rows = cur.fetchall()
    except Exception as e:
        conn.close()
        return json.dumps({"error": str(e)})

    results = [_build_message_result(row) | {"rank": float(row["rank"])} for row in rows]
    conn.close()
    return json.dumps({"results": results, "count": len(results)})


def handle_search_semantic(params: dict) -> str:
    query = params["query"]
    after = params["after"]
    before = params.get("before")
    role = params.get("role")
    limit = params.get("limit", 20)

    # Generate embedding for the query
    from openai import OpenAI
    api_key = os.environ.get(
        "OPENAI_API_KEY",
        open(os.path.expanduser(
            "~/corpus/isaac-workspace-corpus/etc/api-keys/openai.key"
        )).read().strip()
    )
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[query],
    )
    query_embedding = response.data[0].embedding

    conn = get_conn()
    cur = conn.cursor()

    conditions = ["m.timestamp >= %s", "e.embedding IS NOT NULL"]
    values = [after]

    if before:
        conditions.append("m.timestamp <= %s")
        values.append(before)
    if role:
        conditions.append("m.role = %s")
        values.append(role)

    where = " AND ".join(conditions)

    sql = f"""
        SELECT m.content, m.role, m.timestamp, m.sequence_num,
               c.session_id, c.user_name, c.started_at, c.model,
               (e.embedding <=> %s::halfvec) as distance
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        JOIN embeddings e ON m.content_hash = e.content_hash
        WHERE {where}
        ORDER BY distance ASC
        LIMIT %s
    """
    all_values = [str(query_embedding)] + values + [limit]

    try:
        cur.execute(sql, all_values)
        rows = cur.fetchall()
    except Exception as e:
        conn.close()
        return json.dumps({"error": str(e)})

    results = [
        _build_message_result(row) | {"similarity": round(1 - float(row["distance"]), 4)}
        for row in rows
    ]
    conn.close()
    return json.dumps({"results": results, "count": len(results)})


def handle_search_sql(params: dict) -> str:
    sql = params["sql"].strip()

    # Safety: only allow SELECT/WITH
    first_word = sql.split()[0].upper() if sql else ""
    if first_word not in ("SELECT", "WITH", "EXPLAIN"):
        return json.dumps({"error": "Only SELECT/WITH/EXPLAIN queries allowed"})

    conn = get_conn()
    cur = conn.cursor()

    try:
        cur.execute(sql)
        if cur.description:
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            # RealDictCursor returns dicts already
            results = []
            for row in rows:
                clean = {}
                for k, v in row.items():
                    if isinstance(v, datetime):
                        clean[k] = v.isoformat()
                    elif isinstance(v, (list, dict)):
                        clean[k] = v
                    else:
                        clean[k] = v
                results.append(clean)
            conn.close()
            return json.dumps({"columns": columns, "rows": results, "count": len(results)})
        else:
            conn.close()
            return json.dumps({"message": "Query executed, no results"})
    except Exception as e:
        conn.close()
        return json.dumps({"error": str(e)})


def handle_list_recent(params: dict) -> str:
    limit = params.get("limit", 10)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT c.session_id, c.user_name, c.machine, c.model,
               c.started_at, c.last_message_at, c.message_count,
               (SELECT content FROM messages WHERE conversation_id = c.id
                AND role = 'user' ORDER BY sequence_num LIMIT 1) as first_message
        FROM conversations c
        ORDER BY c.last_message_at DESC NULLS LAST
        LIMIT %s
    """, (limit,))

    rows = cur.fetchall()
    conn.close()

    results = []
    for row in rows:
        first_msg = row["first_message"] or ""
        results.append({
            "session_id": row["session_id"],
            "user": row["user_name"],
            "machine": row["machine"],
            "model": row["model"],
            "started_at": row["started_at"].isoformat() if row["started_at"] else None,
            "last_message_at": row["last_message_at"].isoformat() if row["last_message_at"] else None,
            "message_count": row["message_count"],
            "first_message_preview": first_msg[:200] + ("..." if len(first_msg) > 200 else ""),
        })

    return json.dumps({"conversations": results, "count": len(results)})


def handle_get_conversation(params: dict) -> str:
    session_id = params["session_id"]

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT session_id, user_name, machine, project, model,
               started_at, last_message_at, message_count, ingested_at
        FROM conversations WHERE session_id = %s
    """, (session_id,))

    row = cur.fetchone()
    if not row:
        conn.close()
        return json.dumps({"error": f"Conversation {session_id} not found"})

    # Get message role counts
    cur.execute("""
        SELECT role, COUNT(*) as count
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE c.session_id = %s
        GROUP BY role
    """, (session_id,))
    role_counts = {r["role"]: r["count"] for r in cur.fetchall()}

    conn.close()

    return json.dumps({
        "session_id": row["session_id"],
        "user": row["user_name"],
        "machine": row["machine"],
        "project": row["project"],
        "model": row["model"],
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "last_message_at": row["last_message_at"].isoformat() if row["last_message_at"] else None,
        "message_count": row["message_count"],
        "role_counts": role_counts,
        "ingested_at": row["ingested_at"].isoformat() if row["ingested_at"] else None,
    })


def handle_get_messages(params: dict) -> str:
    session_id = params["session_id"]
    role = params.get("role")
    after = params.get("after")
    before = params.get("before")

    conn = get_conn()
    cur = conn.cursor()

    conditions = ["c.session_id = %s"]
    values = [session_id]

    if role:
        conditions.append("m.role = %s")
        values.append(role)
    if after:
        conditions.append("m.timestamp >= %s")
        values.append(after)
    if before:
        conditions.append("m.timestamp <= %s")
        values.append(before)

    where = " AND ".join(conditions)

    cur.execute(f"""
        SELECT m.role, m.content, m.thinking, m.tool_calls,
               m.timestamp, m.sequence_num, m.is_sidechain
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE {where}
        ORDER BY m.sequence_num
    """, values)

    rows = cur.fetchall()
    conn.close()

    messages = []
    for row in rows:
        msg = {
            "role": row["role"],
            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
            "sequence": row["sequence_num"],
        }
        if row["content"]:
            msg["content"] = row["content"]
        if row["thinking"]:
            msg["thinking"] = row["thinking"][:500] + "..." if len(row["thinking"] or "") > 500 else row["thinking"]
        if row["tool_calls"]:
            # Summarize tool calls instead of dumping full JSON
            calls = row["tool_calls"]
            if isinstance(calls, list):
                msg["tools_used"] = [c.get("name", "?") for c in calls]
        if row["is_sidechain"]:
            msg["is_sidechain"] = True
        messages.append(msg)

    return json.dumps({"session_id": session_id, "messages": messages, "count": len(messages)})


HANDLERS = {
    "search_text": handle_search_text,
    "search_semantic": handle_search_semantic,
    "search_sql": handle_search_sql,
    "list_recent": handle_list_recent,
    "get_conversation": handle_get_conversation,
    "get_messages": handle_get_messages,
}


# ── MCP stdio protocol ──────────────────────────────────────────────

def read_message():
    """Read a JSON-RPC message from stdin."""
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)


def write_message(msg):
    """Write a JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def handle_initialize(msg):
    return {
        "jsonrpc": "2.0",
        "id": msg["id"],
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "conversations",
                "version": "2.0.0",
            },
        },
    }


def handle_tools_list(msg):
    return {
        "jsonrpc": "2.0",
        "id": msg["id"],
        "result": {"tools": TOOL_DEFINITIONS},
    }


def handle_tools_call(msg):
    params = msg.get("params", {})
    tool_name = params.get("name")
    tool_args = params.get("arguments", {})

    handler = HANDLERS.get(tool_name)
    if not handler:
        return {
            "jsonrpc": "2.0",
            "id": msg["id"],
            "result": {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True,
            },
        }

    try:
        result_text = handler(tool_args)
    except Exception as e:
        logger.error(f"Tool {tool_name} failed: {e}")
        result_text = json.dumps({"error": str(e)})

    return {
        "jsonrpc": "2.0",
        "id": msg["id"],
        "result": {
            "content": [{"type": "text", "text": result_text}],
        },
    }


def main():
    """Run the MCP server on stdio."""
    logger.info("Conversations MCP server starting")

    while True:
        msg = read_message()
        if msg is None:
            break

        method = msg.get("method", "")

        if method == "initialize":
            write_message(handle_initialize(msg))
        elif method == "notifications/initialized":
            pass  # no response needed
        elif method == "tools/list":
            write_message(handle_tools_list(msg))
        elif method == "tools/call":
            write_message(handle_tools_call(msg))
        elif method == "ping":
            write_message({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
        else:
            if "id" in msg:
                write_message({
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                })


if __name__ == "__main__":
    main()
