"""Tests for database operations. Requires a live conversationsdb_v2 connection."""

import os
import pytest
from pathlib import Path

# Skip all tests if DB is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("CONVERSATIONS_DB_PASSWORD"),
    reason="CONVERSATIONS_DB_PASSWORD not set — skipping DB tests"
)

from v2.parser import parse_conversation
from v2.db import get_connection, ensure_schema, upsert_conversation

FIXTURE_DIR = Path(__file__).parent
SMALL_FIXTURE = FIXTURE_DIR / "fixture-small.jsonl"


@pytest.fixture
def db_conn():
    """Get a DB connection and ensure schema exists."""
    conn = get_connection()
    ensure_schema(conn)
    yield conn
    conn.close()


class TestSchema:
    def test_tables_exist(self, db_conn):
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name IN ('conversations', 'messages')"
            )
            tables = {row[0] for row in cur.fetchall()}
        assert "conversations" in tables
        assert "messages" in tables

    def test_indexes_exist(self, db_conn):
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename IN ('conversations', 'messages')"
            )
            indexes = {row[0] for row in cur.fetchall()}
        assert "idx_messages_content_trgm" in indexes
        assert "idx_conversations_unique" in indexes


class TestUpsert:
    def test_insert_new_conversation(self, db_conn):
        parsed = parse_conversation(SMALL_FIXTURE)
        conv_id = upsert_conversation(
            db_conn, parsed,
            user_name="test_user", machine="test_machine",
            source_file="test/fixture-small.jsonl"
        )
        assert conv_id is not None

        # Verify in DB
        with db_conn.cursor() as cur:
            cur.execute("SELECT message_count FROM conversations WHERE id = %s", (conv_id,))
            row = cur.fetchone()
            assert row[0] == parsed.message_count

        # Cleanup
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id = %s", (conv_id,))
        db_conn.commit()

    def test_idempotent_upsert(self, db_conn):
        parsed = parse_conversation(SMALL_FIXTURE)

        # Insert twice
        conv_id1 = upsert_conversation(
            db_conn, parsed,
            user_name="test_user", machine="test_machine",
            source_file="test/fixture-small.jsonl"
        )
        conv_id2 = upsert_conversation(
            db_conn, parsed,
            user_name="test_user", machine="test_machine",
            source_file="test/fixture-small.jsonl"
        )

        assert conv_id1 == conv_id2  # Same conversation, same ID

        # Verify only one conversation exists
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM conversations WHERE session_id = %s AND machine = %s",
                (parsed.session_id, "test_machine")
            )
            assert cur.fetchone()[0] == 1

            # Verify correct message count (not doubled)
            cur.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = %s",
                (conv_id1,)
            )
            assert cur.fetchone()[0] == parsed.message_count

        # Cleanup
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id = %s", (conv_id1,))
        db_conn.commit()

    def test_text_search_works(self, db_conn):
        parsed = parse_conversation(SMALL_FIXTURE)
        conv_id = upsert_conversation(
            db_conn, parsed,
            user_name="test_user", machine="test_machine",
            source_file="test/fixture-small.jsonl"
        )

        # Trigram search
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = %s AND content IS NOT NULL",
                (conv_id,)
            )
            has_content = cur.fetchone()[0] > 0
            assert has_content

        # Cleanup
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id = %s", (conv_id,))
        db_conn.commit()
