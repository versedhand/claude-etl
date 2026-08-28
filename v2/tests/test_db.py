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




class TestAppendOnlyFastPath:
    """The sync cron re-ingests a live transcript every minute. Before 2026-08-28
    each pass DELETEd and re-INSERTed every message of that session, so WAL scaled
    with (session size x times saved) — measured at 25-33 GB/day off a 2.5 GB
    table, filling the backup volume. These tests pin the two properties that make
    appending safe: take the fast path on a real append, REFUSE it on anything not
    provably a prefix.

    ⚠️ Every test gets its OWN session_id and cleans up in a finally. They shared
    the fixture's session_id at first, so one failure skipped its cleanup and the
    next three died on SourceFileCollision instead of their own assertion — four
    reds, one of them real. A cascading suite cannot tell you which check fired.
    """

    MACHINE = "test_append_machine"

    @staticmethod
    def _file(tmp_path, n_lines=None, name="t.jsonl"):
        """A transcript, optionally truncated to its first n_lines — i.e. the same
        session mid-flight."""
        lines = SMALL_FIXTURE.read_text().splitlines(keepends=True)
        p = tmp_path / name
        p.write_text("".join(lines if n_lines is None else lines[:n_lines]))
        return p

    def _ingest(self, db_conn, path, session_id):
        parsed = parse_conversation(path)
        parsed.session_id = session_id
        conv_id = upsert_conversation(
            db_conn, parsed, user_name="test_user", machine=self.MACHINE,
            source_file=str(path),
        )
        return parsed, conv_id

    @staticmethod
    def _rows(cur, conv_id):
        """sequence_num -> xmin. xmin is the load-bearing part: counting rows cannot
        tell an append from a full rewrite, because both end with the right number of
        correct rows. xmin changes if and only if the row was written again."""
        cur.execute(
            "SELECT sequence_num, xmin::text FROM messages WHERE conversation_id = %s",
            (conv_id,),
        )
        return dict(cur.fetchall())

    def _drop(self, db_conn, session_id):
        with db_conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE session_id = %s AND machine = %s",
                (session_id, self.MACHINE),
            )
        db_conn.commit()

    def test_append_does_not_rewrite_existing_rows(self, db_conn, tmp_path):
        sid = "append-grow-test"
        try:
            f = self._file(tmp_path, 20, "grow.jsonl")
            first, conv_id = self._ingest(db_conn, f, sid)
            assert first.message_count > 0, "prefix parsed to nothing — test is vacuous"
            with db_conn.cursor() as cur:
                before = self._rows(cur, conv_id)

            f.write_text(SMALL_FIXTURE.read_text())   # the file grows, as a live one does
            full, conv_id2 = self._ingest(db_conn, f, sid)
            assert conv_id2 == conv_id
            assert full.message_count > first.message_count, \
                "full fixture no longer than its prefix — test is vacuous"

            with db_conn.cursor() as cur:
                after = self._rows(cur, conv_id)
            assert len(after) == full.message_count          # correctness
            assert {s: v for s, v in after.items() if s in before} == before, \
                "existing message rows were rewritten — the DELETE path ran"
        finally:
            self._drop(db_conn, sid)

    def test_compaction_is_not_mistaken_for_an_append(self, db_conn, tmp_path):
        """⛔ Why a tail-only check is unsafe: a rewritten transcript can be LONGER
        than what is stored while sharing none of its messages."""
        sid = "append-compaction-test"
        try:
            f = self._file(tmp_path, 20, "compact.jsonl")
            self._ingest(db_conn, f, sid)
            with db_conn.cursor() as cur:
                cur.execute(
                    "UPDATE messages SET uuid = 'stale-' || sequence_num::text "
                    "WHERE conversation_id = (SELECT id FROM conversations "
                    "WHERE session_id = %s AND machine = %s)", (sid, self.MACHINE),
                )
            db_conn.commit()

            f.write_text(SMALL_FIXTURE.read_text())
            full, conv_id = self._ingest(db_conn, f, sid)

            with db_conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FILTER (WHERE uuid LIKE 'stale-%%'), COUNT(*) "
                    "FROM messages WHERE conversation_id = %s", (conv_id,),
                )
                stale, total = cur.fetchone()
            assert stale == 0, "divergent rows survived — fast path taken on a non-prefix"
            assert total == full.message_count
        finally:
            self._drop(db_conn, sid)

    def test_edited_content_falls_back_to_replace(self, db_conn, tmp_path):
        """Same uuids, changed content. A uuid-only check would call this a prefix."""
        sid = "append-edit-test"
        try:
            f = self._file(tmp_path, 20, "edit.jsonl")
            self._ingest(db_conn, f, sid)
            with db_conn.cursor() as cur:
                cur.execute(
                    "UPDATE messages SET content_hash = 'deadbeef' WHERE conversation_id = "
                    "(SELECT id FROM conversations WHERE session_id = %s AND machine = %s) "
                    "AND sequence_num = (SELECT MAX(sequence_num) FROM messages "
                    "WHERE conversation_id = (SELECT id FROM conversations "
                    "WHERE session_id = %s AND machine = %s))",
                    (sid, self.MACHINE, sid, self.MACHINE),
                )
                assert cur.rowcount == 1, "fault injection did not land — test is vacuous"
            db_conn.commit()

            f.write_text(SMALL_FIXTURE.read_text())
            full, conv_id = self._ingest(db_conn, f, sid)
            with db_conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FILTER (WHERE content_hash = 'deadbeef'), COUNT(*) "
                    "FROM messages WHERE conversation_id = %s", (conv_id,),
                )
                edited, total = cur.fetchone()
            assert edited == 0, "edited row survived — content was not checked"
            assert total == full.message_count
        finally:
            self._drop(db_conn, sid)

    def test_reingest_of_unchanged_file_still_replaces(self, db_conn, tmp_path):
        """No growth is not an append — the old idempotency guarantee is unchanged."""
        sid = "append-unchanged-test"
        try:
            f = self._file(tmp_path, 20, "same.jsonl")
            parsed, conv_id = self._ingest(db_conn, f, sid)
            self._ingest(db_conn, f, sid)
            with db_conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = %s", (conv_id,))
                assert cur.fetchone()[0] == parsed.message_count
        finally:
            self._drop(db_conn, sid)
