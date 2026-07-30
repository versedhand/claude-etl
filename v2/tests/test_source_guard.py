"""Tests for the D1 source-collision guard (_assert_same_source).

The guard sits in front of a destructive DELETE. These tests assert it fails
CLOSED: the only permitted overwrites are (a) re-ingesting the identical file and
(b) a provable relocation of a transcript named after the session it claims.

The pure-unit tests need no database. The live-DB test at the bottom is skipped
unless CONVERSATIONS_DB_PASSWORD is set; it neutralises commit() and forces a
rollback so it can never persist anything.

Run:  cd ~/projects/claude-etl && python3 -m pytest v2/tests/test_source_guard.py -v
"""

import os
import uuid

import pytest

from v2.db import SourceFileCollision, _assert_same_source
from v2.subagent import subagent_identity

SID = "8f2a1c44-0000-4000-8000-000000000001"


def _allows(stored, incoming, session_id=SID):
    """True if the guard permits the overwrite."""
    try:
        _assert_same_source(stored, incoming, session_id)
        return True
    except SourceFileCollision:
        return False


class TestAllowed:
    def test_identical_path_is_idempotent_reingest(self):
        # Rule 1. Without this every conversation would stop updating.
        p = f"proj-a/{SID}.jsonl"
        assert _allows(p, p)

    def test_relocation_of_transcript_named_after_its_session(self):
        # Rule 4: project directory renamed. Same session UUID filename.
        assert _allows(f"old-proj/{SID}.jsonl", f"new-proj/{SID}.jsonl")


class TestBlocked:
    def test_same_basename_different_directory_is_blocked(self):
        # Hole (a). 'journal.jsonl' repeats in every workflow directory; a bare
        # basename match let one file's messages replace an unrelated file's.
        sub_sid, _ = subagent_identity(f"proj/{SID}/subagents/workflows/wf_1/journal.jsonl")
        assert not _allows(
            f"proj/{SID}/subagents/workflows/wf_1/journal.jsonl",
            f"proj/{SID}/subagents/workflows/wf_2/journal.jsonl",
            session_id=sub_sid,
        )

    def test_same_basename_agent_file_different_dir_is_blocked(self):
        sub_sid, _ = subagent_identity(f"proj/{SID}/subagents/agent-aa19.jsonl")
        assert not _allows(
            f"projA/{SID}/subagents/agent-aa19.jsonl",
            f"projB/{SID}/subagents/agent-aa19.jsonl",
            session_id=sub_sid,
        )

    def test_matching_basename_that_is_not_the_session_id_is_blocked(self):
        # Basenames match but the file is not named after the session it claims,
        # so relocation is not provable.
        assert not _allows("old-proj/journal.jsonl", "new-proj/journal.jsonl")

    @pytest.mark.parametrize(
        "stored",
        [
            "legacy-import",
            "legacy-web-import",
            "legacy-import-v1mig-20260730",
            "legacy-import-v1mig-20260730-remnant",
            "data-a318eac0-0f78-4c9e-8755-98c7acea7e0e-1777663639-01554b1d-batch-0000.zip",
        ],
    )
    def test_non_jsonl_stored_source_is_blocked(self, stored):
        # Hole (b): this used to RETURN EARLY, allowing a file ingest to wipe any
        # of the 84,925 non-.jsonl-sourced conversations, including all 83 rows
        # migrated from v1.
        assert not _allows(stored, f"proj/{SID}.jsonl")

    def test_non_jsonl_incoming_source_is_blocked(self):
        assert not _allows(f"proj/{SID}.jsonl", "legacy-import")

    def test_two_unrelated_transcripts_are_blocked(self):
        other = "99999999-0000-4000-8000-00000000ffff"
        assert not _allows(f"proj/{SID}.jsonl", f"proj/{other}.jsonl")

    def test_error_names_both_paths(self):
        with pytest.raises(SourceFileCollision) as e:
            _assert_same_source("legacy-import", f"proj/{SID}.jsonl", SID)
        msg = str(e.value)
        assert "legacy-import" in msg and SID in msg


class TestSubagentIdentityNoCollision:
    """A subagent session_id can never satisfy the relocation branch."""

    def test_sibling_journals_get_distinct_identities(self):
        a, _ = subagent_identity(f"proj/{SID}/subagents/workflows/wf_1/journal.jsonl")
        b, _ = subagent_identity(f"proj/{SID}/subagents/workflows/wf_2/journal.jsonl")
        assert a != b

    def test_subagent_session_id_never_equals_a_file_stem(self):
        sid, parent = subagent_identity(f"proj/{SID}/subagents/agent-aa19.jsonl")
        assert sid.startswith("sub:") and ":" in sid
        assert parent == SID
        # A stem comes from a filename and cannot contain ':' in these paths.
        assert sid != "agent-aa19"


@pytest.mark.skipif(
    not os.environ.get("CONVERSATIONS_DB_PASSWORD"),
    reason="CONVERSATIONS_DB_PASSWORD not set — skipping live-DB test",
)
class TestLiveDatabaseFailsClosed:
    def test_real_upsert_refuses_to_touch_a_legacy_conversation(self):
        """Exercise the real upsert against a real legacy row.

        commit() is neutralised and the transaction is rolled back, so this
        cannot persist a change even if the guard were removed.
        """
        from v2.db import get_connection, upsert_conversation
        from v2.parser import ParsedConversation

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT c.session_id, c.machine, c.user_name, c.source_file,
                              (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id)
                       FROM conversations c
                       WHERE c.source_file = 'legacy-import'
                       ORDER BY c.message_count DESC LIMIT 1"""
                )
                row = cur.fetchone()
            if row is None:
                pytest.skip("no legacy-import conversation present")
            sid, machine, user, stored_src, before = row

            parsed = ParsedConversation(
                session_id=sid, project="x", model=None,
                started_at=None, last_message_at=None,
                message_count=0, messages=[],
            )

            # psycopg2's connection.commit is read-only, so wrap it. Every other
            # attribute (cursor(), rollback()) passes straight through to the real
            # connection — only persistence is disabled.
            class NoCommit:
                def __init__(self, real):
                    self._real = real

                def __getattr__(self, name):
                    return getattr(self._real, name)

                def commit(self):
                    raise AssertionError("commit() must not be reached in this test")

            with pytest.raises(SourceFileCollision):
                upsert_conversation(
                    NoCommit(conn), parsed, user_name=user, machine=machine,
                    source_file=f"some-project/{sid}.jsonl",
                    file_hash=uuid.uuid4().hex,
                )
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            conn.close()

        # Read back on a FRESH connection — a different path than the one that
        # would have made the change.
        verify = get_connection()
        try:
            with verify.cursor() as cur:
                cur.execute(
                    """SELECT (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id),
                              c.source_file
                       FROM conversations c
                       WHERE c.session_id = %s AND c.machine = %s AND c.user_name = %s""",
                    (sid, machine, user),
                )
                after, src_after = cur.fetchone()
        finally:
            verify.close()

        assert after == before, f"messages lost: {before} -> {after}"
        assert src_after == stored_src
