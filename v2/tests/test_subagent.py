"""Regression tests for D1: subagent transcripts destroying parent conversations.

A subagent transcript carries the PARENT session's `sessionId` in every record.
Ingesting one used to resolve to the parent's conversation row and delete its
messages. These tests lock in both halves of the fix:

  1. identity  — a subagent transcript gets its own path-derived session_id
                 that can never equal a parent session UUID
  2. the guard — even if identity were bypassed, upsert_conversation refuses to
                 overwrite a conversation whose stored source_file is a
                 different transcript
"""

import os
from datetime import datetime, timezone

import pytest

from v2.subagent import is_subagent_path, subagent_identity, is_subagent_session_id
from v2.parser import ParsedConversation, ParsedMessage

pytestmark_db = pytest.mark.skipif(
    not os.environ.get("CONVERSATIONS_DB_PASSWORD"),
    reason="CONVERSATIONS_DB_PASSWORD not set — skipping DB tests",
)

PARENT_UUID = "d5c84f86-9b1d-4fb4-86a3-b30d5926fe87"


class TestSubagentDetection:
    def test_parent_transcript_is_not_subagent(self):
        assert not is_subagent_path(f"proj/{PARENT_UUID}.jsonl")

    def test_subagent_transcript_detected(self):
        assert is_subagent_path(f"proj/{PARENT_UUID}/subagents/agent-abc.jsonl")

    def test_nested_workflow_subagent_detected(self):
        assert is_subagent_path(
            f"proj/{PARENT_UUID}/subagents/workflows/wf_1/agent-abc.jsonl"
        )

    def test_substring_in_project_name_is_not_a_match(self):
        # Only a full path SEGMENT named 'subagents' counts.
        assert not is_subagent_path("-home-subagents-notes/abc.jsonl")


class TestSubagentIdentity:
    def test_identity_never_equals_parent_uuid(self):
        sid, parent = subagent_identity(
            f"proj/{PARENT_UUID}/subagents/agent-abc.jsonl"
        )
        assert sid != PARENT_UUID
        assert parent == PARENT_UUID
        assert is_subagent_session_id(sid)

    def test_sibling_journals_get_distinct_identities(self):
        # journal.jsonl carries no agent id — filename alone is NOT unique.
        a, _ = subagent_identity(f"proj/{PARENT_UUID}/subagents/workflows/wf_1/journal.jsonl")
        b, _ = subagent_identity(f"proj/{PARENT_UUID}/subagents/workflows/wf_2/journal.jsonl")
        assert a != b

    def test_identity_is_deterministic(self):
        p = f"proj/{PARENT_UUID}/subagents/agent-abc.jsonl"
        assert subagent_identity(p) == subagent_identity(p)

    def test_rejects_non_subagent_path(self):
        with pytest.raises(ValueError):
            subagent_identity("proj/abc.jsonl")


def _parsed(session_id, n, tag):
    msgs = [
        ParsedMessage(
            uuid=f"{tag}-{i}", parent_uuid=None, role="user", content=f"{tag} body {i}",
            thinking=None, tool_calls=None, tool_results=None, is_sidechain=False,
            timestamp=datetime.now(timezone.utc), sequence_num=i,
        )
        for i in range(n)
    ]
    now = datetime.now(timezone.utc)
    return ParsedConversation(
        session_id=session_id, project="p", model="m",
        started_at=now, last_message_at=now, message_count=n, messages=msgs,
    )


@pytestmark_db
class TestSourceCollisionGuard:
    SID = "D1REGRESSION-TEST"
    MACHINE = "test_machine"
    USER = "test_user"

    @pytest.fixture
    def conn(self):
        from v2.db import get_connection
        c = get_connection()
        with c.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE session_id = %s", (self.SID,))
        c.commit()
        yield c
        with c.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE session_id = %s", (self.SID,))
        c.commit()
        c.close()

    def _count(self, conn):
        with conn.cursor() as cur:
            cur.execute(
                """SELECT count(*) FROM messages m JOIN conversations c
                   ON m.conversation_id = c.id WHERE c.session_id = %s""",
                (self.SID,),
            )
            return cur.fetchone()[0]

    def test_different_transcript_cannot_destroy_messages(self, conn):
        from v2.db import upsert_conversation, SourceFileCollision

        upsert_conversation(conn, _parsed(self.SID, 3, "real"),
                            self.USER, self.MACHINE, "proj/abc.jsonl")
        assert self._count(conn) == 3

        with pytest.raises(SourceFileCollision):
            upsert_conversation(conn, _parsed(self.SID, 99, "subagent"),
                                self.USER, self.MACHINE,
                                "proj/abc/subagents/agent-xyz.jsonl")
        conn.rollback()
        # The whole point: the real messages are still there.
        assert self._count(conn) == 3

    def test_same_file_reingest_still_allowed(self, conn):
        from v2.db import upsert_conversation

        upsert_conversation(conn, _parsed(self.SID, 3, "v1"),
                            self.USER, self.MACHINE, "proj/abc.jsonl")
        upsert_conversation(conn, _parsed(self.SID, 5, "v2"),
                            self.USER, self.MACHINE, "proj/abc.jsonl")
        assert self._count(conn) == 5

    def test_relocated_file_still_allowed(self, conn):
        # Project directory renamed — same transcript, so this must NOT be blocked.
        from v2.db import upsert_conversation

        upsert_conversation(conn, _parsed(self.SID, 3, "v1"),
                            self.USER, self.MACHINE, "proj/abc.jsonl")
        upsert_conversation(conn, _parsed(self.SID, 7, "v2"),
                            self.USER, self.MACHINE, "renamed/abc.jsonl")
        assert self._count(conn) == 7
