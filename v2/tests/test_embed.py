"""Tests for embedding logic — skip rules, content hashing, DB operations."""

import os
import pytest

from v2.embed import should_embed, content_hash, compute_hashes_for_messages, get_embedding_stats

# DB tests require connection
pytestmark_db = pytest.mark.skipif(
    not os.environ.get("CONVERSATIONS_DB_PASSWORD"),
    reason="CONVERSATIONS_DB_PASSWORD not set"
)


class TestShouldEmbed:
    """Test the skip rules that filter out noise."""

    def test_user_message_embeddable(self):
        assert should_embed("user", "What should we do about the housing situation?", None)

    def test_assistant_message_embeddable(self):
        assert should_embed("assistant", "I recommend talking to Hannah about extending the lease.", None)

    def test_system_role_skipped(self):
        assert not should_embed("system", "You are a helpful assistant.", None)

    def test_empty_content_skipped(self):
        assert not should_embed("user", "", None)
        assert not should_embed("user", None, None)
        assert not should_embed("user", "   ", None)

    def test_short_content_IS_embedded(self):
        """Short turns are embedded ON PURPOSE. There is no minimum length.

        This test asserted the opposite until 2026-07-30. Commit 9a450e4 deliberately
        removed MIN_CONTENT_LENGTH=20 because Isaac's ratifications are terse by design,
        so a byte floor preferentially deleted the decision record from the index — the
        verified case being the user turn "ok let's go" (11 chars, 2026-07-27), the Finch
        Harbor brand ratification cited across the corpus as a decided fact, which was
        `embedded = f` and therefore unreachable by semantic search.

        The behaviour change was intended and is correct; only this assertion was left
        behind, so the suite has been red ever since and could not act as a gate.
        Inverted to lock in the rule that actually holds.
        """
        assert should_embed("user", "ok", None)
        assert should_embed("user", "yes", None)
        assert should_embed("assistant", "Done.", None)

    def test_long_content_skipped(self):
        long_text = "x" * 5001
        assert not should_embed("user", long_text, None)

    def test_code_heavy_content_skipped(self):
        json_dump = '{"key": "value", "nested": {"a": 1, "b": [2,3,4]}, "more": {"c": true}}'
        assert not should_embed("assistant", json_dump, None)

    def test_tool_only_turn_skipped(self):
        # Assistant turn with tool calls but no text
        assert not should_embed("assistant", "", [{"name": "Read", "input": {}}])

    def test_mixed_content_embeddable(self):
        # Real assistant message with some code but mostly text
        msg = "The file at /home/user/config.json needs to be updated. I changed the port from 8080 to 9090 because the old port was conflicting with the proxy server."
        assert should_embed("assistant", msg, None)

    def test_borderline_length_included(self):
        # Exactly at minimum length
        msg = "This is just enough."
        assert should_embed("user", msg, None)

    def test_exactly_max_length_excluded(self):
        msg = "a" * 5001
        assert not should_embed("user", msg, None)

    def test_at_max_length_included(self):
        msg = "a " * 2499 + "ab"  # 5000 chars, mostly alpha
        assert should_embed("user", msg, None)


class TestContentHash:
    """Test content hashing for dedup."""

    def test_deterministic(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello world!")
        assert h1 != h2

    def test_returns_hex_string(self):
        h = content_hash("test")
        assert len(h) == 64  # SHA256 hex
        assert all(c in "0123456789abcdef" for c in h)


class TestEmbedStats:
    """Test embedding statistics queries."""

    @pytest.mark.skipif(
        not os.environ.get("CONVERSATIONS_DB_PASSWORD"),
        reason="CONVERSATIONS_DB_PASSWORD not set"
    )
    def test_stats_returns_dict(self):
        from v2.db import get_connection
        conn = get_connection()
        stats = get_embedding_stats(conn)
        conn.close()

        assert "total_messages" in stats
        assert "hashed" in stats
        assert "embedded" in stats
        assert "pending_unique_hashes" in stats
        assert all(isinstance(v, int) for v in stats.values())


class TestComputeHashes:
    """Test batch hash computation."""

    @pytest.mark.skipif(
        not os.environ.get("CONVERSATIONS_DB_PASSWORD"),
        reason="CONVERSATIONS_DB_PASSWORD not set"
    )
    def test_compute_hashes_idempotent(self):
        from v2.db import get_connection
        conn = get_connection()

        # Run twice — second run should find nothing to do
        n1 = compute_hashes_for_messages(conn)
        n2 = compute_hashes_for_messages(conn)

        assert n2 == 0  # everything already hashed
        conn.close()
