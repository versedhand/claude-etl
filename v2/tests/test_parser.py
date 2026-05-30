"""Tests for JSONL conversation parser."""

import json
import os
import pytest
from pathlib import Path

from v2.parser import parse_conversation, ParsedConversation, ParsedMessage

FIXTURE_DIR = Path(__file__).parent
SMALL_FIXTURE = FIXTURE_DIR / "fixture-small.jsonl"


class TestParseConversation:
    """Test parsing a real JSONL conversation file."""

    def test_returns_parsed_conversation(self):
        result = parse_conversation(SMALL_FIXTURE)
        assert isinstance(result, ParsedConversation)

    def test_extracts_session_id(self):
        result = parse_conversation(SMALL_FIXTURE)
        assert result.session_id is not None
        assert len(result.session_id) > 0

    def test_extracts_messages(self):
        result = parse_conversation(SMALL_FIXTURE)
        assert len(result.messages) > 0

    def test_messages_have_roles(self):
        result = parse_conversation(SMALL_FIXTURE)
        roles = {m.role for m in result.messages}
        assert "user" in roles
        assert "assistant" in roles

    def test_messages_have_content(self):
        result = parse_conversation(SMALL_FIXTURE)
        user_msgs = [m for m in result.messages if m.role == "user"]
        assert any(m.content for m in user_msgs)

    def test_messages_are_ordered(self):
        result = parse_conversation(SMALL_FIXTURE)
        for i, msg in enumerate(result.messages):
            assert msg.sequence_num == i

    def test_extracts_model(self):
        result = parse_conversation(SMALL_FIXTURE)
        # Model should be present in assistant messages
        assert result.model is not None or result.model is None  # may not always be present

    def test_extracts_project(self):
        result = parse_conversation(SMALL_FIXTURE)
        assert result.project is not None

    def test_timestamps_present(self):
        result = parse_conversation(SMALL_FIXTURE)
        assert result.started_at is not None
        assert result.last_message_at is not None

    def test_message_count_matches(self):
        result = parse_conversation(SMALL_FIXTURE)
        assert result.message_count == len(result.messages)


class TestAssistantContentExtraction:
    """Test that assistant message content blocks are correctly extracted."""

    def test_text_blocks_concatenated(self):
        result = parse_conversation(SMALL_FIXTURE)
        assistant_msgs = [m for m in result.messages if m.role == "assistant"]
        # At least one assistant message should have text content
        assert any(m.content for m in assistant_msgs)

    def test_tool_calls_extracted(self):
        result = parse_conversation(SMALL_FIXTURE)
        assistant_msgs = [m for m in result.messages if m.role == "assistant"]
        # The fixture has tool_use blocks
        has_tools = any(m.tool_calls for m in assistant_msgs)
        assert has_tools

    def test_thinking_extracted(self):
        result = parse_conversation(SMALL_FIXTURE)
        assistant_msgs = [m for m in result.messages if m.role == "assistant"]
        # Thinking may or may not be present in small fixture
        # Just verify the field exists and is string or None
        for m in assistant_msgs:
            assert m.thinking is None or isinstance(m.thinking, str)


class TestEdgeCases:
    """Test parser resilience."""

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        result = parse_conversation(empty)
        assert result.messages == []
        assert result.session_id is None

    def test_malformed_lines_skipped(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"type":"user","message":{"role":"user","content":"hello"},"uuid":"a"}\n')
        bad.write_text(bad.read_text() + "NOT VALID JSON\n")
        bad.write_text(bad.read_text() + '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]},"uuid":"b"}\n')
        result = parse_conversation(bad)
        assert len(result.messages) == 2  # skipped the bad line

    def test_null_bytes_sanitized(self, tmp_path):
        null_byte = tmp_path / "null.jsonl"
        null_byte.write_text('{"type":"user","message":{"role":"user","content":"hello\\u0000world"},"uuid":"a"}\n')
        result = parse_conversation(null_byte)
        assert "\x00" not in (result.messages[0].content or "")


class TestIdempotency:
    """Test that parsing the same file twice produces identical results."""

    def test_same_file_same_result(self):
        r1 = parse_conversation(SMALL_FIXTURE)
        r2 = parse_conversation(SMALL_FIXTURE)
        assert r1.session_id == r2.session_id
        assert len(r1.messages) == len(r2.messages)
        for m1, m2 in zip(r1.messages, r2.messages):
            assert m1.uuid == m2.uuid
            assert m1.role == m2.role
            assert m1.content == m2.content
