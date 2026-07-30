"""Subagent transcript identification.

Claude Code writes subagent transcripts INSIDE the parent session's directory:

    <project>/<parent-session-uuid>.jsonl                        <- parent transcript
    <project>/<parent-session-uuid>/subagents/agent-<id>.jsonl   <- subagent transcript
    <project>/<parent-session-uuid>/subagents/workflows/wf_<id>/agent-<id>.jsonl
    <project>/<parent-session-uuid>/subagents/workflows/wf_<id>/journal.jsonl

Every record inside a subagent transcript carries the PARENT's `sessionId`.
Ingesting one therefore used to resolve to the parent's conversation row and
overwrite it (the D1 data-destruction bug: db.upsert_conversation deletes the
existing row's messages before inserting the new file's).

The fix is identity, not filtering: a subagent transcript gets its own
conversation identity derived from its PATH, which is unique on disk and can
never collide with a parent session UUID because of the `sub:` prefix
(a UUID contains no ':').
"""

from pathlib import PurePosixPath

SUBAGENT_DIR = "subagents"
SUBAGENT_PREFIX = "sub:"


def is_subagent_path(rel_path) -> bool:
    """True if this relative path is a subagent transcript.

    Matches on a full path SEGMENT named 'subagents' so a project or file
    merely containing the substring cannot false-positive.
    """
    return SUBAGENT_DIR in PurePosixPath(str(rel_path)).parts


def subagent_identity(rel_path) -> tuple[str, str | None]:
    """Derive (session_id, parent_session_id) for a subagent transcript.

    Returns a session_id of the form:
        sub:<parent-session-uuid>:<path-suffix-after-subagents>

    The suffix keeps every path segment below 'subagents/', so sibling
    workflow directories that both contain e.g. 'journal.jsonl' stay distinct.

    >>> subagent_identity("proj/8f2a/subagents/agent-aa19.jsonl")
    ('sub:8f2a:agent-aa19', '8f2a')
    >>> subagent_identity("proj/8f2a/subagents/workflows/wf_12/journal.jsonl")
    ('sub:8f2a:workflows:wf_12:journal', '8f2a')
    """
    parts = PurePosixPath(str(rel_path)).parts
    if SUBAGENT_DIR not in parts:
        raise ValueError(f"Not a subagent path: {rel_path}")

    idx = parts.index(SUBAGENT_DIR)

    # The segment immediately before 'subagents' is the parent session dir.
    parent_session_id = parts[idx - 1] if idx > 0 else None

    suffix_parts = list(parts[idx + 1:])
    if not suffix_parts:
        raise ValueError(f"Subagent path has nothing below 'subagents/': {rel_path}")

    # Strip the .jsonl extension from the final segment.
    last = suffix_parts[-1]
    if last.endswith(".jsonl"):
        suffix_parts[-1] = last[: -len(".jsonl")]

    suffix = ":".join(suffix_parts)
    session_id = f"{SUBAGENT_PREFIX}{parent_session_id}:{suffix}"
    return session_id, parent_session_id


def is_subagent_session_id(session_id: str) -> bool:
    """True if a session_id was minted by subagent_identity()."""
    return bool(session_id) and str(session_id).startswith(SUBAGENT_PREFIX)
