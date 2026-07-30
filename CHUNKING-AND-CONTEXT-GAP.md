# Known gap: long messages are not embedded, and terse messages are not findable

**Status: DEFERRED, deliberately. Not done in the 2026-07-30 D6 phase.**
**Owner: unassigned. This document exists so the gap is a decision, not an oversight.**

Two apparently separate defects are one architectural change, and that is the reason
neither was fixed here. Doing either one properly means abandoning the invariant the
whole embedding store is built on:

> **one content hash → one vector**

Both fixes replace it with **one message → N vectors**. That is a schema change, a
search-adapter change, and a mandatory eval re-baseline. It is not an embedder tweak, and
splitting it into two half-migrations would be worse than doing neither.

---

## Gap 1 — `MAX_CONTENT_LENGTH = 5000` skips long prose (F8)

**Measured 2026-07-30:** 17,695 distinct pending contents exceed 5,000 characters and are
permanently skipped. They are never embedded, so semantic search cannot return them.

**The brief described this set as "real prose where decisions get explained." That claim
was checked, and it is only partly true.** Measured composition of the 17,695:

| class | n | share |
|---|---|---|
| unclassified ("other") | 12,648 | 71.5% |
| web search results | 2,452 | 13.9% |
| browser tool responses (`take_snapshot`, `fill_form`) | 1,854 | 10.5% |
| page snapshots (`RootWebArea` accessibility trees) | 741 | 4.2% |

So **28.5% is unambiguous tool output.** Hand-reading 12 random rows from the remaining
"other" bucket found it mixed, and mostly machine as well: file reads with line-number
gutters (`1→`, `1\t`), agent task prompts (*"You are research worker 07…"*, *"You are a
calibrated evaluator…"*), `<task-notification>` blocks, JSON result payloads, pasted
Ansible config, and compaction summaries (*"This session is being continued from a
previous conversation…"*).

**But a real minority is genuine and valuable** — roughly 2–3 of the 12 sampled, i.e. an
estimated 2,000–3,000 messages overall. Examples actually seen:

- `[user 8,464]` *"next on the list is expenses. specifically recurring ones. i don't do a
  budget so these are necessary expenses…"* — a real Isaac decision conversation.
- `[user 9,287]` a podcast transcript on outsourcing.
- `[user 9,560]` a Night Watch session log written by an agent.

**Why the alpha-ratio filter does not catch the machine share:** it rejects on
*punctuation density*, and these are English-prose-shaped. Measured alpha ratios of the
tool output above run 0.72–0.91, far above the 0.50 threshold. A web-search result page
reads, statistically, like an essay.

**Consequence for whoever does this work: chunking alone would be a mistake.** Embedding
all 17,695 would spend most of the cost and most of the added ranking competition on
machine output. **Chunking must be paired with the D5 noise filter**, so the ~2,000–3,000
genuine long messages are the ones that get chunked. That ordering is the finding.

**Do not "fix" this by raising the cap.** `text-embedding-3-large` accepts 8,191 tokens, so
a higher cap would admit more documents, but a single vector averaged over a 12,000-char
document is a poor retrieval target: the specific paragraph that answers the query is
diluted by everything else in the message. Raising the cap trades a visible miss for an
invisible ranking loss, which is worse — the miss can at least be measured.

---

## Gap 2 — terse ratifications are eligible but not findable

`MIN_CONTENT_LENGTH = 20` **was removed** on 2026-07-30, because it meant Isaac's short
decisive turns were never embedded at all. Verified case: the user turn `"ok let's go"`
(11 chars, 2026-07-27) is the Finch Harbor brand ratification cited across the corpus as a
decided fact — it was `embedded = f`.

**Removing the floor makes those rows ELIGIBLE. It does not make them FINDABLE.** A vector
computed from the bare string `"ok let's go"` matches other short affirmations; it does not
match *"when did Isaac approve the placement brand name"*. The semantic content of a
ratification lives in the turn it is answering, not in the ratification itself.

This is the same shape as Gap 1 — the unit being embedded (the whole message, exactly once)
is the wrong unit. There it is too big; here it is too small.

---

## The single change that closes both

```sql
CREATE TABLE embedding_chunks (
    content_hash  text    NOT NULL,
    chunk_idx     int     NOT NULL,
    chunk_text    text    NOT NULL,   -- what was actually embedded
    embedding     halfvec NOT NULL,
    PRIMARY KEY (content_hash, chunk_idx)
);
```

- **Long messages** → overlapping windows (~2,000 chars, ~200 overlap so a paragraph is
  never split across a boundary without appearing whole in one window).
- **Short messages** → one chunk carrying a context prefix: the preceding turn, plus the
  message itself. The stored `chunk_text` differs from `messages.content`, which is why
  the column has to exist rather than being reconstructed at query time.
- **Search** joins `embedding_chunks` and takes the best-scoring chunk per message
  (`DISTINCT ON (m.id) … ORDER BY m.id, score`), instead of joining `embeddings` on
  `content_hash`.

### Costs, stated honestly

1. **Re-baseline is mandatory.** Chunking changes what wins for *every* query, not just
   the newly-covered ones — more candidate vectors per message shifts the ranking of
   messages that already retrieved fine. `eval/BASELINE-2026-07-30.md` and the 33-query set
   are what makes this measurable rather than a guess. Anyone doing this work must re-run
   the eval and report the delta per mode, including regressions.
2. **Context-prefixed chunks break pure content-addressing.** Today the same text appearing
   in two conversations shares one vector. A chunk that includes the *preceding* turn is
   conversation-specific, so dedup weakens and storage grows.
3. **Cost is not the obstacle.** ~17,695 long contents ≈ a few dollars at
   $0.13/1M tokens. The obstacle is that it touches the schema, the adapter and the
   measured baseline together.

---

## What is true today (2026-07-30)

| | state |
|---|---|
| Messages 1–5,000 chars, prose | embedded, backfilled, scheduled every 15 min |
| Messages > 5,000 chars | **never embedded** — 17,695 distinct contents |
| Messages < 20 chars | now embedded (floor removed); retrieval quality inherently weak |
| Machine dumps (logs/JSON) | correctly excluded by the alpha-ratio filter (2.4%) |
| thinking / tool_calls / tool_results | never embedded, per Isaac's 2026-07-30 ruling |

The watchdog's coverage check measures against the **eligible** denominator, i.e. it counts
the >5,000-char messages as legitimately skipped. **So a green coverage reading does not
mean this gap is closed** — it means the pipeline is doing what it currently claims to do.
That is the honest reading, and the reason this file exists.
