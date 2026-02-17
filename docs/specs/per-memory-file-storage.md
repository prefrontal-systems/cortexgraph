# Spec: Per-Memory File Storage

**Status:** Proposed
**Author:** Scot Campbell
**Date:** 2026-02-15
**Affects:** `storage/jsonl_storage.py`, `core/consolidation.py`, `agents/*`, `tools/*`
**Related Issues:** #95 (hybrid SQLite index), #107 (schema consolidation), #109 (dedup failures), #110 (manual archive/delete)

## Motivation

CortexGraph's STM uses an append-only JSONL file (`memories.jsonl`) as its storage backend. This is a single-writer data structure. When the vault syncs across multiple machines via git, two machines appending to the same file creates merge conflicts that git cannot resolve semantically. The current workaround — a newest-wins merge driver — loses writes from the losing side.

The deeper problem: the append-only log also makes Ebbinghaus pruning and hippocampal consolidation harder than they need to be. Both operations require mutating or removing entries from a stream, which means either rewriting the file (compaction) or accumulating tombstones. The data structure is fighting the biological model.

## Design Principle

**The filesystem is the index.** Each memory is a file. The set of files in the directory *is* the set of living memories. Creating a memory creates a file. Pruning deletes a file. Consolidation creates a new file and deletes the sources. The directory listing is always consistent — there is no divergence between "what the log says" and "what's actually alive."

This mirrors what LTM promotion already does (one markdown file per promoted memory) and aligns with the hybrid architecture proposed in #95: **source files are canonical, any SQLite/vector index is derived and rebuildable.**

## Storage Layout

```
{storage_path}/
├── memories/
│   ├── {uuid}.json          # One file per memory
│   ├── {uuid}.json
│   └── ...
├── relations/
│   ├── {uuid}.json          # One file per relation
│   └── ...
└── .meta.json               # Storage metadata (version, last compaction, machine-id)
```

### Memory File Format

Each `{uuid}.json` contains a single Memory object, identical to the current JSONL line format:

```json
{
  "id": "a1b2c3d4-...",
  "content": "STOPPER Protocol prevents computational impulsivity via 7-step...",
  "gist": "STOPPER: 7-step EF regulation protocol preventing computational impulsivity via Slow/Think/Observe/Plan/Prepare/Execute/Read cycle. Converges with DBT STOP.",
  "meta": {
    "tags": ["stopper", "efit", "executive-function"],
    "source": "claude-code-session",
    "context": "Discussion of regulatory frameworks",
    "extra": {}
  },
  "created_at": 1739548800,
  "last_used": 1739635200,
  "use_count": 3,
  "strength": 1.2,
  "status": "active",
  "promoted_at": null,
  "promoted_to": null,
  "embed": null,
  "entities": ["STOPPER", "DBT", "executive function"],
  "review_priority": 0.25,
  "last_review_at": 1739635200,
  "review_count": 2,
  "cross_domain_count": 1
}
```

### Relation File Format

Each relation in `relations/{uuid}.json`:

```json
{
  "id": "r1r2r3r4-...",
  "from_memory_id": "a1b2c3d4-...",
  "to_memory_id": "e5f6g7h8-...",
  "relation_type": "consolidated_from",
  "strength": 1.0,
  "created_at": 1739635200,
  "metadata": {
    "cluster_id": "clust-xxx",
    "cluster_cohesion": 0.91
  }
}
```

### `.meta.json`

```json
{
  "storage_version": 2,
  "created_at": 1739548800,
  "machine_id": "macbook-sc",
  "last_gc_at": null,
  "last_consolidation_at": null,
  "migrated_from": "jsonl-v1"
}
```

The `machine_id` is informational only — it records which machine last performed maintenance operations but has no effect on read/write behavior.

## Gist Field and Token-Efficient Recall

### The Problem

When CortexGraph injects memories into LLM prompts (enhance, recall, auto_recall), the `content` field is the token cost center. Loading 20 memories at 200 tokens each costs 4,000 tokens. Most of the time, the full verbatim content isn't needed — a compressed summary would suffice to determine relevance and provide context.

### Biological Model: Gist vs. Verbatim Memory Traces

This maps to a well-established distinction in memory research. The hippocampus stores detailed **verbatim traces** (episodic, context-rich, full fidelity). The neocortex stores **gist traces** (semantic, compressed, essential meaning). When you recall "STOPPER," you get the 2-second gist first; verbatim details reconstruct on demand.

CortexGraph currently stores only verbatim traces (`content`). Adding a `gist` field implements the dual-trace model at the individual memory level. (This complements #107 Schema Consolidation, which implements gist at the *collection* level.)

### New Field: `gist`

Add a `gist` field to the `Memory` model:

```python
class Memory(BaseModel):
    # ... existing fields ...
    content: str = Field(description="Full verbatim memory content")
    gist: str | None = Field(
        default=None,
        description="Token-compressed summary for LLM context injection (20-50 tokens)"
    )
```

The `gist` is a semantically compressed version of `content` — same meaning, 50-75% fewer tokens. It preserves entities, relationships, numbers, and domain-specific terms while stripping linguistic scaffolding.

### Generation via token-reducer

Use the [`token-reducer`](https://pypi.org/project/token-reducer/) library for offline, LLM-free gist generation:

```bash
pip install token-reducer
```

```python
from token_reducer import compress_text

def generate_gist(content: str) -> str:
    """Generate a token-compressed gist of memory content.

    Uses token-reducer's semantic compression pipeline:
    entity abstraction, proposition extraction, deduplication.
    No LLM calls — runs locally, fast, deterministic.
    """
    return compress_text(content, level="moderate")
```

Compression levels and their use cases:
- **Light** (5-15% reduction): Near-verbatim, minimal information loss. Not worth the complexity.
- **Moderate** (20-40% reduction): Good balance — preserves all key entities and relationships, strips filler. **Default for gist generation.**
- **Aggressive** (50-75% reduction): Maximum compression. Useful for schema-level summaries (#107) but may lose nuance for individual memories.

### When Gists Are Generated

1. **At save time:** When `save_memory()` is called and `gist` is None, auto-generate using token-reducer. This is the common path — gists are computed once and cached.

```python
def save_memory(self, memory: Memory) -> None:
    if memory.gist is None and memory.content:
        memory.gist = generate_gist(memory.content)
    # ... write to file
```

2. **At consolidation:** When memories are merged, the consolidated memory gets a new gist generated from the merged content.

3. **At split:** Each atomic memory from a split gets its own gist generated from its content.

4. **On demand:** A `regenerate_gist(memory_id)` utility for when content is updated or compression settings change.

5. **Migration:** Existing memories without gists get backfilled in a batch pass (similar to `backfill_embeddings`).

### How Gists Are Used

**Default: gist-first recall.** When memories are loaded into LLM context (enhance, recall, search results), inject the `gist` instead of `content`:

```python
def format_memory_for_context(memory: Memory) -> str:
    """Format a memory for LLM context injection."""
    text = memory.gist or memory.content  # Fallback to full content if no gist
    return f"[{memory.id[:8]}] {text} (tags: {', '.join(memory.meta.tags)})"
```

**Expand on demand.** If the LLM needs full detail for a specific memory, it can call `open_memories(memory_id)` to get the complete `content`. This is the "gist → verbatim" retrieval cascade — same as how human memory works.

### Token Savings

Concrete example with 20 memories loaded during recall:

| Mode | Avg tokens/memory | Total tokens | Savings |
|------|-------------------|-------------|---------|
| Full content | 200 | 4,000 | — |
| Moderate gist | 130 | 2,600 | 35% |
| Aggressive gist | 80 | 1,600 | 60% |

Over hundreds of recall/enhance operations per day, this materially reduces token consumption.

### Gist Quality Validation

Not all content compresses equally well. Short memories (<50 tokens) may not benefit from gist compression — the gist could be longer than the original. Guard against this:

```python
def generate_gist(content: str) -> str | None:
    """Generate gist, or return None if content is already concise."""
    if len(content.split()) < 30:  # ~40 tokens, already concise
        return None  # Use content directly, no gist needed
    gist = compress_text(content, level="moderate")
    # Don't save a gist that's barely shorter than the original
    if len(gist) > len(content) * 0.85:
        return None
    return gist
```

### Dependency Note

`token-reducer` is an optional dependency — CortexGraph should function without it. If not installed, gist generation is skipped and `gist` remains None. All code paths that read `gist` already fall back to `content`.

```toml
# pyproject.toml
[project.optional-dependencies]
gist = ["token-reducer>=0.1"]
```

## Operations Mapped to Filesystem

| Operation | Current (JSONL) | New (Per-File) |
|-----------|----------------|----------------|
| **save_memory** | Append line to `memories.jsonl` | Write `memories/{id}.json` |
| **get_memory** | Lookup in-memory dict | Read `memories/{id}.json` (or from in-memory cache) |
| **touch_memory** | Append updated line, old line becomes stale | Overwrite `memories/{id}.json` |
| **delete_memory** | Append `_deleted` marker | Delete `memories/{id}.json` |
| **search_memory** | Scan in-memory dict | Scan in-memory cache (loaded at startup) |
| **gc** | Append markers, then compact | Delete files below threshold |
| **consolidation** | Append new + markers for old, then compact | Write new file, delete source files |
| **compact** | Rewrite entire file | **No longer needed** — directory is always clean |
| **create_relation** | Append line to `relations.jsonl` | Write `relations/{id}.json` |

### Compaction Is Eliminated

The single biggest simplification: **compaction goes away entirely.** There are no tombstones to clean up, no stale entries to deduplicate, no file rewrites. Each file is the current state of its memory. The filesystem is always compact by definition.

## Git Sync Behavior

### Normal Operation (No Conflicts)

Machine A creates `memories/aaa.json`, machine B creates `memories/bbb.json`. Git merges trivially — different files, no conflict.

### Same Memory Modified on Two Machines

Machine A touches memory `ccc` (updates `last_used`), machine B also touches `ccc`. Git reports a conflict on `memories/ccc.json`. Resolution options:

1. **Latest-`last_used` wins** — a simple merge driver that parses both JSON objects and keeps the one with the more recent `last_used` timestamp. This is correct because `touch` is commutative and the later touch represents the more recent state.
2. **Manual resolution** — rare enough that manual merge is acceptable.

A custom git merge driver for `memories/*.json` and `relations/*.json`:

```bash
#!/bin/bash
# .git-merge-memory.sh
# For per-memory JSON files: keep the version with the later last_used timestamp
BASE="$1" LOCAL="$2" REMOTE="$3"

local_ts=$(python3 -c "import json; print(json.load(open('$LOCAL')).get('last_used', 0))" 2>/dev/null || echo 0)
remote_ts=$(python3 -c "import json; print(json.load(open('$REMOTE')).get('last_used', 0))" 2>/dev/null || echo 0)

if [ "$remote_ts" -gt "$local_ts" ]; then
    cp "$REMOTE" "$LOCAL"
fi
exit 0
```

In `.gitattributes`:

```
.cortexgraph/stm/memories/*.json merge=memory-latest
.cortexgraph/stm/relations/*.json merge=memory-latest
```

### Consolidation and GC

When consolidation or GC runs on one machine, it creates and deletes files. Git propagates these changes cleanly:
- New consolidated memory file → added
- Source memory files → deleted
- `consolidated_from` relation files → added

If the other machine modified a source memory that was consolidated on machine A, git will show a conflict (modify vs. delete). The merge driver can handle this by checking: if the file was deleted because it was consolidated (check for a `consolidated_from` relation pointing to it), accept the deletion. Otherwise, flag for manual review.

**Rule: Only run consolidation and GC on one designated machine** (the Mac, as source of truth). VPS boxes only create and touch memories — they don't run maintenance. This eliminates the consolidation-vs-modification conflict class entirely.

## Hippocampal Consolidation (Memory Merging)

The existing consolidation pipeline (`core/consolidation.py`, `agents/semantic_merge.py`, `agents/cluster_detector.py`) works with minimal changes. The core algorithm remains the same — detect clusters, merge content, create consolidated memory, archive/delete sources.

### What Changes

The `execute_consolidation()` function currently calls:
1. `storage.save_memory(consolidated_memory)` → now writes `memories/{new_id}.json`
2. `storage.create_relations_batch(relations)` → now writes `relations/{id}.json` for each
3. `storage.delete_memories_batch(original_ids)` → now deletes `memories/{id}.json` for each

**Atomicity:** The consolidation should be performed as a single git commit:

```
git add memories/{new_id}.json relations/{rel1}.json relations/{rel2}.json
git rm memories/{old1}.json memories/{old2}.json memories/{old3}.json
git commit -m "consolidate: merge 3 STOPPER memories into {new_id}"
```

This ensures the directory never shows an intermediate state where both the consolidated and source memories exist (or where sources are deleted but the consolidated version doesn't exist yet).

### Consolidation Provenance

The `consolidated_from` relations already track provenance. With per-file storage, git history adds another layer: `git log --follow memories/{new_id}.json` shows when it was created, and the commit message references the source IDs. `git log --diff-filter=D -- memories/{old_id}.json` shows when a source was consumed.

## Memory Splitting (New Capability)

Consolidation merges related memories into abstractions. **Splitting** is the inverse: decomposing a complex memory into atomic parts. This models **memory differentiation** — as understanding deepens, general knowledge becomes more specific.

### When to Split

A memory is a splitting candidate when:
1. **Content length exceeds threshold** (e.g., >500 chars) AND contains multiple distinct ideas
2. **Tag diversity is high** — many tags suggest the memory covers multiple domains
3. **Entity count is high** — multiple named entities suggest multiple topics
4. **Cross-domain usage is high** — the memory keeps appearing in different contexts, suggesting it's a composite
5. **Manual trigger** — user identifies a memory that should be split

### Split Operation

```python
def execute_split(
    memory: Memory,
    storage: Storage,
    split_contents: list[str],
    split_tags: list[list[str]] | None = None,
    split_entities: list[list[str]] | None = None,
) -> dict[str, Any]:
    """
    Split a complex memory into atomic parts.

    Args:
        memory: Source memory to split
        storage: Storage instance
        split_contents: Content for each new atomic memory
        split_tags: Optional per-memory tags (inherits from source if None)
        split_entities: Optional per-memory entities (inherits from source if None)

    Returns:
        Result dict with new memory IDs and metadata

    Process:
    1. Create N new Memory objects from split_contents
    2. Each inherits: source, context, strength (possibly reduced)
    3. Each gets: its own tags/entities (or inherits from parent)
    4. Create 'split_from' relations: new → original
    5. Archive or delete the original
    6. Commit atomically
    """
```

### Split Provenance

A new relation type `split_from` tracks lineage:

```json
{
  "from_memory_id": "new-atomic-1",
  "to_memory_id": "original-complex",
  "relation_type": "split_from",
  "metadata": {
    "split_index": 0,
    "total_splits": 3
  }
}
```

Add `split_from` to the valid relation types alongside `consolidated_from`, `related`, `causes`, `supports`, `contradicts`.

### Strength Inheritance

When a memory splits:
- Each atomic memory inherits `strength` from the parent (they're equally "known")
- `use_count` is inherited (the knowledge has been accessed that many times)
- `created_at` is inherited (the knowledge origin date)
- `last_used` is set to now (the split is a form of access)

### Filesystem Effect

Splitting memory `abc`:

```
# Before
memories/abc.json          (complex, multi-topic)

# After
memories/abc.json          (deleted or archived)
memories/split-1.json      (atomic: topic A)
memories/split-2.json      (atomic: topic B)
memories/split-3.json      (atomic: topic C)
relations/rel-1.json       (split-1 → abc, type: split_from)
relations/rel-2.json       (split-2 → abc, type: split_from)
relations/rel-3.json       (split-3 → abc, type: split_from)
```

Git commit: `split: decompose abc into 3 atomic memories (split-1, split-2, split-3)`

## Ebbinghaus Decay and Pruning

Per-file storage makes the decay lifecycle visible and simple.

### Pruning (GC)

The `gc` tool walks `memories/`, computes decay scores, and deletes files below `forget_threshold`:

```python
for filepath in memories_dir.iterdir():
    memory = load_memory(filepath)
    score = calculate_score(memory)
    if score < config.forget_threshold:
        filepath.unlink()
        # Also clean up orphaned relations
        cleanup_relations_for(memory.id)
```

Git shows exactly what decayed away: `git log --diff-filter=D -- memories/` lists all pruned memories with their commit timestamps.

### Natural Spaced Repetition

Unchanged from current implementation. `touch_memory` overwrites the file with updated `last_used`, `use_count`, and `review_priority`. The danger-zone calculation and review blending work identically — they operate on in-memory state loaded at startup.

## In-Memory Cache

The per-file backend still loads all memories into RAM at startup for fast queries (same as the current JSONL backend). The in-memory dict `_memories: dict[str, Memory]` and tag index remain unchanged.

### Startup Loading

```python
def connect(self) -> None:
    """Load all memory files into in-memory cache."""
    for filepath in self.memories_dir.iterdir():
        if filepath.suffix == '.json':
            memory = Memory.model_validate_json(filepath.read_text())
            self._memories[memory.id] = memory
    for filepath in self.relations_dir.iterdir():
        if filepath.suffix == '.json':
            relation = Relation.model_validate_json(filepath.read_text())
            self._relations[relation.id] = relation
    self._build_tag_index()
    self._connected = True
```

### Write-Through

All mutations write to both the in-memory cache and the filesystem:

```python
def save_memory(self, memory: Memory) -> None:
    self._memories[memory.id] = memory
    self._update_tag_index(memory)
    filepath = self.memories_dir / f"{memory.id}.json"
    filepath.write_text(memory.model_dump_json(indent=2))
```

The `indent=2` produces human-readable JSON — a per-file benefit over the compact JSONL format. Individual memory files are small enough that pretty-printing has no meaningful cost.

## Interaction with Existing Issues

### #95 — Hybrid SQLite Index

Per-file storage strengthens the hybrid architecture proposed in #95. The SQLite index (with sqlite-vec for vectors, FTS5 for text, recursive CTEs for graph traversal) becomes a derived index that rebuilds from the `memories/` and `relations/` directories:

```bash
cortexgraph rebuild-index
# Scans memories/*.json and relations/*.json
# Populates SQLite with embeddings, FTS, and graph edges
# Index can be .gitignored — it's derived
```

The per-file format also simplifies incremental indexing: on startup, compare file mtimes against the last-indexed timestamp to only re-index changed memories.

### #107 — Schema Consolidation (Cortical Memory Layer)

Schemas are a natural extension of the per-file model. Schema files live alongside memory files with a `"status": "schema"` field:

```
memories/
├── {uuid}.json          # Regular STM memories
├── {uuid}.json          # Schema (status: "schema", higher strength)
└── ...
```

Schema consolidation (LTM clusters → compressed schemas) produces new schema files via the same create-and-link pattern used by memory consolidation. The `consolidated_from` provenance chain tracks which LTM memories a schema distills.

### #108 — Schema Bundles

Bundle definitions are configuration, not memory state. They reference schema IDs and live in a separate config directory. No storage layer changes needed.

### #109 — Clustering/Dedup Failures

Per-file storage doesn't directly fix the clustering bug (likely an embedding issue), but it simplifies debugging: you can `diff` any two memory files to verify content similarity, and you can inspect individual files to check whether embeddings exist (`"embed": null` vs `"embed": [0.1, ...]`).

### #110 — Manual Archive/Delete by ID

Trivially solved with per-file storage. Archiving a memory means updating its `"status"` field in the file. Deleting means removing the file. No append-only log gymnastics, no tombstones.

```python
def archive_memory(self, memory_id: str) -> None:
    memory = self._memories[memory_id]
    memory.status = MemoryStatus.ARCHIVED
    self.save_memory(memory)  # Overwrites file with updated status

def delete_memory(self, memory_id: str) -> None:
    filepath = self.memories_dir / f"{memory_id}.json"
    filepath.unlink()
    del self._memories[memory_id]
    self._cleanup_relations_for(memory_id)
```

## Migration Path

### Phase 1: Add Per-File Storage Backend

Implement `PerFileStorage` as a new storage backend alongside the existing `JSONLStorage` and `SQLiteStorage`. All three share the same interface (`save_memory`, `get_memory`, `list_memories`, etc.).

Configuration:

```bash
CORTEXGRAPH_STORAGE_BACKEND=perfile    # New option
CORTEXGRAPH_STORAGE_PATH=~/Vaults/notes/.cortexgraph/stm
```

### Phase 2: Migration Tool

A one-time migration script reads `memories.jsonl` and `relations.jsonl`, then writes individual files:

```bash
cortexgraph migrate --from jsonl --to perfile
```

This:
1. Reads the current JSONL files (resolving duplicates and tombstones)
2. Writes one `memories/{id}.json` per living memory
3. Writes one `relations/{id}.json` per living relation
4. Creates `.meta.json` with migration metadata
5. Backs up the original JSONL files

### Phase 3: Make Per-File the Default

After validation, change the default backend from `jsonl` to `perfile`. The JSONL backend remains available for backward compatibility.

### Phase 4: Add Splitting

Implement the `execute_split()` function and expose it as an MCP tool:

```python
# MCP tool: split_memory
split_memory(
    memory_id: str,
    split_contents: list[str],     # Content for each atomic memory
    split_tags: list[list[str]] | None = None,
    split_entities: list[list[str]] | None = None,
)
```

### Phase 5: Gist Backfill

Backfill gists for all existing memories:

```bash
cortexgraph backfill-gists [--level moderate] [--dry-run]
```

This reads each `memories/{id}.json`, generates a gist via token-reducer if missing, and writes the file back. Similar pattern to `backfill_embeddings`. Memories under 30 words skip gist generation (already concise enough).

### Phase 6: Git Merge Driver

Ship the `memory-latest` merge driver and document the git configuration for multi-machine sync setups.

## Performance Considerations

### File Count

At CortexGraph's target scale (1,000–10,000 memories), the file count is well within what modern filesystems handle efficiently. ext4, APFS, and NTFS all support millions of files per directory without degradation.

For extremely large memory stores (>50K), a sharded directory layout could be introduced:

```
memories/
├── a/
│   ├── a1b2c3d4-....json
│   └── ab12cd34-....json
├── b/
│   └── b3c4d5e6-....json
└── ...
```

This is not needed initially and can be added later without changing the storage interface.

### Startup Time

Loading N individual files is slower than reading one JSONL file due to filesystem overhead (N open/read/close operations vs. 1). For 1,000 memories, the difference is negligible (<100ms). For 10,000 memories, it may add 0.5–1 second. This is acceptable for a startup-time cost.

If startup becomes a bottleneck, the SQLite index from #95 can serve as a fast-load cache: load from SQLite at startup, then verify/sync against files in the background.

### Write Latency

Individual file writes are faster than JSONL appends for single-memory operations (no seek-to-end, no shared file lock). Batch operations are slightly slower (N file writes vs. N appends to one file), but batches are rare in normal operation (only consolidation and migration).

## Testing Strategy

1. **Unit tests:** All existing `JSONLStorage` tests adapted for `PerFileStorage` — same interface, same behavior
2. **Migration tests:** Round-trip: JSONL → per-file → verify all memories and relations preserved
3. **Concurrency tests:** Two processes writing different memories simultaneously (should never conflict)
4. **Git merge tests:** Simulate multi-machine sync scenarios, verify merge driver behavior
5. **Consolidation tests:** Verify atomic create + delete behavior, provenance chain integrity
6. **Split tests:** Verify atomic split behavior, relation creation, strength inheritance

## Summary

| Aspect | Current (JSONL) | Proposed (Per-File) |
|--------|-----------------|---------------------|
| **Storage format** | 2 append-only files | N individual JSON files |
| **Git sync** | Conflict-prone (same file) | Conflict-rare (different files) |
| **Compaction** | Required (tombstones accumulate) | Eliminated (directory is always clean) |
| **Consolidation** | Append new + tombstone old | Create new file + delete old files |
| **Splitting** | Not implemented | Natural: create N files from 1 |
| **Pruning/GC** | Tombstone + compact | Delete file |
| **Human readability** | Lines in a large file | Individual pretty-printed JSON files |
| **Archive/delete by ID** | Append marker (issue #110) | Modify or delete file |
| **Debugging** | grep the JSONL | Open the individual file |
| **Merge driver** | Newest-wins (loses data) | Latest-`last_used` per file (no data loss) |

### New: Gist Field (Dual-Trace Memory)

| Aspect | Detail |
|--------|--------|
| **Field** | `gist: str \| None` on Memory model |
| **Purpose** | Token-compressed summary for LLM context injection |
| **Generation** | Offline via `token-reducer` library (no LLM calls) |
| **Compression** | 20-40% reduction at moderate level, 50-75% at aggressive |
| **Usage** | Loaded by default in enhance/recall; full content on demand |
| **Dependency** | Optional (`pip install token-reducer`) — graceful fallback to `content` |
| **Biological model** | Gist trace (neocortex) vs. verbatim trace (hippocampus) |

The per-memory file model aligns the storage layer with CortexGraph's biological model: memories are individual entities that are born, reinforced, consolidated, split, and eventually pruned — and each of those lifecycle events is a visible filesystem operation with full git history. The gist field adds the dual-trace distinction: fast, cheap semantic recall via gist; full fidelity retrieval via content on demand.
