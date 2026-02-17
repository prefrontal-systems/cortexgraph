"""Maintenance CLI for JSONL storage.

Expose storage statistics, compaction, and embeddings backfill operations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .jsonl_storage import JSONLStorage

# Optional dependency for embeddings
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def cmd_stats(storage_path: Path | None) -> int:
    storage = JSONLStorage(storage_path=storage_path)
    storage.connect()
    stats = storage.get_storage_stats()
    print(json.dumps(stats, indent=2))
    return 0


def cmd_compact(storage_path: Path | None, *, quiet: bool = False) -> int:
    storage = JSONLStorage(storage_path=storage_path)
    storage.connect()
    before = storage.get_storage_stats()
    result = storage.compact()
    after = storage.get_storage_stats()
    if quiet:
        print(json.dumps({"result": result}, indent=2))
    else:
        print("Before:")
        print(json.dumps(before, indent=2))
        print("\nCompaction:")
        print(json.dumps(result, indent=2))
        print("\nAfter:")
        print(json.dumps(after, indent=2))
    return 0


def cmd_backfill_embeddings(
    storage_path: Path | None,
    *,
    model: str = "all-MiniLM-L6-v2",
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Backfill embeddings for memories that don't have them."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print(json.dumps({"error": "sentence-transformers not installed"}, indent=2))
        return 1

    storage = JSONLStorage(storage_path=storage_path)
    storage.connect()

    # Get all memories
    memories = storage.list_memories()

    # Filter to those without embeddings (or all if force=True)
    if force:
        targets = memories[:limit] if limit else memories
    else:
        targets = [m for m in memories if m.embed is None]
        if limit:
            targets = targets[:limit]

    if not targets:
        print(json.dumps({"message": "No memories need embeddings backfill"}, indent=2))
        return 0

    print(f"{'DRY RUN: ' if dry_run else ''}Processing {len(targets)} memories...")

    if dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "total_memories": len(memories),
                    "memories_without_embeddings": len([m for m in memories if m.embed is None]),
                    "would_process": len(targets),
                },
                indent=2,
            )
        )
        return 0

    # Load model
    if SentenceTransformer is None:
        print("ERROR: sentence-transformers not available")
        return 1

    print(f"Loading model: {model}...")
    embedding_model = SentenceTransformer(model)

    # Process memories
    processed = 0
    errors = 0

    for i, memory in enumerate(targets, 1):
        try:
            # Generate embedding
            embedding = embedding_model.encode(memory.content, convert_to_numpy=True)
            memory.embed = embedding.tolist()

            # Save back to storage
            storage.save_memory(memory)
            processed += 1

            if i % 10 == 0:
                print(f"  Processed {i}/{len(targets)}...")

        except Exception as e:
            errors += 1
            print(f"  Error processing memory {memory.id}: {e}")

    result = {
        "success": True,
        "processed": processed,
        "errors": errors,
        "model": model,
        "message": f"Backfilled embeddings for {processed} memories",
    }

    print(json.dumps(result, indent=2))
    return 0 if errors == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="JSONL storage maintenance")
    parser.add_argument(
        "--storage-path",
        type=Path,
        help="Override storage path (defaults to STM_STORAGE_PATH or config)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="Show storage stats")
    p_stats.set_defaults(func=lambda args: cmd_stats(args.storage_path))

    p_compact = sub.add_parser("compact", help="Compact JSONL files")
    p_compact.add_argument("--quiet", action="store_true", help="Only print compaction result")
    p_compact.set_defaults(func=lambda args: cmd_compact(args.storage_path, quiet=args.quiet))

    p_backfill = sub.add_parser(
        "backfill-embeddings", help="Generate embeddings for memories without them"
    )
    p_backfill.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model to use")
    p_backfill.add_argument("--limit", type=int, help="Maximum number of memories to process")
    p_backfill.add_argument(
        "--force", action="store_true", help="Regenerate embeddings even if they exist"
    )
    p_backfill.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without doing it"
    )
    p_backfill.set_defaults(
        func=lambda args: cmd_backfill_embeddings(
            args.storage_path,
            model=args.model,
            limit=args.limit,
            force=args.force,
            dry_run=args.dry_run,
        )
    )

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
