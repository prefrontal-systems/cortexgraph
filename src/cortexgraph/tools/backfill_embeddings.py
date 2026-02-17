"""Backfill embeddings tool - generate embeddings for memories that lack them."""

from typing import Any

from ..context import db, mcp
from ..security.validators import validate_positive_int

# Optional dependency for embeddings
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@mcp.tool()
def backfill_embeddings(
    model: str = "all-MiniLM-L6-v2",
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate embeddings for memories that lack them.

    Args:
        model: Model name (default: all-MiniLM-L6-v2).
        limit: Max memories to process (1-10k, None=all).
        force: Regenerate existing embeddings.
        dry_run: Preview only.

    Returns:
        Dict with: success, processed, errors, model, total_memories,
        memories_without_embeddings, message.

    Raises:
        ValueError: Invalid limit.
        ImportError: sentence-transformers not installed.
    """
    # Check if sentence-transformers is available
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return {
            "success": False,
            "error": "sentence-transformers not installed",
            "hint": "Install with: pip install sentence-transformers",
        }

    # Input validation
    if limit is not None:
        limit = validate_positive_int(limit, "limit", max_value=10000)

    # Get all memories
    memories = db.list_memories()
    total_count = len(memories)

    # Filter to those without embeddings (or all if force=True)
    if force:
        targets = memories[:limit] if limit else memories
        message_prefix = "Would regenerate" if dry_run else "Regenerating"
    else:
        targets = [m for m in memories if m.embed is None]
        if limit:
            targets = targets[:limit]
        message_prefix = "Would backfill" if dry_run else "Backfilled"

    without_embeddings = len([m for m in memories if m.embed is None])

    # Handle case where nothing needs processing
    if not targets:
        return {
            "success": True,
            "dry_run": dry_run,
            "processed": 0,
            "errors": 0,
            "model": model,
            "total_memories": total_count,
            "memories_without_embeddings": without_embeddings,
            "message": "No memories need embeddings backfill",
        }

    # Dry run - return preview
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "processed": 0,
            "errors": 0,
            "model": model,
            "total_memories": total_count,
            "memories_without_embeddings": without_embeddings,
            "would_process": len(targets),
            "message": f"Dry run: Would process {len(targets)} memories with model {model}",
        }

    # Load embedding model
    if SentenceTransformer is None:
        return {
            "success": False,
            "error": "sentence-transformers not available",
            "model": model,
        }

    try:
        embedding_model = SentenceTransformer(model)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load model {model}",
            "details": str(e),
            "hint": "Check model name or try default: all-MiniLM-L6-v2",
        }

    # Process memories
    processed = 0
    errors = 0
    error_details = []

    for memory in targets:
        try:
            # Generate embedding
            embedding = embedding_model.encode(memory.content, convert_to_numpy=True)
            memory.embed = embedding.tolist()

            # Save back to storage
            db.save_memory(memory)
            processed += 1

        except Exception as e:
            errors += 1
            error_details.append(
                {
                    "memory_id": memory.id,
                    "error": str(e),
                }
            )
            # Continue processing other memories
            continue

    # Build result
    result = {
        "success": errors == 0,
        "dry_run": False,
        "processed": processed,
        "errors": errors,
        "model": model,
        "total_memories": total_count,
        "memories_without_embeddings": without_embeddings - processed,  # Updated count
        "message": f"{message_prefix} embeddings for {processed} memories",
    }

    # Include error details if any errors occurred
    if errors > 0:
        result["error_details"] = error_details[:10]  # Limit to first 10 errors
        if errors > 10:
            result["additional_errors"] = errors - 10

    return result
