"""Search memory tool."""

import time
from typing import Any, cast

from ..config import get_config
from ..context import db, mcp
from ..core.decay import calculate_score
from ..core.pagination import paginate_list, validate_pagination_params
from ..core.review import blend_search_results, get_memories_due_for_review
from ..core.search_common import is_pagination_requested, validate_search_params
from ..core.similarity import cosine_similarity, text_similarity
from ..core.text_utils import truncate_content
from ..performance import time_operation
from ..storage.models import SearchResult

# Optional dependency for embeddings
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Global model cache to avoid reloading on every request
_model_cache: dict[str, Any] = {}


def _get_embedding_model(model_name: str) -> "SentenceTransformer | None":
    """Get cached embedding model or create new one."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
        return None

    if model_name not in _model_cache:
        try:
            _model_cache[model_name] = SentenceTransformer(model_name)
        except Exception:
            return None

    return cast("SentenceTransformer", _model_cache[model_name])


def _generate_query_embedding(query: str) -> list[float] | None:
    """Generate embedding for search query."""
    config = get_config()
    if not config.enable_embeddings or not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    model = _get_embedding_model(config.embed_model)
    if model is None:
        return None

    try:
        embedding = model.encode(query, convert_to_numpy=True)
        return cast(list[float], embedding.tolist())
    except Exception:
        return None


@mcp.tool()
@time_operation("search_memory")
def search_memory(
    query: str | None = None,
    tags: list[str] | None = None,
    status: str | list[str] | None = None,
    top_k: int = 10,
    window_days: int | None = None,
    min_score: float | None = None,
    use_embeddings: bool = False,
    include_review_candidates: bool = True,
    page: int | None = None,
    page_size: int | None = None,
    preview_length: int | None = None,
) -> dict[str, Any]:
    """Search memories with filters and pagination.

    Args:
        query: Search text (max 50k chars).
        tags: Filter by tags (max 50).
        status: Filter by status ('active', 'promoted', 'archived' or list of these).
                Defaults to ['active', 'promoted'] if None.
        top_k: Max results (1-100).
        window_days: Recent memories only (1-3650 days).
        min_score: Min decay score (0.0-1.0).
        use_embeddings: Enable semantic search.
        include_review_candidates: Include review-due memories.
        page: Page number (default: 1).
        page_size: Results per page (10-100, default: 10).
        preview_length: Content chars (0-5000, default: 300).

    Returns:
        Dict with results list and pagination metadata.

    Raises:
        ValueError: Invalid parameters.
    """
    # Validate parameters using shared validation
    params = validate_search_params(
        query=query,
        tags=tags,
        status=status,
        limit=top_k,
        window_days=window_days,
        min_score=min_score,
        preview_length=preview_length,
        page=page,
        page_size=page_size,
        use_embeddings=use_embeddings,
    )

    # Check if pagination was explicitly requested
    pagination_requested = is_pagination_requested(page, page_size)

    now = int(time.time())

    memories = db.search_memories(
        tags=params.tags,
        status=params.status,
        window_days=params.window_days,
        limit=params.limit * 3,
    )

    query_embed = None
    if params.use_embeddings and params.query:
        config = get_config()
        if config.enable_embeddings:
            query_embed = _generate_query_embedding(params.query)

    results: list[SearchResult] = []
    for memory in memories:
        score = calculate_score(
            use_count=memory.use_count,
            last_used=memory.last_used,
            strength=memory.strength,
            now=now,
        )

        if params.min_score is not None and score < params.min_score:
            continue

        similarity = None
        if query_embed and memory.embed:
            # Semantic similarity using embeddings
            similarity = cosine_similarity(query_embed, memory.embed)

        relevance = 1.0
        if params.query and not params.use_embeddings:
            # Fallback: Use Jaccard similarity for better semantic matching
            # This matches the sophisticated fallback in clustering.py
            text_sim = text_similarity(params.query, memory.content)
            # Scale to 1.0-2.0 range (0.0 similarity = 1.0 relevance, 1.0 similarity = 2.0 relevance)
            relevance = 1.0 + text_sim

        final_score = score * relevance
        if similarity is not None:
            final_score = score * similarity

        results.append(SearchResult(memory=memory, score=final_score, similarity=similarity))

    results.sort(key=lambda r: r.score, reverse=True)

    # Natural spaced repetition: blend in review candidates
    final_memories = [r.memory for r in results[: params.limit]]

    if include_review_candidates and params.query:
        # Get memories for review queue matching search status
        all_active = db.search_memories(status=params.status, limit=10000)

        # Get memories due for review
        review_queue = get_memories_due_for_review(all_active, min_priority=0.3, limit=20)

        # Filter review candidates for relevance to query
        relevant_reviews = []
        for mem in review_queue:
            is_relevant = False

            # Check semantic similarity if embeddings available
            if query_embed and mem.embed:
                sim = cosine_similarity(query_embed, mem.embed)
                if sim and sim > 0.6:  # Somewhat relevant
                    is_relevant = True
            # Fallback: Use Jaccard similarity for text matching
            elif params.query:
                text_sim = text_similarity(params.query, mem.content)
                if text_sim > 0.3:  # Some token overlap
                    is_relevant = True

            if is_relevant:
                relevant_reviews.append(mem)

        # Blend primary results with review candidates
        if relevant_reviews:
            config = get_config()
            final_memories = blend_search_results(
                final_memories,
                relevant_reviews,
                blend_ratio=config.review_blend_ratio,
            )

    # Convert back to SearchResult format for final output
    final_results = []
    for mem in final_memories:
        # Find the original SearchResult if it exists
        original = next((r for r in results if r.memory.id == mem.id), None)
        if original:
            final_results.append(original)
        else:
            # It's a review candidate, calculate fresh score
            score = calculate_score(
                use_count=mem.use_count,
                last_used=mem.last_used,
                strength=mem.strength,
                now=now,
            )
            final_results.append(SearchResult(memory=mem, score=score, similarity=None))

    # Apply pagination only if requested
    if pagination_requested:
        # Validate and get non-None values
        valid_page, valid_page_size = validate_pagination_params(page, page_size)
        paginated = paginate_list(final_results, page=valid_page, page_size=valid_page_size)
        return {
            "success": True,
            "count": len(paginated.items),
            "results": [
                {
                    "id": r.memory.id,
                    "content": truncate_content(r.memory.content, params.preview_length),
                    "tags": r.memory.meta.tags,
                    "score": round(r.score, 4),
                    "similarity": round(r.similarity, 4) if r.similarity else None,
                    "use_count": r.memory.use_count,
                    "last_used": r.memory.last_used,
                    "age_days": round((now - r.memory.created_at) / 86400, 1),
                    "review_priority": round(r.memory.review_priority, 4)
                    if r.memory.review_priority > 0
                    else None,
                }
                for r in paginated.items
            ],
            "pagination": paginated.to_dict(),
        }
    else:
        # No pagination - return all results
        return {
            "success": True,
            "count": len(final_results),
            "results": [
                {
                    "id": r.memory.id,
                    "content": truncate_content(r.memory.content, params.preview_length),
                    "tags": r.memory.meta.tags,
                    "score": round(r.score, 4),
                    "similarity": round(r.similarity, 4) if r.similarity else None,
                    "use_count": r.memory.use_count,
                    "last_used": r.memory.last_used,
                    "age_days": round((now - r.memory.created_at) / 86400, 1),
                    "review_priority": round(r.memory.review_priority, 4)
                    if r.memory.review_priority > 0
                    else None,
                }
                for r in final_results
            ],
        }
