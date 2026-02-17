"""Save memory tool."""

import logging
import time
import uuid
from typing import Any, cast

from ..config import get_config
from ..context import db, mcp
from ..performance import time_operation
from ..security.secrets import detect_secrets, format_secret_warning, should_warn_about_secrets
from ..security.validators import (
    MAX_CONTENT_LENGTH,
    MAX_ENTITIES_COUNT,
    MAX_TAGS_COUNT,
    validate_entity,
    validate_list_length,
    validate_string_length,
    validate_tag,
)
from ..storage.models import Memory, MemoryMetadata

logger = logging.getLogger(__name__)

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


def _generate_embedding(content: str) -> list[float] | None:
    """Generate embedding for content if embeddings are enabled."""
    config = get_config()
    if not config.enable_embeddings or not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    model = _get_embedding_model(config.embed_model)
    if model is None:
        return None

    try:
        embedding = model.encode(content, convert_to_numpy=True)
        return cast(list[float], embedding.tolist())
    except Exception:
        return None


@mcp.tool()
@time_operation("save_memory")
def save_memory(
    content: str,
    tags: list[str] | None = None,
    entities: list[str] | None = None,
    source: str | None = None,
    context: str | None = None,
    meta: dict[str, Any] | None = None,
    strength: float | None = None,
) -> dict[str, Any]:
    """Save memory to short-term storage with auto-enrichment.

    Args:
        content: Memory content (max 50k chars).
        tags: Tags (max 50).
        entities: Named entities (max 100, auto-extracted if None).
        source: Source (max 500 chars).
        context: Context (max 1k chars).
        meta: Custom metadata dict.
        strength: Base strength (1.0-2.0, auto-calculated if None).

    Returns:
        Dict with: success, memory_id, message, has_embedding, enrichment_applied.

    Raises:
        ValueError: Invalid input.
    """
    # Input validation
    content = cast(
        str, validate_string_length(content, MAX_CONTENT_LENGTH, "content", allow_empty=False)
    )

    if tags is not None:
        tags = validate_list_length(tags, MAX_TAGS_COUNT, "tags")
        tags = [validate_tag(tag, f"tags[{i}]") for i, tag in enumerate(tags)]

    if entities is not None:
        entities = validate_list_length(entities, MAX_ENTITIES_COUNT, "entities")
        entities = [validate_entity(entity, f"entities[{i}]") for i, entity in enumerate(entities)]

    if source is not None:
        source = cast(str, validate_string_length(source, 500, "source", allow_none=True))

    if context is not None:
        context = cast(str, validate_string_length(context, 1000, "context", allow_none=True))

    # Auto-enrichment preprocessing (v0.6.0)
    config = get_config()
    enrichment_applied = False

    if config.enable_preprocessing:
        from ..preprocessing import EntityExtractor, ImportanceScorer, PhraseDetector

        # Initialize preprocessing components (cached at module level)
        phrase_detector = PhraseDetector()
        entity_extractor = EntityExtractor()
        importance_scorer = ImportanceScorer()

        # Detect importance signals
        phrase_signals = phrase_detector.detect(content)

        # Auto-extract entities if not provided
        if entities is None:
            entities = entity_extractor.extract(content)
            enrichment_applied = True

        # Auto-calculate strength if not provided
        if strength is None:
            strength = importance_scorer.score(
                content, entities=entities, importance_marker=phrase_signals["importance_marker"]
            )
            enrichment_applied = True
    else:
        # Default strength if preprocessing disabled
        if strength is None:
            strength = 1.0

    # Validate strength
    if strength is not None and (strength < 1.0 or strength > 2.0):
        raise ValueError("strength must be between 1.0 and 2.0")

    # Secrets detection (if enabled)
    config = get_config()
    if config.detect_secrets:
        matches = detect_secrets(content)
        if should_warn_about_secrets(matches):
            warning = format_secret_warning(matches)
            logger.warning(f"Secrets detected in memory content:\n{warning}")
            # Note: We still save the memory but warn the user

    # Create metadata
    metadata = MemoryMetadata(
        tags=tags or [],
        source=source,
        context=context,
        extra=meta or {},
    )

    # Generate ID and embedding
    memory_id = str(uuid.uuid4())
    embed = _generate_embedding(content)

    # Create memory
    now = int(time.time())
    memory = Memory(
        id=memory_id,
        content=content,
        meta=metadata,
        created_at=now,
        last_used=now,
        use_count=0,
        embed=embed,
        entities=entities or [],
        strength=strength if strength is not None else 1.0,
    )

    # Save to database
    db.save_memory(memory)

    return {
        "success": True,
        "memory_id": memory_id,
        "message": f"Memory saved with ID: {memory_id}",
        "has_embedding": embed is not None,
        "enrichment_applied": enrichment_applied,
        "auto_entities": len(entities or []) if enrichment_applied else 0,
        "calculated_strength": strength,
    }
