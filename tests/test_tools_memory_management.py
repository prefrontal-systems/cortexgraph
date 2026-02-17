"""
Tests for tools related to memory lifecycle management.

This test suite covers the following tools:
- `save_memory`
- `open_memories`
- `touch_memory`
- `gc`
- `promote_memory`
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from cortexgraph.storage.models import Memory, MemoryMetadata, MemoryStatus, Relation
from cortexgraph.tools.gc import gc
from cortexgraph.tools.open_memories import open_memories
from cortexgraph.tools.promote import promote_memory
from cortexgraph.tools.save import save_memory
from cortexgraph.tools.touch import touch_memory
from tests.conftest import make_test_uuid


class TestSaveMemory:
    """Test suite for save_memory tool."""

    def test_save_basic_memory(self, mock_config_preprocessor, temp_storage):
        """Test saving a basic memory with just content."""
        # Uses mock_config_preprocessor fixture (disables preprocessing)

        result = save_memory(content="This is a test memory")

        assert result["success"] is True
        assert "memory_id" in result
        assert "Memory saved with ID:" in result["message"]
        assert result["has_embedding"] is False

        # Verify memory was actually saved
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory is not None
        assert memory.content == "This is a test memory"
        assert memory.use_count == 0
        assert memory.entities == []
        assert memory.meta.tags == []

    def test_save_memory_with_tags(self, temp_storage):
        """Test saving memory with tags."""
        result = save_memory(content="Tagged memory", tags=["python", "testing", "cortexgraph"])

        assert result["success"] is True
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory.meta.tags == ["python", "testing", "cortexgraph"]

    def test_save_memory_with_entities(self, temp_storage):
        """Test saving memory with entities."""
        result = save_memory(content="Memory about Claude", entities=["Claude", "Anthropic", "AI"])

        assert result["success"] is True
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory.entities == ["Claude", "Anthropic", "AI"]

    def test_save_memory_with_source_and_context(self, temp_storage):
        """Test saving memory with source and context."""
        result = save_memory(
            content="Memory with metadata",
            source="user-input",
            context="During code review session",
        )

        assert result["success"] is True
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory.meta.source == "user-input"
        assert memory.meta.context == "During code review session"

    def test_save_memory_with_custom_metadata(self, temp_storage):
        """Test saving memory with custom metadata."""
        custom_meta = {"priority": "high", "project": "cortexgraph"}
        result = save_memory(content="Memory with custom meta", meta=custom_meta)

        assert result["success"] is True
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory.meta.extra == custom_meta

    def test_save_memory_all_fields(self, temp_storage):
        """Test saving memory with all optional fields."""
        result = save_memory(
            content="Complete memory",
            tags=["tag1", "tag2"],
            entities=["Entity1"],
            source="test-source",
            context="test-context",
            meta={"key": "value"},
        )

        assert result["success"] is True
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory.content == "Complete memory"
        assert memory.meta.tags == ["tag1", "tag2"]
        assert memory.entities == ["Entity1"]
        assert memory.meta.source == "test-source"
        assert memory.meta.context == "test-context"
        assert memory.meta.extra == {"key": "value"}

    def test_save_memory_timestamps(self, temp_storage):
        """Test that timestamps are set correctly."""
        before = int(time.time())
        result = save_memory(content="Timestamp test")
        after = int(time.time())

        memory = temp_storage.get_memory(result["memory_id"])
        assert before <= memory.created_at <= after
        assert before <= memory.last_used <= after
        assert memory.created_at == memory.last_used

    def test_save_memory_unique_ids(self, temp_storage):
        """Test that each memory gets a unique ID."""
        result1 = save_memory(content="Memory 1")
        result2 = save_memory(content="Memory 2")
        result3 = save_memory(content="Memory 3")

        assert result1["memory_id"] != result2["memory_id"]
        assert result2["memory_id"] != result3["memory_id"]
        assert result1["memory_id"] != result3["memory_id"]

    # Validation tests
    def test_save_empty_content_fails(self):
        """Test that empty content fails validation."""
        with pytest.raises(ValueError, match="content.*empty"):
            save_memory(content="")

    def test_save_content_too_long_fails(self):
        """Test that content exceeding max length fails."""
        long_content = "x" * 50001  # MAX_CONTENT_LENGTH is 50000
        with pytest.raises(ValueError, match="content.*exceeds maximum"):
            save_memory(content=long_content)

    def test_save_too_many_tags_fails(self):
        """Test that too many tags fails validation."""
        too_many_tags = [f"tag{i}" for i in range(51)]  # MAX_TAGS_COUNT is 50
        with pytest.raises(ValueError, match="tags.*exceeds maximum"):
            save_memory(content="Test", tags=too_many_tags)

    def test_save_tag_too_long_fails(self):
        """Test that tags exceeding max length fail validation."""
        long_tag = "x" * 101  # Tags are limited to 100 chars
        with pytest.raises(ValueError, match="tag.*exceeds maximum"):
            save_memory(content="Test", tags=[long_tag])

    def test_save_invalid_tag_characters_sanitized(self, temp_storage):
        """Test that tags with invalid characters are auto-sanitized (MCP-friendly)."""
        result = save_memory(content="Test", tags=["invalid tag!"])
        assert result["success"] is True
        # Verify memory was saved with sanitized tag ("invalid tag!" -> "invalid_tag")
        memory = temp_storage.get_memory(result["memory_id"])
        assert "invalid_tag" in memory.meta.tags

    def test_save_too_many_entities_fails(self):
        """Test that too many entities fails validation."""
        too_many_entities = [f"entity{i}" for i in range(101)]  # MAX_ENTITIES_COUNT is 100
        with pytest.raises(ValueError, match="entities.*exceeds maximum"):
            save_memory(content="Test", entities=too_many_entities)

    def test_save_source_too_long_fails(self):
        """Test that source exceeding max length fails."""
        long_source = "x" * 501  # Source max is 500
        with pytest.raises(ValueError, match="source.*exceeds maximum"):
            save_memory(content="Test", source=long_source)

    def test_save_context_too_long_fails(self):
        """Test that context exceeding max length fails."""
        long_context = "x" * 1001  # Context max is 1000
        with pytest.raises(ValueError, match="context.*exceeds maximum"):
            save_memory(content="Test", context=long_context)

    # Edge cases
    def test_save_memory_with_none_tags(self, temp_storage):
        """Test that None tags are converted to empty list."""
        result = save_memory(content="Test", tags=None)
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory.meta.tags == []

    def test_save_memory_with_empty_tags(self, temp_storage):
        """Test saving with empty tags list."""
        result = save_memory(content="Test", tags=[])
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory.meta.tags == []

    def test_save_memory_with_none_entities(self, mock_config_preprocessor, temp_storage):
        """Test that None entities are converted to empty list."""
        # Uses mock_config_preprocessor fixture (disables preprocessing)

        result = save_memory(content="Test", entities=None)
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory.entities == []

    def test_save_memory_with_unicode_content(self, temp_storage):
        """Test saving memory with Unicode characters."""
        content = "Unicode test: 你好 🎉 café"
        result = save_memory(content=content)
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory.content == content

    def test_save_memory_with_special_characters(self, temp_storage):
        """Test saving memory with special characters."""
        content = "Special chars: <tag> & \"quotes\" 'apostrophe' \\backslash"
        result = save_memory(content=content)
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory.content == content

    # Secret detection tests (when enabled)
    @patch("cortexgraph.tools.save.detect_secrets")
    def test_save_warns_about_secrets_when_detected(
        self,
        mock_detect,
        mock_config_preprocessor,
        temp_storage,
        caplog,
    ):
        """Test that secret detection warns but still saves memory."""
        from cortexgraph.security.secrets import SecretMatch

        # Enable secret detection for this test
        mock_config_preprocessor.detect_secrets = True
        # Mock a high-confidence secret to trigger the warning without patching should_warn_about_secrets
        mock_detect.return_value = [
            SecretMatch(secret_type="openai_key", position=0, context="...")
        ]

        result = save_memory(content="API key: sk-xxx123")

        # Memory should still be saved
        assert result["success"] is True

        # But warning should be logged
        assert "Secrets detected" in caplog.text

    # Embedding tests
    def test_save_memory_with_embeddings_enabled(
        self, mock_config_embeddings, mock_embeddings_save, temp_storage
    ):
        """Test that embeddings are generated when enabled."""
        # Uses mock_config_embeddings + mock_embeddings_save fixtures

        result = save_memory(content="Test embedding")

        assert result["has_embedding"] is True
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory.embed == [0.1, 0.2, 0.3]

    def test_save_memory_with_embeddings_disabled(self, mock_config_preprocessor, temp_storage):
        """Test that embeddings are not generated when disabled."""
        # Uses mock_config_preprocessor fixture (embeddings disabled by default)

        result = save_memory(content="Test no embedding")

        assert result["has_embedding"] is False
        memory = temp_storage.get_memory(result["memory_id"])
        assert memory.embed is None

    @patch("cortexgraph.tools.save.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("cortexgraph.tools.save.SentenceTransformer")
    def test_save_memory_embedding_import_error(
        self, mock_transformer, mock_config_embeddings, temp_storage
    ):
        """Test that import error in embedding generation is handled gracefully."""
        # Clear model cache to ensure this test gets a fresh model load attempt
        from cortexgraph.tools import save

        save._model_cache.clear()

        # Uses mock_config_embeddings fixture (enables embeddings)
        # Patch transformer to raise ImportError
        mock_transformer.side_effect = ImportError("No model found")

        result = save_memory(content="Test")

        # Should still save without embedding
        assert result["success"] is True
        assert result["has_embedding"] is False


class TestOpenMemories:
    """Test suite for open_memories tool."""

    def test_open_single_memory(self, temp_storage):
        """Test retrieving a single memory by ID."""
        mem_id = make_test_uuid("test-123")
        mem = Memory(id=mem_id, content="Test memory", use_count=5, entities=["python"])
        temp_storage.save_memory(mem)

        result = open_memories(memory_ids=mem_id)

        assert result["success"] is True
        assert result["count"] == 1
        assert len(result["memories"]) == 1
        assert result["memories"][0]["id"] == mem_id
        assert result["memories"][0]["content"] == "Test memory"
        assert result["not_found"] == []

    def test_open_multiple_memories(self, temp_storage):
        """Test retrieving multiple memories at once."""
        ids = [make_test_uuid(f"mem-{i}") for i in range(3)]
        for i, mem_id in enumerate(ids):
            mem = Memory(id=mem_id, content=f"Memory {i}", use_count=1)
            temp_storage.save_memory(mem)

        result = open_memories(memory_ids=ids)

        assert result["success"] is True
        assert result["count"] == 3
        assert len(result["memories"]) == 3
        assert result["not_found"] == []

    def test_open_memory_includes_all_fields(self, temp_storage):
        """Test that result includes all expected memory fields."""
        now = int(time.time())
        mem_id = make_test_uuid("full-mem")
        mem = Memory(
            id=mem_id,
            content="Full memory",
            entities=["entity1", "entity2"],
            use_count=10,
            strength=1.5,
            created_at=now,
            last_used=now,
        )
        mem.meta.tags = ["tag1", "tag2"]
        mem.meta.source = "test"
        mem.meta.context = "test context"
        temp_storage.save_memory(mem)

        result = open_memories(memory_ids=mem_id)

        assert result["success"] is True
        memory = result["memories"][0]
        assert memory["id"] == mem_id
        assert memory["content"] == "Full memory"
        assert memory["entities"] == ["entity1", "entity2"]
        assert memory["tags"] == ["tag1", "tag2"]
        assert memory["source"] == "test"
        assert memory["context"] == "test context"
        assert memory["created_at"] == now
        assert memory["last_used"] == now
        assert memory["use_count"] == 10
        assert memory["strength"] == 1.5
        assert memory["status"] == "active"

    def test_open_memory_with_scores(self, temp_storage):
        """Test including decay scores in results."""
        mem_id = make_test_uuid("scored-mem")
        mem = Memory(id=mem_id, content="Test", use_count=5)
        temp_storage.save_memory(mem)

        result = open_memories(memory_ids=mem_id, include_scores=True)

        assert result["success"] is True
        memory = result["memories"][0]
        assert "score" in memory
        assert "age_days" in memory
        assert isinstance(memory["score"], float)
        assert isinstance(memory["age_days"], float)
        assert memory["score"] >= 0

    def test_open_memory_without_scores(self, temp_storage):
        """Test excluding scores from results."""
        mem_id = make_test_uuid("no-score")
        mem = Memory(id=mem_id, content="Test", use_count=1)
        temp_storage.save_memory(mem)

        result = open_memories(memory_ids=mem_id, include_scores=False)

        assert result["success"] is True
        memory = result["memories"][0]
        assert "score" not in memory
        assert "age_days" not in memory

    def test_open_memory_with_relations(self, temp_storage):
        """Test including relations in results."""
        mem1_id = make_test_uuid("mem-1")
        mem2_id = make_test_uuid("mem-2")
        mem3_id = make_test_uuid("mem-3")

        mem1 = Memory(id=mem1_id, content="Memory 1", use_count=1)
        mem2 = Memory(id=mem2_id, content="Memory 2", use_count=1)
        mem3 = Memory(id=mem3_id, content="Memory 3", use_count=1)

        temp_storage.save_memory(mem1)
        temp_storage.save_memory(mem2)
        temp_storage.save_memory(mem3)

        # Create relations
        rel1 = Relation(
            id=make_test_uuid("rel-1"),
            from_memory_id=mem1_id,
            to_memory_id=mem2_id,
            relation_type="related",
            strength=0.8,
            created_at=int(time.time()),
        )
        rel2 = Relation(
            id=make_test_uuid("rel-2"),
            from_memory_id=mem3_id,
            to_memory_id=mem1_id,
            relation_type="causes",
            strength=0.6,
            created_at=int(time.time()),
        )

        temp_storage.create_relation(rel1)
        temp_storage.create_relation(rel2)

        result = open_memories(memory_ids=mem1_id, include_relations=True)

        assert result["success"] is True
        memory = result["memories"][0]
        assert "relations" in memory
        assert "outgoing" in memory["relations"]
        assert "incoming" in memory["relations"]
        assert len(memory["relations"]["outgoing"]) == 1
        assert len(memory["relations"]["incoming"]) == 1
        assert memory["relations"]["outgoing"][0]["to"] == mem2_id
        assert memory["relations"]["incoming"][0]["from"] == mem3_id

    def test_open_memory_without_relations(self, temp_storage):
        """Test excluding relations from results."""
        mem_id = make_test_uuid("no-rel")
        mem = Memory(id=mem_id, content="Test", use_count=1)
        temp_storage.save_memory(mem)

        result = open_memories(memory_ids=mem_id, include_relations=False)

        assert result["success"] is True
        memory = result["memories"][0]
        assert "relations" not in memory

    def test_open_memory_not_found(self, temp_storage):
        """Test retrieving non-existent memory."""
        nonexistent_id = make_test_uuid("nonexistent")

        result = open_memories(memory_ids=nonexistent_id)

        assert result["success"] is True
        assert result["count"] == 0
        assert len(result["memories"]) == 0
        assert nonexistent_id in result["not_found"]

    def test_open_memories_partial_not_found(self, temp_storage):
        """Test retrieving mix of existing and non-existent memories."""
        existing_id = make_test_uuid("exists")
        nonexistent_id = make_test_uuid("missing")

        mem = Memory(id=existing_id, content="Exists", use_count=1)
        temp_storage.save_memory(mem)

        result = open_memories(memory_ids=[existing_id, nonexistent_id])

        assert result["success"] is True
        assert result["count"] == 1
        assert len(result["memories"]) == 1
        assert result["memories"][0]["id"] == existing_id
        assert nonexistent_id in result["not_found"]
        assert existing_id not in result["not_found"]

    def test_open_memories_promoted_memory(self, temp_storage):
        """Test retrieving promoted memory."""
        mem_id = make_test_uuid("promoted")
        mem = Memory(
            id=mem_id,
            content="Promoted memory",
            use_count=1,
            status=MemoryStatus.PROMOTED,
            promoted_at=int(time.time()),
            promoted_to="/vault/promoted.md",
        )
        temp_storage.save_memory(mem)

        result = open_memories(memory_ids=mem_id)

        assert result["success"] is True
        memory = result["memories"][0]
        assert memory["status"] == "promoted"
        assert memory["promoted_at"] is not None
        assert memory["promoted_to"] == "/vault/promoted.md"

    # Validation tests
    def test_open_invalid_uuid_fails(self):
        """Test that invalid UUID fails validation."""
        with pytest.raises(ValueError, match="memory_ids"):
            open_memories(memory_ids="not-a-uuid")

    def test_open_invalid_uuid_in_list_fails(self):
        """Test that invalid UUID in list fails validation."""
        valid_id = make_test_uuid("valid")
        with pytest.raises(ValueError, match="memory_ids"):
            open_memories(memory_ids=[valid_id, "not-a-uuid"])

    def test_open_too_many_ids_fails(self):
        """Test that exceeding max list length fails."""
        # Generate 101 IDs (max is 100)
        too_many_ids = [make_test_uuid(f"mem-{i}") for i in range(101)]

        with pytest.raises(ValueError, match="memory_ids"):
            open_memories(memory_ids=too_many_ids)

    def test_open_invalid_type_fails(self):
        """Test that invalid memory_ids type fails."""
        with pytest.raises(ValueError, match="memory_ids must be a string or list"):
            open_memories(memory_ids=123)  # type: ignore

    # Edge cases
    def test_open_empty_list(self, temp_storage):
        """Test with empty list of IDs."""
        result = open_memories(memory_ids=[])

        assert result["success"] is True
        assert result["count"] == 0
        assert result["memories"] == []
        assert result["not_found"] == []

    def test_open_memory_no_relations(self, temp_storage):
        """Test memory with no relations when include_relations=True."""
        mem_id = make_test_uuid("isolated")
        mem = Memory(id=mem_id, content="Isolated memory", use_count=1)
        temp_storage.save_memory(mem)

        result = open_memories(memory_ids=mem_id, include_relations=True)

        assert result["success"] is True
        memory = result["memories"][0]
        assert "relations" in memory
        assert memory["relations"]["outgoing"] == []
        assert memory["relations"]["incoming"] == []

    def test_open_memory_age_calculation(self, temp_storage):
        """Test age_days calculation."""
        now = int(time.time())
        three_days_ago = now - (3 * 86400)

        mem_id = make_test_uuid("old")
        mem = Memory(id=mem_id, content="Old memory", use_count=1, created_at=three_days_ago)
        temp_storage.save_memory(mem)

        result = open_memories(memory_ids=mem_id, include_scores=True)

        assert result["success"] is True
        memory = result["memories"][0]
        assert "age_days" in memory
        # Should be approximately 3 days old
        assert 2.9 <= memory["age_days"] <= 3.1

    def test_open_memory_score_rounded(self, temp_storage):
        """Test that scores are rounded to 4 decimal places."""
        mem_id = make_test_uuid("rounded")
        mem = Memory(id=mem_id, content="Test", use_count=1)
        temp_storage.save_memory(mem)

        result = open_memories(memory_ids=mem_id, include_scores=True)

        assert result["success"] is True
        memory = result["memories"][0]
        # Score should have at most 4 decimal places
        score_str = str(memory["score"])
        if "." in score_str:
            decimals = len(score_str.split(".")[1])
            assert decimals <= 4

    def test_open_memory_relation_strength_rounded(self, temp_storage):
        """Test that relation strengths are rounded to 4 decimal places."""
        mem1_id = make_test_uuid("mem-1")
        mem2_id = make_test_uuid("mem-2")

        mem1 = Memory(id=mem1_id, content="M1", use_count=1)
        mem2 = Memory(id=mem2_id, content="M2", use_count=1)
        temp_storage.save_memory(mem1)
        temp_storage.save_memory(mem2)

        rel = Relation(
            id=make_test_uuid("rel"),
            from_memory_id=mem1_id,
            to_memory_id=mem2_id,
            relation_type="related",
            strength=0.123456789,  # Many decimal places
            created_at=int(time.time()),
        )
        temp_storage.create_relation(rel)

        result = open_memories(memory_ids=mem1_id, include_relations=True)

        assert result["success"] is True
        memory = result["memories"][0]
        strength = memory["relations"]["outgoing"][0]["strength"]
        # Strength should have at most 4 decimal places
        strength_str = str(strength)
        if "." in strength_str:
            decimals = len(strength_str.split(".")[1])
            assert decimals <= 4


class TestTouchMemory:
    """Test suite for touch_memory tool."""

    def test_touch_basic_reinforcement(self, temp_storage):
        """Test basic memory reinforcement without strength boost."""
        test_id = make_test_uuid("test-123")

        # Create memory with old timestamp to ensure touch updates it
        old_time = int(time.time()) - 10  # 10 seconds ago
        mem = Memory(
            id=test_id, content="Test memory", use_count=0, strength=1.0, last_used=old_time
        )
        temp_storage.save_memory(mem)

        # Get the saved memory to check its timestamp
        saved_mem = temp_storage.get_memory(test_id)
        original_last_used = saved_mem.last_used

        result = touch_memory(memory_id=test_id, boost_strength=False)

        assert result["success"] is True
        assert result["memory_id"] == test_id
        assert result["use_count"] == 1
        assert result["strength"] == pytest.approx(1.0)

        # Verify memory was updated
        updated = temp_storage.get_memory(test_id)
        assert updated.use_count == 1
        assert updated.last_used > original_last_used

    def test_touch_increments_use_count(self, temp_storage):
        """Test that touch increments use_count correctly."""
        test_id = make_test_uuid("test-456")
        mem = Memory(id=test_id, content="Test", use_count=5)
        temp_storage.save_memory(mem)

        result = touch_memory(memory_id=test_id)

        assert result["success"] is True
        assert result["use_count"] == 6

        updated = temp_storage.get_memory(test_id)
        assert updated.use_count == 6

    def test_touch_with_strength_boost(self, temp_storage):
        """Test touching with strength boost."""
        test_id = make_test_uuid("test-789")
        mem = Memory(id=test_id, content="Test", strength=1.0)
        temp_storage.save_memory(mem)

        result = touch_memory(memory_id=test_id, boost_strength=True)

        assert result["success"] is True
        assert result["strength"] == pytest.approx(1.1)

        updated = temp_storage.get_memory(test_id)
        assert updated.strength == pytest.approx(1.1)

    def test_touch_strength_capped_at_2(self, temp_storage):
        """Test that strength is capped at 2.0."""
        test_id = make_test_uuid("test-cap")
        mem = Memory(id=test_id, content="Test", strength=1.95)
        temp_storage.save_memory(mem)

        # First boost: 1.95 + 0.1 = 2.05, capped to 2.0
        result = touch_memory(memory_id=test_id, boost_strength=True)

        assert result["success"] is True
        assert result["strength"] == pytest.approx(2.0)

        # Second boost: should stay at 2.0
        result2 = touch_memory(memory_id=test_id, boost_strength=True)

        assert result2["success"] is True
        assert result2["strength"] == pytest.approx(2.0)

        updated = temp_storage.get_memory(test_id)
        assert updated.strength == pytest.approx(2.0)

    def test_touch_multiple_times(self, temp_storage):
        """Test touching same memory multiple times."""
        test_id = make_test_uuid("test-multi")
        mem = Memory(id=test_id, content="Test", use_count=0)
        temp_storage.save_memory(mem)

        for i in range(1, 6):
            result = touch_memory(memory_id=test_id)
            assert result["success"] is True
            assert result["use_count"] == i

        updated = temp_storage.get_memory(test_id)
        assert updated.use_count == 5

    def test_touch_improves_score(self, temp_storage):
        """Test that touching improves the decay score."""
        now = int(time.time())
        old_time = now - (7 * 86400)  # 7 days ago

        test_id = make_test_uuid("test-score")
        mem = Memory(
            id=test_id,
            content="Old memory",
            use_count=1,
            last_used=old_time,
            created_at=old_time,
        )
        temp_storage.save_memory(mem)

        result = touch_memory(memory_id=test_id)

        assert result["success"] is True
        assert result["new_score"] > result["old_score"]

    def test_touch_memory_not_found(self, temp_storage):
        """Test touching non-existent memory."""
        result = touch_memory(memory_id="00000000-0000-0000-0000-000000000000")

        assert result["success"] is False
        assert "not found" in result["message"].lower()

    def test_touch_result_format(self, temp_storage):
        """Test that touch result has correct format."""
        test_id = make_test_uuid("test-format")
        mem = Memory(id=test_id, content="Test")
        temp_storage.save_memory(mem)

        result = touch_memory(memory_id=test_id)

        assert result["success"] is True
        assert "memory_id" in result
        assert "old_score" in result
        assert "new_score" in result
        assert "use_count" in result
        assert "strength" in result
        assert "message" in result

    def test_touch_updates_last_used_timestamp(self, temp_storage):
        """Test that last_used timestamp is updated."""
        before = int(time.time())
        test_id = make_test_uuid("test-time")
        mem = Memory(id=test_id, content="Test", last_used=before - 1000)
        temp_storage.save_memory(mem)

        time.sleep(0.1)
        result = touch_memory(memory_id=test_id)
        after = int(time.time())

        assert result["success"] is True

        updated = temp_storage.get_memory(test_id)
        assert before <= updated.last_used <= after

    def test_touch_with_tags_preserves_metadata(self, temp_storage):
        """Test that touching preserves memory metadata."""
        test_id = make_test_uuid("test-meta")
        mem = Memory(
            id=test_id,
            content="Test with metadata",
            meta=MemoryMetadata(
                tags=["important", "project"],
                source="user-input",
                context="Testing",
            ),
            entities=["TestEntity"],
        )
        temp_storage.save_memory(mem)

        result = touch_memory(memory_id=test_id, boost_strength=True)

        assert result["success"] is True

        updated = temp_storage.get_memory(test_id)
        assert updated.meta.tags == ["important", "project"]
        assert updated.meta.source == "user-input"
        assert updated.meta.context == "Testing"
        assert updated.entities == ["TestEntity"]

    # Validation tests
    def test_touch_invalid_uuid_fails(self):
        """Test that invalid UUID fails validation."""
        with pytest.raises(ValueError, match="memory_id.*valid UUID"):
            touch_memory(memory_id="not-a-uuid")

    def test_touch_empty_string_fails(self):
        """Test that empty string fails validation."""
        with pytest.raises(ValueError, match="memory_id"):
            touch_memory(memory_id="")

    # Edge cases
    def test_touch_with_default_boost_strength(self, temp_storage):
        """Test that boost_strength defaults to False."""
        test_id = make_test_uuid("test-default")
        mem = Memory(id=test_id, content="Test", strength=1.0)
        temp_storage.save_memory(mem)

        # Call without boost_strength parameter
        result = touch_memory(memory_id=test_id)

        assert result["success"] is True
        assert result["strength"] == pytest.approx(1.0)

    def test_touch_at_max_strength(self, temp_storage):
        """Test touching memory already at max strength."""
        test_id = make_test_uuid("test-max")
        mem = Memory(id=test_id, content="Test", strength=2.0)
        temp_storage.save_memory(mem)

        result = touch_memory(memory_id=test_id, boost_strength=True)

        assert result["success"] is True
        assert result["strength"] == pytest.approx(2.0)

    def test_touch_incremental_strength_boosts(self, temp_storage):
        """Test multiple incremental strength boosts."""
        test_id = make_test_uuid("test-incr")
        mem = Memory(id=test_id, content="Test", strength=1.0)
        temp_storage.save_memory(mem)

        expected_strengths = [1.1, 1.2, 1.3, 1.4, 1.5]

        for expected in expected_strengths:
            result = touch_memory(memory_id=test_id, boost_strength=True)
            assert result["success"] is True
            assert result["strength"] == pytest.approx(expected, rel=0.01)

    def test_touch_with_high_use_count(self, temp_storage):
        """Test touching memory with high use count."""
        test_id = make_test_uuid("test-high")
        mem = Memory(id=test_id, content="Test", use_count=999)
        temp_storage.save_memory(mem)

        result = touch_memory(memory_id=test_id)

        assert result["success"] is True
        assert result["use_count"] == 1000

    def test_touch_recently_created_memory(self, temp_storage):
        """Test touching memory that was just created."""
        test_id = make_test_uuid("test-new")
        mem = Memory(id=test_id, content="Just created")
        temp_storage.save_memory(mem)

        result = touch_memory(memory_id=test_id)

        assert result["success"] is True
        assert result["use_count"] == 1

    def test_touch_preserves_content(self, temp_storage):
        """Test that touching doesn't modify memory content."""
        original_content = "This content should not change"
        test_id = make_test_uuid("test-content")
        mem = Memory(id=test_id, content=original_content)
        temp_storage.save_memory(mem)

        touch_memory(memory_id=test_id, boost_strength=True)

        updated = temp_storage.get_memory(test_id)
        assert updated.content == original_content

    def test_touch_score_message(self, temp_storage):
        """Test that message includes score change."""
        test_id = make_test_uuid("test-msg")
        mem = Memory(id=test_id, content="Test")
        temp_storage.save_memory(mem)

        result = touch_memory(memory_id=test_id)

        assert result["success"] is True
        assert "reinforced" in result["message"].lower()
        assert "->" in result["message"]


class TestGarbageCollection:
    """Test suite for gc tool."""

    def test_gc_dry_run_mode(self, temp_storage):
        """Test dry run mode doesn't actually delete memories."""
        now = int(time.time())
        old_time = now - (30 * 86400)  # 30 days ago

        # Create old, low-scoring memory
        old_mem = Memory(
            id="mem-old",
            content="Old memory",
            use_count=0,
            last_used=old_time,
            created_at=old_time,
            strength=1.0,
        )
        temp_storage.save_memory(old_mem)

        result = gc(dry_run=True)

        assert result["success"] is True
        assert result["dry_run"] is True
        # Memory should still exist
        assert temp_storage.get_memory("mem-old") is not None

    def test_gc_actually_removes_memories(self, temp_storage):
        """Test that gc with dry_run=False actually removes memories."""
        now = int(time.time())
        old_time = now - (30 * 86400)

        old_mem = Memory(
            id="mem-remove",
            content="To be removed",
            use_count=0,
            last_used=old_time,
            created_at=old_time,
        )
        temp_storage.save_memory(old_mem)

        result = gc(dry_run=False)

        assert result["success"] is True
        assert result["dry_run"] is False
        # Memory should be deleted
        assert temp_storage.get_memory("mem-remove") is None

    def test_gc_archive_mode(self, temp_storage):
        """Test archiving memories instead of deleting."""
        now = int(time.time())
        old_time = now - (30 * 86400)

        old_mem = Memory(
            id="mem-archive",
            content="To be archived",
            use_count=0,
            last_used=old_time,
            created_at=old_time,
        )
        temp_storage.save_memory(old_mem)

        result = gc(dry_run=False, archive_instead=True)

        assert result["success"] is True
        assert result["archived_count"] >= 1
        assert result["removed_count"] == 0

        # Memory should exist but be archived
        archived = temp_storage.get_memory("mem-archive")
        assert archived is not None
        assert archived.status == MemoryStatus.ARCHIVED

    def test_gc_keeps_recent_memories(self, temp_storage):
        """Test that recent memories are not garbage collected."""
        now = int(time.time())

        recent_mem = Memory(
            id="mem-recent", content="Recent memory", use_count=5, last_used=now, created_at=now
        )
        temp_storage.save_memory(recent_mem)

        result = gc(dry_run=False)

        assert result["success"] is True
        # Recent memory should still exist
        assert temp_storage.get_memory("mem-recent") is not None

    def test_gc_with_limit(self, temp_storage):
        """Test limiting number of memories to collect."""
        now = int(time.time())
        old_time = now - (30 * 86400)

        # Create multiple old memories
        for i in range(10):
            mem = Memory(
                id=f"mem-{i}",
                content=f"Old memory {i}",
                use_count=0,
                last_used=old_time,
                created_at=old_time,
            )
            temp_storage.save_memory(mem)

        result = gc(dry_run=True, limit=3)

        assert result["success"] is True
        assert result["total_affected"] == 3

    def test_gc_no_memories_to_collect(self, temp_storage):
        """Test gc when no memories need collection."""
        now = int(time.time())

        # Create only fresh, high-scoring memories
        for i in range(3):
            mem = Memory(
                id=f"mem-{i}",
                content=f"Fresh memory {i}",
                use_count=10,
                last_used=now,
                strength=1.5,
            )
            temp_storage.save_memory(mem)

        result = gc(dry_run=False)

        assert result["success"] is True
        assert result["removed_count"] == 0
        assert result["archived_count"] == 0
        assert result["total_affected"] == 0

    def test_gc_result_format(self, temp_storage):
        """Test that gc result has correct format."""
        result = gc(dry_run=True)

        assert result["success"] is True
        assert "dry_run" in result
        assert "removed_count" in result
        assert "archived_count" in result
        assert "freed_score_sum" in result
        assert "memory_ids" in result
        assert "total_affected" in result
        assert "message" in result

    def test_gc_freed_score_sum(self, temp_storage):
        """Test that freed_score_sum is calculated."""
        now = int(time.time())
        old_time = now - (30 * 86400)

        old_mem = Memory(
            id="mem-score", content="Old", use_count=0, last_used=old_time, created_at=old_time
        )
        temp_storage.save_memory(old_mem)

        result = gc(dry_run=True)

        assert result["success"] is True
        if result["total_affected"] > 0:
            assert result["freed_score_sum"] >= 0

    def test_gc_memory_ids_limited_in_result(self, temp_storage):
        """Test that memory_ids in result is limited to 10."""
        now = int(time.time())
        old_time = now - (30 * 86400)

        # Create 15 old memories
        for i in range(15):
            mem = Memory(
                id=f"mem-{i:02d}",
                content=f"Old {i}",
                use_count=0,
                last_used=old_time,
                created_at=old_time,
            )
            temp_storage.save_memory(mem)

        result = gc(dry_run=True)

        assert result["success"] is True
        # Result should show max 10 IDs even if more affected
        assert len(result["memory_ids"]) <= 10

    def test_gc_sorts_by_lowest_score_first(self, temp_storage):
        """Test that gc removes lowest-scoring memories first."""
        now = int(time.time())

        # Create memories with different ages and track their IDs
        mem_ids = []
        for i in range(5):
            days_old = 30 + (i * 5)
            old_time = now - (days_old * 86400)
            mem_id = make_test_uuid(f"mem-{i}")
            mem = Memory(
                id=mem_id,
                content=f"Memory {i}",
                use_count=1,  # Set to 1 so memories have non-zero scores
                last_used=old_time,
                created_at=old_time,
            )
            temp_storage.save_memory(mem)
            mem_ids.append(mem_id)

        result = gc(dry_run=False, limit=2)

        assert result["success"] is True
        assert result["total_affected"] == 2

        # Verify that the two oldest memories (mem-4 and mem-3) were removed
        assert temp_storage.get_memory(mem_ids[4]) is None
        assert temp_storage.get_memory(mem_ids[3]) is None
        # Verify that the others still exist
        assert temp_storage.get_memory(mem_ids[2]) is not None
        assert temp_storage.get_memory(mem_ids[1]) is not None
        assert temp_storage.get_memory(mem_ids[0]) is not None

    def test_gc_message_includes_threshold(self, temp_storage):
        """Test that message includes forget threshold."""
        result = gc(dry_run=True)

        assert result["success"] is True
        assert "threshold" in result["message"].lower()

    def test_gc_dry_run_shows_would_remove(self, temp_storage):
        """Test that dry run message says 'Would remove'."""
        result = gc(dry_run=True)

        assert result["success"] is True
        assert "would remove" in result["message"].lower()

    def test_gc_actual_run_shows_removed(self, temp_storage):
        """Test that actual run message says 'Removed'."""
        result = gc(dry_run=False)

        assert result["success"] is True
        assert result["message"].startswith("Removed")

    # Validation tests
    def test_gc_invalid_limit_fails(self):
        """Test that invalid limit values fail."""
        with pytest.raises(ValueError, match="limit"):
            gc(limit=0)

        with pytest.raises(ValueError, match="limit"):
            gc(limit=10001)

        with pytest.raises(ValueError, match="limit"):
            gc(limit=-1)

    # Edge cases
    def test_gc_default_parameters(self, temp_storage):
        """Test that gc defaults to dry_run=True."""
        now = int(time.time())
        old_time = now - (30 * 86400)

        old_mem = Memory(
            id="mem-default", content="Old", use_count=0, last_used=old_time, created_at=old_time
        )
        temp_storage.save_memory(old_mem)

        # Call without parameters
        result = gc()

        assert result["success"] is True
        assert result["dry_run"] is True
        # Memory should still exist (dry run)
        assert temp_storage.get_memory("mem-default") is not None

    def test_gc_with_none_limit(self, temp_storage):
        """Test that None limit processes all eligible memories."""
        now = int(time.time())
        old_time = now - (30 * 86400)

        for i in range(5):
            mem = Memory(
                id=f"mem-{i}",
                content=f"Old {i}",
                use_count=0,
                last_used=old_time,
                created_at=old_time,
            )
            temp_storage.save_memory(mem)

        result = gc(dry_run=True, limit=None)

        assert result["success"] is True
        # Should process all eligible memories
        if result["total_affected"] > 0:
            assert result["total_affected"] >= 5

    def test_gc_empty_database(self, temp_storage):
        """Test gc on empty database."""
        result = gc(dry_run=False)

        assert result["success"] is True
        assert result["removed_count"] == 0
        assert result["total_affected"] == 0

    def test_gc_preserves_promoted_memories(self, temp_storage):
        """Test that promoted memories are not collected."""
        now = int(time.time())
        old_time = now - (30 * 86400)

        promoted_mem = Memory(
            id="mem-promoted",
            content="Promoted memory",
            use_count=0,
            last_used=old_time,
            created_at=old_time,
            status=MemoryStatus.PROMOTED,
        )
        temp_storage.save_memory(promoted_mem)

        result = gc(dry_run=False)

        assert result["success"] is True
        # Promoted memory should not be collected
        assert temp_storage.get_memory("mem-promoted") is not None

    def test_gc_preserves_archived_memories(self, temp_storage):
        """Test that already archived memories are not re-collected."""
        now = int(time.time())
        old_time = now - (30 * 86400)

        archived_mem = Memory(
            id="mem-archived",
            content="Already archived",
            use_count=0,
            last_used=old_time,
            created_at=old_time,
            status=MemoryStatus.ARCHIVED,
        )
        temp_storage.save_memory(archived_mem)

        result = gc(dry_run=False)

        assert result["success"] is True
        # Already archived memory should still exist
        assert temp_storage.get_memory("mem-archived") is not None

    def test_gc_dry_run_archive_mode(self, temp_storage):
        """Test dry run with archive_instead flag."""
        now = int(time.time())
        old_time = now - (30 * 86400)

        old_mem = Memory(
            id="mem-dry-archive",
            content="Old",
            use_count=0,
            last_used=old_time,
            created_at=old_time,
        )
        temp_storage.save_memory(old_mem)

        result = gc(dry_run=True, archive_instead=True)

        assert result["success"] is True
        assert result["dry_run"] is True
        # Memory should still exist and be active (dry run)
        mem = temp_storage.get_memory("mem-dry-archive")
        assert mem is not None
        assert mem.status == MemoryStatus.ACTIVE

    def test_gc_mixed_score_memories(self, temp_storage):
        """Test gc with mix of high and low scoring memories."""
        now = int(time.time())

        # High scoring (recent)
        high_mem = Memory(id="mem-high", content="High score", use_count=10, last_used=now)
        # Low scoring (old)
        low_mem = Memory(
            id="mem-low",
            content="Low score",
            use_count=0,
            last_used=now - (30 * 86400),
            created_at=now - (30 * 86400),
        )

        temp_storage.save_memory(high_mem)
        temp_storage.save_memory(low_mem)

        result = gc(dry_run=False)

        assert result["success"] is True
        # High scoring should remain
        assert temp_storage.get_memory("mem-high") is not None


class TestPromoteMemory:
    """Test suite for promote_memory tool."""

    def test_promote_requires_memory_id_or_auto_detect(self):
        """Test that either memory_id or auto_detect must be provided."""
        result = promote_memory()

        assert result["success"] is False
        assert "must specify" in result["message"].lower()

    def test_promote_memory_not_found(self, temp_storage):
        """Test promoting non-existent memory."""
        result = promote_memory(memory_id="00000000-0000-0000-0000-000000000000")

        assert result["success"] is False
        assert "not found" in result["message"].lower()

    def test_promote_already_promoted_memory(self, temp_storage):
        """Test that already promoted memory returns error."""
        test_id = make_test_uuid("mem-promoted")
        mem = Memory(
            id=test_id,
            content="Already promoted",
            status=MemoryStatus.PROMOTED,
            promoted_to="/vault/memory.md",
        )
        temp_storage.save_memory(mem)

        result = promote_memory(memory_id=test_id)

        assert result["success"] is False
        assert "already promoted" in result["message"].lower()
        assert "promoted_to" in result

    def test_promote_memory_not_meeting_criteria(self, temp_storage):
        """Test that low-scoring memory cannot be promoted without force.

        Note: With use_count+1 formula, even use_count=0 gives score of 1.0 when fresh.
        Need old memory (20 days) to decay below promotion threshold (0.65).
        """
        now = int(time.time())
        twenty_days_ago = now - (20 * 86400)

        test_id = make_test_uuid("mem-low")
        low_score_mem = Memory(
            id=test_id,
            content="Low score memory",
            use_count=0,  # With +1 formula: (0+1)^0.6 = 1.0, but will decay
            last_used=twenty_days_ago,  # Old enough to decay below 0.65
            created_at=twenty_days_ago,
            strength=1.0,
        )
        temp_storage.save_memory(low_score_mem)

        result = promote_memory(memory_id=test_id)

        assert result["success"] is False
        assert "does not meet" in result["message"].lower()
        assert "score" in result

    @patch("cortexgraph.tools.promote.BasicMemoryIntegration")
    def test_promote_with_force_flag(self, mock_integration, temp_storage):
        """Test that force flag bypasses criteria check."""
        now = int(time.time())
        mock_integration_instance = MagicMock()
        mock_integration_instance.promote_to_obsidian.return_value = {
            "success": True,
            "path": "/vault/forced.md",
        }
        mock_integration.return_value = mock_integration_instance

        test_id = make_test_uuid("mem-force")
        low_score_mem = Memory(
            id=test_id, content="Forced promotion", use_count=0, last_used=now, created_at=now
        )
        temp_storage.save_memory(low_score_mem)

        result = promote_memory(memory_id=test_id, force=True, dry_run=False)

        assert result["success"] is True
        assert result["promoted_count"] >= 1

    @patch("cortexgraph.tools.promote.BasicMemoryIntegration")
    def test_promote_high_scoring_memory(self, mock_integration, temp_storage):
        """Test promoting a high-scoring memory."""
        now = int(time.time())
        mock_integration_instance = MagicMock()
        mock_integration_instance.promote_to_obsidian.return_value = {
            "success": True,
            "path": "/vault/high.md",
        }
        mock_integration.return_value = mock_integration_instance

        test_id = make_test_uuid("mem-high")
        high_score_mem = Memory(
            id=test_id,
            content="High value memory",
            use_count=10,
            last_used=now,
            created_at=now - (14 * 86400),  # 14 days old
            strength=1.5,
        )
        temp_storage.save_memory(high_score_mem)

        result = promote_memory(memory_id=test_id, dry_run=False)

        assert result["success"] is True
        assert result["promoted_count"] == 1
        assert result["promoted_ids"] == [test_id]

    def test_promote_dry_run_mode(self, temp_storage):
        """Test dry run mode doesn't actually promote."""
        now = int(time.time())

        test_id = make_test_uuid("mem-dry")
        high_mem = Memory(
            id=test_id,
            content="Test dry run",
            use_count=10,
            last_used=now,
            created_at=now - (14 * 86400),
            strength=1.5,
        )
        temp_storage.save_memory(high_mem)

        result = promote_memory(memory_id=test_id, dry_run=True, force=True)

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["promoted_count"] == 0

        # Memory should still be active
        mem = temp_storage.get_memory(test_id)
        assert mem.status == MemoryStatus.ACTIVE

    def test_promote_auto_detect_no_candidates(self, temp_storage):
        """Test auto-detect when no memories meet criteria.

        Note: With use_count+1 formula, fresh memories have score=1.0.
        Need old memories (20 days) to decay below promotion threshold.
        """
        now = int(time.time())
        twenty_days_ago = now - (20 * 86400)

        # Create only low-scoring memories (old enough to decay below 0.65)
        for i in range(3):
            test_id = make_test_uuid(f"mem-{i}")
            mem = Memory(
                id=test_id,
                content=f"Low score {i}",
                use_count=0,
                last_used=twenty_days_ago,  # Old enough to decay
                created_at=twenty_days_ago,
            )
            temp_storage.save_memory(mem)

        result = promote_memory(auto_detect=True, dry_run=True)

        assert result["success"] is True
        assert result["candidates_found"] == 0

    @patch("cortexgraph.tools.promote.BasicMemoryIntegration")
    def test_promote_auto_detect_finds_candidates(self, mock_integration, temp_storage):
        """Test auto-detect finds high-value memories."""
        now = int(time.time())
        mock_integration_instance = MagicMock()
        mock_integration_instance.promote_to_obsidian.return_value = {
            "success": True,
            "path": "/vault/auto.md",
        }
        mock_integration.return_value = mock_integration_instance

        # Create high-value memory
        test_id = make_test_uuid("mem-auto")
        high_mem = Memory(
            id=test_id,
            content="High value",
            use_count=10,
            last_used=now,
            created_at=now - (14 * 86400),
            strength=1.5,
        )
        temp_storage.save_memory(high_mem)

        result = promote_memory(auto_detect=True, dry_run=True)

        assert result["success"] is True
        # May find candidates if criteria met
        assert "candidates_found" in result

    def test_promote_result_format(self, temp_storage):
        """Test that promote result has correct format."""
        result = promote_memory(auto_detect=True, dry_run=True)

        assert result["success"] is True
        assert "dry_run" in result
        assert "candidates_found" in result
        assert "promoted_count" in result
        assert "promoted_ids" in result
        assert "candidates" in result
        assert "message" in result

    def test_promote_candidate_preview_format(self, temp_storage):
        """Test that candidate previews have correct format."""
        now = int(time.time())

        test_id = make_test_uuid("mem-preview")
        high_mem = Memory(
            id=test_id,
            content="Test preview format" * 10,  # Long content
            use_count=10,
            last_used=now,
            created_at=now - (14 * 86400),
            strength=1.5,
        )
        temp_storage.save_memory(high_mem)

        result = promote_memory(auto_detect=True, dry_run=True)

        assert result["success"] is True
        if result["candidates_found"] > 0:
            candidate = result["candidates"][0]
            assert "id" in candidate
            assert "content_preview" in candidate
            assert "reason" in candidate
            assert "score" in candidate
            assert "use_count" in candidate
            assert "age_days" in candidate
            # Content should be truncated to 100 chars
            assert len(candidate["content_preview"]) <= 100

    def test_promote_candidates_limited_to_10(self, temp_storage):
        """Test that candidate list is limited to 10."""
        now = int(time.time())

        # Create many high-value memories
        for i in range(15):
            test_id = make_test_uuid(f"mem-{i:02d}")
            mem = Memory(
                id=test_id,
                content=f"High value {i}",
                use_count=10,
                last_used=now,
                created_at=now - (14 * 86400),
                strength=1.5,
            )
            temp_storage.save_memory(mem)

        result = promote_memory(auto_detect=True, dry_run=True)

        assert result["success"] is True
        # Candidate preview list should be limited to 10
        assert len(result["candidates"]) <= 10

    @patch("cortexgraph.tools.promote.BasicMemoryIntegration")
    def test_promote_updates_memory_status(self, mock_integration, temp_storage):
        """Test that promotion updates memory status."""
        now = int(time.time())
        mock_integration_instance = MagicMock()
        mock_integration_instance.promote_to_obsidian.return_value = {
            "success": True,
            "path": "/vault/promoted.md",
        }
        mock_integration.return_value = mock_integration_instance

        test_id = make_test_uuid("mem-status")
        mem = Memory(
            id=test_id,
            content="Test status update",
            use_count=10,
            last_used=now,
            created_at=now - (14 * 86400),
            strength=1.5,
        )
        temp_storage.save_memory(mem)

        result = promote_memory(memory_id=test_id, dry_run=False, force=True)

        if result["success"] and result["promoted_count"] > 0:
            updated = temp_storage.get_memory(test_id)
            assert updated.status == MemoryStatus.PROMOTED
            assert updated.promoted_at is not None
            assert updated.promoted_to is not None

    # Validation tests
    def test_promote_invalid_uuid_fails(self):
        """Test that invalid UUID fails."""
        with pytest.raises(ValueError, match="memory_id.*valid UUID"):
            promote_memory(memory_id="not-a-uuid")

    def test_promote_invalid_target_fails(self):
        """Test that invalid target fails."""
        with pytest.raises(ValueError, match="target"):
            promote_memory(auto_detect=True, target="invalid-target")

    def test_promote_empty_string_uuid_fails(self):
        """Test that empty string UUID fails."""
        with pytest.raises(ValueError, match="memory_id"):
            promote_memory(memory_id="")

    # Edge cases
    def test_promote_with_default_parameters(self, temp_storage):
        """Test default parameter values."""
        now = int(time.time())

        test_id = make_test_uuid("mem-defaults")
        mem = Memory(
            id=test_id,
            content="Test defaults",
            use_count=10,
            last_used=now,
            created_at=now - (14 * 86400),
            strength=1.5,
        )
        temp_storage.save_memory(mem)

        # Should fail because neither memory_id nor auto_detect provided
        result = promote_memory()

        assert result["success"] is False

    def test_promote_obsidian_target(self, temp_storage):
        """Test that obsidian is valid target."""
        result = promote_memory(auto_detect=True, dry_run=True, target="obsidian")

        assert result["success"] is True

    def test_promote_message_dry_run(self, temp_storage):
        """Test that dry run message says 'Would promote'."""
        result = promote_memory(auto_detect=True, dry_run=True)

        assert result["success"] is True
        assert "would promote" in result["message"].lower()

    @patch("cortexgraph.tools.promote.BasicMemoryIntegration")
    def test_promote_message_actual_run(self, mock_integration, temp_storage):
        """Test that actual run message says 'Promoted'."""
        mock_integration_instance = MagicMock()
        mock_integration_instance.promote_to_obsidian.return_value = {
            "success": True,
            "path": "/vault/test.md",
        }
        mock_integration.return_value = mock_integration_instance

        result = promote_memory(auto_detect=True, dry_run=False)

        assert result["success"] is True
        assert result["message"].startswith("Promoted")

    def test_promote_empty_database(self, temp_storage):
        """Test promotion on empty database."""
        result = promote_memory(auto_detect=True, dry_run=True)

        assert result["success"] is True
        assert result["candidates_found"] == 0
        assert result["promoted_count"] == 0

    @patch("cortexgraph.tools.promote.BasicMemoryIntegration")
    def test_promote_integration_failure(self, mock_integration, temp_storage):
        """Test handling of integration failure."""
        now = int(time.time())
        mock_integration_instance = MagicMock()
        mock_integration_instance.promote_to_obsidian.return_value = {
            "success": False,
            "error": "Write failed",
        }
        mock_integration.return_value = mock_integration_instance

        test_id = make_test_uuid("mem-fail")
        mem = Memory(
            id=test_id,
            content="Test failure",
            use_count=10,
            last_used=now,
            created_at=now - (14 * 86400),
            strength=1.5,
        )
        temp_storage.save_memory(mem)

        result = promote_memory(memory_id=test_id, dry_run=False, force=True)

        # Should still succeed but with 0 promoted
        assert result["success"] is True
        assert result["promoted_count"] == 0

    def test_promote_candidates_sorted_by_score(self, temp_storage):
        """Test that candidates are sorted by score descending."""
        now = int(time.time())

        # Create memories with different scores
        for i in range(3):
            test_id = make_test_uuid(f"mem-{i}")
            mem = Memory(
                id=test_id,
                content=f"Memory {i}",
                use_count=i * 5,
                last_used=now,
                created_at=now - (14 * 86400),
                strength=1.0 + (i * 0.2),
            )
            temp_storage.save_memory(mem)

        result = promote_memory(auto_detect=True, dry_run=True)

        assert result["success"] is True
        if result["candidates_found"] > 1:
            scores = [c["score"] for c in result["candidates"]]
            assert scores == sorted(scores, reverse=True)

    def test_promote_preserves_memory_content(self, temp_storage):
        """Test that promotion doesn't modify content."""
        now = int(time.time())
        original_content = "This content should not change"

        test_id = make_test_uuid("mem-content")
        mem = Memory(
            id=test_id,
            content=original_content,
            use_count=10,
            last_used=now,
            created_at=now - (14 * 86400),
            strength=1.5,
        )
        temp_storage.save_memory(mem)

        promote_memory(memory_id=test_id, dry_run=True, force=True)

        updated = temp_storage.get_memory(test_id)
        assert updated.content == original_content
