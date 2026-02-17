"""Pytest configuration and shared fixtures for CortexGraph test suite.

This module provides reusable fixtures and utilities for testing the CortexGraph
temporal memory system. All fixtures defined here are automatically available to
all test files without explicit imports.

Core Fixtures
------------
test_config : Config (autouse=True)
    Automatically applied to all tests. Sets consistent decay parameters and
    disables embeddings by default.

temp_storage : JSONLStorage
    Creates isolated temporary JSONL storage for each test. Automatically
    monkey-patches global db instances across 11 tool modules for proper
    test isolation.

Config Mock Fixtures
------------------
mock_config_preprocessor : Config
    Config with enable_preprocessing=False for testing legacy behavior without
    auto-enrichment of entities and strength.

mock_config_embeddings : Config
    Config with enable_embeddings=True and test model configured for testing
    semantic search with embeddings.

Embedding Mock Fixtures
----------------------
mock_embeddings_save : MagicMock
    Mocks SentenceTransformer for the save module. Returns predictable test
    embeddings [0.1, 0.2, 0.3]. Use with mock_config_embeddings.

mock_embeddings_search : MagicMock
    Mocks SentenceTransformer for the search module. Returns predictable test
    embeddings [0.1, 0.2, 0.3]. Use with mock_config_embeddings.

Utility Functions
----------------
make_test_uuid(name: str) -> str
    Generates deterministic UUIDs for reproducible tests. Same input always
    returns same UUID.

mock_embeddings_setup(monkeypatch, module_path) -> MagicMock
    Helper function to setup embedding mocks for any module. Used internally
    by mock_embeddings_* fixtures.

Usage Examples
-------------
Basic test with storage::

    def test_save_memory(temp_storage):
        mem = Memory(id="test", content="Test")
        temp_storage.save_memory(mem)
        assert temp_storage.get_memory("test") is not None

Test with preprocessing disabled::

    def test_no_preprocessing(mock_config_preprocessor, temp_storage):
        result = save_memory(content="Test")
        # Entities won't be auto-extracted

Test with embeddings::

    def test_embeddings(
        mock_config_embeddings,
        mock_embeddings_save,
        temp_storage
    ):
        result = save_memory(content="Test")
        assert result["has_embedding"] is True

Notes
-----
- All fixtures use pytest's function scope by default (new instance per test)
- temp_storage automatically cleans up after each test
- Config fixtures patch at global level to avoid module-specific coupling
- Embedding fixtures return predictable embeddings for deterministic testing

See Also
--------
tests/README.md : Comprehensive test documentation and patterns
"""

import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

import cortexgraph.context
import cortexgraph.tools.auto_recall_tool
import cortexgraph.tools.cluster
import cortexgraph.tools.consolidate
import cortexgraph.tools.create_relation
import cortexgraph.tools.gc
import cortexgraph.tools.open_memories
import cortexgraph.tools.promote
import cortexgraph.tools.read_graph
import cortexgraph.tools.save
import cortexgraph.tools.search
import cortexgraph.tools.touch
from cortexgraph.config import Config, get_config, set_config
from cortexgraph.storage.jsonl_storage import JSONLStorage

# ============================================================================
# Beads CLI Detection and Skip Markers
# ============================================================================


def is_beads_available() -> bool:
    """Check if the beads CLI (bd) is available on the system."""
    return shutil.which("bd") is not None


# Check once at import time
BEADS_AVAILABLE = is_beads_available()


@pytest.fixture
def requires_beads():
    """Fixture that skips tests if beads CLI is not available.

    Usage:
        def test_something_with_beads(requires_beads, temp_storage):
            # This test will be skipped if bd CLI is not installed
            ...
    """
    if not BEADS_AVAILABLE:
        pytest.skip("beads CLI (bd) not found - skipping test")


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "requires_beads: mark test as requiring beads CLI (bd)")


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with @pytest.mark.requires_beads if bd is unavailable."""
    if BEADS_AVAILABLE:
        return  # bd is available, run all tests

    skip_beads = pytest.mark.skip(reason="beads CLI (bd) not installed")
    for item in items:
        if "requires_beads" in item.keywords:
            item.add_marker(skip_beads)


def make_test_uuid(name: str) -> str:
    """Generate a deterministic UUID for testing based on a name.

    Args:
        name: A short descriptive name (e.g., 'test-123', 'mem-promoted')

    Returns:
        A valid UUID string generated deterministically from the name

    Examples:
        >>> make_test_uuid("test-123")
        'a1b2c3d4-...'  # Always returns the same UUID for "test-123"
    """
    # Use UUID5 with a fixed namespace to generate deterministic UUIDs
    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
    return str(uuid.uuid5(namespace, name))


@pytest.fixture(autouse=True)
def test_config():
    """Set up a test configuration for all tests."""
    config = Config(
        decay_lambda=2.673e-6,
        decay_beta=0.6,
        forget_threshold=0.05,
        promote_threshold=0.65,
        enable_embeddings=False,  # Disable embeddings in tests
    )
    set_config(config)
    yield config


@pytest.fixture
def temp_storage(monkeypatch):
    """Create a temporary JSONL storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir)
        storage = JSONLStorage(storage_path=storage_dir)
        storage.connect()

        # Update global config to point to temp storage
        config = get_config()
        original_storage_path = config.storage_path
        config.storage_path = storage_dir

        # Monkey-patch the global db instance in context and all tool modules

        modules_to_patch = [
            cortexgraph.context,
            cortexgraph.tools.save,
            cortexgraph.tools.search,
            cortexgraph.tools.touch,
            cortexgraph.tools.gc,
            cortexgraph.tools.promote,
            cortexgraph.tools.cluster,
            cortexgraph.tools.consolidate,
            cortexgraph.tools.create_relation,
            cortexgraph.tools.open_memories,
            cortexgraph.tools.read_graph,
            cortexgraph.tools.auto_recall_tool,
        ]
        for module in modules_to_patch:
            monkeypatch.setattr(module, "db", storage)

        yield storage

        # Restore original config
        config.storage_path = original_storage_path
        storage.close()


@pytest.fixture
def mock_config_preprocessor(monkeypatch):
    """Mock config with preprocessing disabled.

    Use this fixture for tests that need legacy behavior without auto-enrichment
    of entities and strength. This is useful for testing basic memory operations
    without the natural language preprocessing layer.

    Example:
        def test_basic_save(mock_config_preprocessor, temp_storage):
            result = save_memory(content="Test")
            # Entities will be empty (not auto-extracted)
    """
    config = get_config()
    config.enable_preprocessing = False
    # Patch at global level to avoid module-specific coupling
    monkeypatch.setattr(cortexgraph.config, "_config", config)
    return config


@pytest.fixture
def mock_config_embeddings(monkeypatch):
    """Mock config with embeddings enabled.

    Use this fixture for tests that need semantic search with embeddings.
    Configures a test model and ensures all embedding-related config fields
    are set appropriately.

    Example:
        def test_semantic_search(mock_config_embeddings, temp_storage):
            result = search_memory(query="AI", use_embeddings=True)
            # Will use mocked embeddings for similarity scoring
    """
    config = get_config()
    config.enable_embeddings = True
    config.embed_model = "test-model"
    config.search_default_preview_length = 300
    # Patch at global level to avoid module-specific coupling
    monkeypatch.setattr(cortexgraph.config, "_config", config)
    return config


def mock_embeddings_setup(monkeypatch, module_path):
    """Setup embedding mocks for a given module.

    This helper creates a mock SentenceTransformer model that returns
    predictable embeddings for testing purposes.

    Args:
        monkeypatch: pytest monkeypatch fixture
        module_path: Module path to patch (e.g., "cortexgraph.tools.save")

    Returns:
        MagicMock: Configured mock model for embedding generation

    Example:
        model = mock_embeddings_setup(monkeypatch, "cortexgraph.tools.save")
        # Now SentenceTransformer in save module returns mock model
    """
    from unittest.mock import MagicMock

    # Set availability flag
    monkeypatch.setattr(f"{module_path}.SENTENCE_TRANSFORMERS_AVAILABLE", True)

    # Create mock model that returns predictable embeddings
    mock_model = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
    mock_model.encode.return_value = mock_embedding

    # Patch transformer class
    mock_transformer_class = MagicMock(return_value=mock_model)
    monkeypatch.setattr(f"{module_path}.SentenceTransformer", mock_transformer_class)

    return mock_model


@pytest.fixture
def mock_embeddings_save(monkeypatch):
    """Embedding mocks for save module.

    Sets up SentenceTransformer mocks for the save_memory tool.
    Use with mock_config_embeddings to test embedding generation.

    Example:
        def test_save_with_embeddings(
            mock_config_embeddings,
            mock_embeddings_save,
            temp_storage
        ):
            result = save_memory(content="Test")
            assert result["has_embedding"] is True
    """
    return mock_embeddings_setup(monkeypatch, "cortexgraph.tools.save")


@pytest.fixture
def mock_embeddings_search(monkeypatch):
    """Embedding mocks for search module.

    Sets up SentenceTransformer mocks for the search_memory tool.
    Use with mock_config_embeddings to test semantic search.

    Example:
        def test_semantic_search(
            mock_config_embeddings,
            mock_embeddings_search,
            temp_storage
        ):
            result = search_memory(query="AI", use_embeddings=True)
            # Will calculate similarity using mocked embeddings
    """
    return mock_embeddings_setup(monkeypatch, "cortexgraph.tools.search")
