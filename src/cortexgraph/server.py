"""MCP Server entry point for CortexGraph."""

import argparse
import logging
import sys
from pathlib import Path

from .config import get_config
from .context import db, mcp
from .core.decay import calculate_halflife
from .security.permissions import ensure_secure_storage, secure_config_file
from .security.secrets import scan_file_for_secrets, should_warn_about_secrets

# Import tools to register them with the decorator
from .tools import (
    analyze_for_recall,
    analyze_message,
    auto_recall_tool,
    cluster,
    consolidate,
    create_relation,
    gc,
    open_memories,
    performance,
    promote,
    read_graph,
    save,
    search,
    search_unified,
    touch,
)

# Explicitly reference imports to satisfy linters (these register MCP tools via decorators)
_TOOL_MODULES = (
    analyze_for_recall,
    analyze_message,
    auto_recall_tool,
    cluster,
    consolidate,
    create_relation,
    gc,
    open_memories,
    performance,
    promote,
    read_graph,
    save,
    search,
    search_unified,
    touch,
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_version() -> str:
    """Get version from pyproject.toml."""
    try:
        # Try Python 3.11+ built-in tomllib
        try:
            import tomllib  # type: ignore[import-not-found]
        except ImportError:
            # Fall back to toml package for older Python
            import tomli as tomllib  # type: ignore[import-not-found]

        # Find pyproject.toml relative to this file
        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"

        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)
                version: str = pyproject_data.get("project", {}).get("version", "unknown")
                return version
    except Exception as e:
        logger.debug(f"Could not read version from pyproject.toml: {e}")

    return "unknown"


def initialize_server() -> None:
    """Initialize logging and database connections."""
    config = get_config()
    logging.getLogger().setLevel(config.log_level)

    logger.info("Initializing CortexGraph")
    logger.info(f"Storage (JSONL): {config.storage_path}")

    model = config.decay_model
    if model == "power_law":
        logger.info(
            "Decay model: power_law (alpha=%.3f, halflife=%.1f days)",
            config.pl_alpha,
            config.pl_halflife_days,
        )
    elif model == "two_component":
        hl_fast = calculate_halflife(config.tc_lambda_fast)
        hl_slow = calculate_halflife(config.tc_lambda_slow)
        logger.info(
            "Decay model: two_component (w_fast=%.2f, hl_fast=%.1f d, hl_slow=%.1f d)",
            config.tc_weight_fast,
            hl_fast,
            hl_slow,
        )
    else:  # exponential
        hl = calculate_halflife(config.decay_lambda)
        logger.info(
            "Decay model: exponential (lambda=%.3e, halflife=%.1f days)",
            config.decay_lambda,
            hl,
        )

    logger.info(f"Embeddings: {'enabled' if config.enable_embeddings else 'disabled'}")

    db.connect()
    logger.info(f"Storage initialized with {db.count_memories()} memories")

    # Apply secure permissions to storage directory
    try:
        stats = ensure_secure_storage(config.storage_path)
        logger.info(f"Secured storage: {stats['files']} files, {stats['directories']} directories")
        if stats["errors"] > 0:
            logger.warning(f"Unable to secure {stats['errors']} items (check permissions)")
    except Exception as e:
        logger.warning(f"Unable to secure storage permissions: {e}")

    # Secure .env file if it exists
    env_path = Path(".env")
    if env_path.exists():
        try:
            secure_config_file(env_path)
            logger.info("Secured .env configuration file")
        except Exception as e:
            logger.warning(f"Unable to secure .env file: {e}")

    # Check XDG config directory .env
    xdg_env = Path.home() / ".config" / "cortexgraph" / ".env"
    if xdg_env.exists() and xdg_env != env_path.resolve():
        try:
            secure_config_file(xdg_env)
            logger.info(f"Secured XDG config file: {xdg_env}")
        except Exception as e:
            logger.warning(f"Unable to secure {xdg_env}: {e}")

    # Check .env files for secrets (if detection enabled)
    if config.detect_secrets:
        env_files_to_check = []
        if env_path.exists():
            env_files_to_check.append(env_path)
        if xdg_env.exists() and xdg_env != env_path.resolve():
            env_files_to_check.append(xdg_env)

        for env_file in env_files_to_check:
            try:
                matches = scan_file_for_secrets(str(env_file))
                if should_warn_about_secrets(matches):
                    logger.warning(
                        f"⚠️  Potential secrets detected in {env_file}! "
                        f"Found {len(matches)} pattern matches. "
                        f"Ensure .env files contain only YOUR secrets, not example values."
                    )
            except Exception:
                # Ignore errors scanning .env (might be locked, etc.)
                pass

    logger.info("MCP server tools registered (15 tools)")


def main_sync() -> None:
    """Synchronous entry point for the server."""
    parser = argparse.ArgumentParser(
        description="CortexGraph: Temporal memory management for AI assistants"
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"cortexgraph {get_version()}",
    )

    parser.parse_args()

    try:
        initialize_server()
        mcp.run()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main_sync()
