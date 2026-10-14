# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
"""
DeepDiver Multi-Agent System

A comprehensive multi-agent system with MCP integration, local workspace management,
and advanced knowledge management capabilities.
"""

import os
import threading

__version__ = "2.0.0"
__author__ = "DeepDiver Team"
__description__ = "Multi-Agent System with MCP and Local Workspace Integration"


_thread_context = threading.local()
_SEARCH_SOURCE_ENV = {
    "websearch": "SEARCH_SOURCE_WEBSEARCH",
    "pubmed": "SEARCH_SOURCE_PUBMED",
    "arxiv": "SEARCH_SOURCE_ARXIV",
    "google_scholar": "SEARCH_SOURCE_GOOGLE_SCHOLAR",
    "scihub": "SEARCH_SOURCE_SCIHUB",
    "springer": "SEARCH_SOURCE_SPRINGER",
    "rag": "SEARCH_SOURCE_RAG",
}


def set_thread_session(
    session_id: str,
    workspace_path: str,
    human_in_loop_phase2: bool = False,
    search_sources=None,
) -> None:
    """Bind request-scoped state to the current worker thread.

    New request entry points use this isolated context instead of mutating
    process-wide environment variables. Readers retain an environment fallback
    for legacy scripts that have not entered through the API layer.
    """
    _thread_context.session_id = str(session_id or "")
    _thread_context.workspace_path = str(workspace_path or "")
    _thread_context.human_in_loop_phase2 = bool(human_in_loop_phase2)
    _thread_context.search_sources = (
        None
        if search_sources is None
        else {str(key).lower(): bool(value) for key, value in search_sources.items()}
    )


def get_thread_session_id() -> str:
    value = getattr(_thread_context, "session_id", None)
    if value is not None:
        return value
    return os.environ.get("AGENT_SESSION_ID", "")


def get_thread_workspace_path() -> str:
    value = getattr(_thread_context, "workspace_path", None)
    if value is not None:
        return value
    return os.environ.get("AGENT_WORKSPACE_PATH", "")


def get_thread_human_in_loop_phase2() -> bool:
    value = getattr(_thread_context, "human_in_loop_phase2", None)
    if value is not None:
        return bool(value)
    return os.environ.get("HUMAN_IN_LOOP_PHASE2", "false").lower() == "true"


def get_thread_search_source(source_key: str, default: bool = True) -> bool:
    """Read one request-scoped search-source preference."""
    key = str(source_key or "").lower()
    values = getattr(_thread_context, "search_sources", None)
    if values is not None:
        return bool(values.get(key, default))
    env_name = _SEARCH_SOURCE_ENV.get(key)
    if not env_name:
        return bool(default)
    return os.environ.get(env_name, str(default)).lower() == "true"


def get_thread_search_sources():
    """Return a complete snapshot suitable for child-thread inheritance."""
    return {
        key: get_thread_search_source(key, True)
        for key in _SEARCH_SOURCE_ENV
    }


def inherit_thread_session(
    session_id: str,
    workspace_path: str,
    human_in_loop_phase2: bool = False,
    search_sources=None,
) -> None:
    """Explicitly copy request state into a ThreadPoolExecutor child thread."""
    set_thread_session(
        session_id,
        workspace_path,
        human_in_loop_phase2,
        search_sources=search_sources,
    )


def clear_thread_session() -> None:
    """Clear request state before a pooled worker thread is reused."""
    for name in ("session_id", "workspace_path", "human_in_loop_phase2", "search_sources"):
        if hasattr(_thread_context, name):
            delattr(_thread_context, name)
