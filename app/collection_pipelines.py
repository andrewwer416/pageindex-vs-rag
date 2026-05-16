"""Per-pipeline collection builders.

Stubs in Phase 1 — the schema and orchestration are settled; the actual
build_rag / build_pageindex_router / build_graphrag functions land in the
following phases. Each currently raises NotImplementedError so the
Collections page can render and we can sanity-check the UX before pipeline
work begins.
"""
from __future__ import annotations
from typing import Callable


def build_rag(col_id: str, doc_ids: list[str]) -> dict:
    """Concatenate per-doc FAISS indices into a collection-wide FAISS, with
    each chunk row carrying doc_id + doc_name for attribution in answers.
    Stub — implemented in Phase 2."""
    raise NotImplementedError("RAG collection builder lands in Phase 2.")


def build_pageindex_router(col_id: str, doc_ids: list[str]) -> dict:
    """For each doc, ask the LLM for a 3-sentence summary + 5 key topics.
    Persist to doc_summaries.json so the query-time router can pick which
    docs to walk. Stub — implemented in Phase 3."""
    raise NotImplementedError("PageIndex doc-router lands in Phase 3.")


def build_graphrag(col_id: str, doc_ids: list[str], progress: Callable[[str, dict], None] | None = None) -> dict:
    """Run LLM-grade entity resolution across all per-doc graphs, merge them,
    detect communities on the unified graph, and summarize. Stub —
    implemented in Phase 4."""
    raise NotImplementedError("GraphRAG collection builder lands in Phase 4.")
