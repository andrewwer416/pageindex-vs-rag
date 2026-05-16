"""Collections of documents — the multi-doc tier on top of `library.py`.

A collection groups a set of ready-state docs from the library and builds three
collection-level indices (RAG, PageIndex doc-router, GraphRAG with entity
resolution). Layout:

  index/collections/<col_id>/
    meta.json                       {"name", "doc_ids", "status", "error",
                                     "created_at", "indexed_at", "stats",
                                     "doc_fingerprint"}
    rag/
      faiss.index                   union of per-doc vectors
      chunks.json                   each chunk row has doc_id + doc_name
    pageindex/
      doc_summaries.json            {doc_id: {name, summary, key_topics}}
    graphrag/
      property_graph_store.json     merged, post entity-resolution
      community_summaries.json
      entity_resolution.json        {canonical_name: [aliases]}
      extraction_debug.json

Doc-add semantics: any change to the doc set invalidates the collection — the
status flips to `stale` and the Collection page blocks queries behind a
'Rebuild required' banner until the user explicitly rebuilds. We track a
fingerprint of the doc_id set + each doc's indexed_at so we can detect when
underlying per-doc indices have been re-built independently.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

from . import config, library


COLLECTIONS_DIR = config.INDEX_DIR / "collections"

# ─── paths ────────────────────────────────────────────────────────────────────

def col_dir(col_id: str) -> Path:
    return COLLECTIONS_DIR / col_id


def meta_path(col_id: str) -> Path:
    return col_dir(col_id) / "meta.json"


def rag_dir(col_id: str) -> Path:
    return col_dir(col_id) / "rag"


def rag_faiss_path(col_id: str) -> Path:
    return rag_dir(col_id) / "faiss.index"


def rag_chunks_path(col_id: str) -> Path:
    return rag_dir(col_id) / "chunks.json"


def pageindex_dir(col_id: str) -> Path:
    return col_dir(col_id) / "pageindex"


def doc_summaries_path(col_id: str) -> Path:
    return pageindex_dir(col_id) / "doc_summaries.json"


def graphrag_dir(col_id: str) -> Path:
    return col_dir(col_id) / "graphrag"


# ─── fingerprint ──────────────────────────────────────────────────────────────

def _doc_fingerprint(doc_ids: list[str]) -> str:
    """A stable hash of (doc_id, indexed_at) pairs for the collection's docs.
    Changes whenever any member doc is re-indexed or the membership changes."""
    rows: list[tuple[str, float]] = []
    for did in sorted(doc_ids):
        try:
            m = library.get_document(did)
        except Exception:
            rows.append((did, 0.0))
            continue
        rows.append((did, float(m.get("indexed_at") or 0)))
    h = hashlib.sha1()
    for did, ts in rows:
        h.update(f"{did}|{ts}\n".encode())
    return h.hexdigest()


# ─── CRUD ─────────────────────────────────────────────────────────────────────

def list_collections() -> list[dict]:
    """Return one meta dict per collection, newest first."""
    COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for child in COLLECTIONS_DIR.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            meta = json.loads((child / "meta.json").read_text())
            meta["col_id"] = child.name
            # Live status: 'stale' if the doc fingerprint changed since last build.
            current_fp = _doc_fingerprint(meta.get("doc_ids", []))
            if meta.get("status") == "ready" and current_fp != meta.get("doc_fingerprint"):
                meta["status"] = "stale"
            out.append(meta)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    out.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return out


def get_collection(col_id: str) -> dict:
    meta = json.loads(meta_path(col_id).read_text())
    meta["col_id"] = col_id
    current_fp = _doc_fingerprint(meta.get("doc_ids", []))
    if meta.get("status") == "ready" and current_fp != meta.get("doc_fingerprint"):
        meta["status"] = "stale"
    return meta


def _write_meta(col_id: str, **fields) -> dict:
    p = meta_path(col_id)
    try:
        meta = json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        meta = {}
    meta.update(fields)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2))
    return meta


def create_collection(name: str, doc_ids: list[str]) -> str:
    """Persist a new collection record. Does NOT trigger indexing — call
    build_collection(col_id, ...) afterwards."""
    if not name.strip():
        raise ValueError("Collection name cannot be empty.")
    # Validate doc_ids exist and are at ready/partial state. Reject non-ready
    # docs up front rather than failing mid-build.
    bad: list[str] = []
    for did in doc_ids:
        try:
            m = library.get_document(did)
        except Exception:
            bad.append(f"{did} (missing)")
            continue
        if m.get("status") not in ("ready", "partial"):
            bad.append(f"{did} ({m.get('status', '?')})")
    if bad:
        raise ValueError(f"Cannot add docs to collection — not ready: {', '.join(bad)}")

    col_id = str(uuid.uuid4())
    col_dir(col_id).mkdir(parents=True, exist_ok=True)
    _write_meta(
        col_id,
        name=name.strip(),
        doc_ids=list(doc_ids),
        status="created",
        error="",
        created_at=time.time(),
        indexed_at=None,
        stats={},
        doc_fingerprint="",
    )
    return col_id


def update_collection_docs(col_id: str, doc_ids: list[str]) -> dict:
    """Replace the collection's doc set. Flips status to `stale` so the UI
    forces a rebuild before queries. Does not delete the old indices — they
    stay around until rebuild overwrites them."""
    _write_meta(col_id, doc_ids=list(doc_ids), status="stale", error="")
    return get_collection(col_id)


def delete_collection(col_id: str) -> tuple[bool, str]:
    d = col_dir(col_id)
    if not d.exists():
        return True, ""
    try:
        shutil.rmtree(d)
        return True, ""
    except (OSError, PermissionError) as e:
        return False, f"Could not delete {col_id}: {e!r}"


# ─── build ────────────────────────────────────────────────────────────────────

def build_collection(
    col_id: str,
    progress: Callable[[str, dict], None] | None = None,
) -> dict:
    """Build all three collection-level indices for `col_id`. Phases:
        'rag', 'pageindex-router', 'graphrag-resolution', 'graphrag-merge',
        'graphrag-communities', 'done', 'failed'.

    Each pipeline is a separate phase block so a failure in one (e.g. GraphRAG
    entity resolution timing out) doesn't tank the whole collection — the
    others land and the failed phase is recorded for the user to retry.
    """
    def _p(phase: str, **kw):
        if progress:
            progress(phase, kw)

    meta = get_collection(col_id)
    doc_ids = meta.get("doc_ids", [])
    if not doc_ids:
        msg = "Collection has no documents."
        _write_meta(col_id, status="failed", error=msg)
        _p("failed", error=msg)
        return get_collection(col_id)

    stats: dict = {}
    pipeline_errors: dict[str, str] = {}

    # ----- Phase A: RAG ---------------------------------------------------
    try:
        _write_meta(col_id, status="rag-building", error="")
        _p("rag", doc_count=len(doc_ids))
        from . import collection_pipelines
        rag_stats = collection_pipelines.build_rag(col_id, doc_ids)
        stats["rag"] = rag_stats
        _p("rag-done", **rag_stats)
    except Exception as e:
        pipeline_errors["rag"] = f"{type(e).__name__}: {e}"
        _p("rag-failed", error=pipeline_errors["rag"])

    # ----- Phase B: PageIndex doc-router ----------------------------------
    try:
        _write_meta(col_id, status="pageindex-router-building")
        _p("pageindex-router", doc_count=len(doc_ids))
        from . import collection_pipelines
        pi_stats = collection_pipelines.build_pageindex_router(col_id, doc_ids)
        stats["pageindex"] = pi_stats
        _p("pageindex-router-done", **pi_stats)
    except Exception as e:
        pipeline_errors["pageindex"] = f"{type(e).__name__}: {e}"
        _p("pageindex-router-failed", error=pipeline_errors["pageindex"])

    # ----- Phase C: GraphRAG entity resolution + merge + communities ------
    try:
        _write_meta(col_id, status="graphrag-resolving")
        _p("graphrag-resolution", doc_count=len(doc_ids))
        from . import collection_pipelines
        gr_stats = collection_pipelines.build_graphrag(col_id, doc_ids, progress=_p)
        stats["graphrag"] = gr_stats
        _p("graphrag-done", **gr_stats)
    except Exception as e:
        import traceback as _tb
        tb = _tb.format_exc()
        print("=" * 70, flush=True)
        print(f"Collection {col_id} GraphRAG build failed:", flush=True)
        print(tb, flush=True)
        print("=" * 70, flush=True)
        pipeline_errors["graphrag"] = f"{type(e).__name__}: {e}\n\n{tb}"
        _p("graphrag-failed", error=pipeline_errors["graphrag"])

    # ----- Finalize -------------------------------------------------------
    fp = _doc_fingerprint(doc_ids)
    if pipeline_errors:
        # If at least one pipeline succeeded, the collection is `partial`;
        # otherwise it's `failed`. Either way persist the per-pipeline error
        # so the UI can show what to retry.
        any_success = any(k in stats for k in ("rag", "pageindex", "graphrag"))
        new_status = "partial" if any_success else "failed"
        _write_meta(
            col_id,
            status=new_status,
            error="; ".join(f"{k}: {v.splitlines()[0]}" for k, v in pipeline_errors.items()),
            stats=stats,
            pipeline_errors=pipeline_errors,
            indexed_at=time.time(),
            doc_fingerprint=fp if new_status == "partial" else "",
        )
        _p("done" if new_status == "partial" else "failed", status=new_status)
    else:
        _write_meta(
            col_id,
            status="ready",
            error="",
            stats=stats,
            pipeline_errors={},
            indexed_at=time.time(),
            doc_fingerprint=fp,
        )
        _p("done", status="ready")

    return get_collection(col_id)


def is_stale(col_id: str) -> bool:
    """True if the collection's saved doc-fingerprint no longer matches the
    current fingerprint of its member docs."""
    meta = get_collection(col_id)
    return meta.get("status") == "stale"


def collection_kwargs(col_id: str) -> dict:
    """Return kwargs each collection-pipeline's answer() will accept. Mirrors
    library.doc_kwargs() so the Collection Compare page renders uniformly."""
    meta = get_collection(col_id)
    return {
        "rag": {
            "faiss_path": rag_faiss_path(col_id),
            "chunks_path": rag_chunks_path(col_id),
            "doc_name": meta.get("name") or col_id,
        },
        "pi": {
            "doc_summaries_path": doc_summaries_path(col_id),
            "doc_ids": meta.get("doc_ids", []),
            "doc_name": meta.get("name") or col_id,
        },
        "graphrag": {
            "graphrag_dir": graphrag_dir(col_id),
            "doc_name": meta.get("name") or col_id,
        },
    }
