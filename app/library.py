"""Library of user-uploaded documents.

Persistent layout:
  index/library/<doc_id>/
    source.<ext>             original upload (.pdf / .md / .txt)
    faiss.index              traditional-RAG vectors
    chunks.json              chunk metadata + text
    pageindex_workspace/     PageIndex tree workspace (one doc per dir)
      <pi_doc_id>.json
      _meta.json
    pageindex_doc_id.txt     points to the workspace doc_id
    meta.json                {"name", "ext", "status", "error", "created_at",
                              "indexed_at", "page_count", "pi_doc_id"}

Status transitions:
  uploaded → rag-indexing → pageindex-indexing → ready
                         ↘                   ↘
                          failed              failed
"""
from __future__ import annotations
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable

from . import config


LIBRARY_DIR = config.INDEX_DIR / "library"
SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt"}


# ---------- paths ----------

def doc_dir(doc_id: str) -> Path:
    return LIBRARY_DIR / doc_id


def source_path(doc_id: str, ext: str | None = None) -> Path:
    """Locate the source file in the doc's dir, optionally with explicit ext."""
    d = doc_dir(doc_id)
    if ext:
        return d / f"source{ext}"
    for p in d.glob("source.*"):
        return p
    raise FileNotFoundError(f"No source file in {d}")


def faiss_path(doc_id: str) -> Path:
    return doc_dir(doc_id) / "faiss.index"


def chunks_path(doc_id: str) -> Path:
    return doc_dir(doc_id) / "chunks.json"


def pageindex_workspace(doc_id: str) -> Path:
    return doc_dir(doc_id) / "pageindex_workspace"


def pageindex_doc_id_file(doc_id: str) -> Path:
    return doc_dir(doc_id) / "pageindex_doc_id.txt"


def graphrag_dir(doc_id: str) -> Path:
    return doc_dir(doc_id) / "graphrag"


def meta_path(doc_id: str) -> Path:
    return doc_dir(doc_id) / "meta.json"


# ---------- CRUD ----------

def list_documents() -> list[dict]:
    """Return one meta dict per document, newest first.
    Skips `.trash-*` dirs left behind by failed Windows deletes."""
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    _gc_trash()  # opportunistic cleanup
    out: list[dict] = []
    for child in LIBRARY_DIR.iterdir():
        if not child.is_dir() or child.name.startswith(".trash-"):
            continue
        try:
            meta = json.loads((child / "meta.json").read_text())
            meta["doc_id"] = child.name
            out.append(meta)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    out.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return out


def get_document(doc_id: str) -> dict:
    meta = json.loads(meta_path(doc_id).read_text())
    meta["doc_id"] = doc_id
    return meta


def _write_meta(doc_id: str, **fields) -> dict:
    p = meta_path(doc_id)
    try:
        meta = json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        meta = {}
    meta.update(fields)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2))
    return meta


def add_document(file_bytes: bytes, original_filename: str) -> str:
    """Persist an uploaded file and return its doc_id. Does NOT trigger indexing.
    Call build_indices(doc_id, …) after."""
    ext = Path(original_filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type {ext}. Allowed: {sorted(SUPPORTED_EXTS)}")

    doc_id = str(uuid.uuid4())
    d = doc_dir(doc_id)
    d.mkdir(parents=True, exist_ok=True)
    src = d / f"source{ext}"
    src.write_bytes(file_bytes)

    _write_meta(
        doc_id,
        name=original_filename,
        ext=ext,
        status="uploaded",
        error="",
        created_at=time.time(),
        indexed_at=None,
        page_count=None,
        pi_doc_id=None,
    )
    return doc_id


def delete_document(doc_id: str) -> tuple[bool, str]:
    """Remove a document directory. Returns (ok, message).

    On Windows it's common for a file inside the doc dir to be held open by
    streamlit's filewatcher, the OS search-indexer, or AV — `shutil.rmtree`
    then raises PermissionError on the first locked file and leaves the rest
    behind. Strategy:
      1. Try a normal rmtree (works on Linux always, and on Windows when
         no locks remain).
      2. On Windows: retry a few times with a short delay.
      3. As a last resort, rename the directory to `.trash-<doc_id>` so it
         disappears from list_documents() even if the OS can't free the
         bytes right now. Trash dirs are cleaned up on the next library
         operation.
    """
    import time
    d = doc_dir(doc_id)
    if not d.exists():
        return True, ""

    _gc_trash()  # opportunistic cleanup of previous failed deletes

    last_err = None
    for attempt in range(4):
        try:
            shutil.rmtree(d)
            return True, ""
        except (OSError, PermissionError) as e:
            last_err = e
            time.sleep(0.25 * (attempt + 1))

    # Fall back: rename so the doc disappears from list_documents() even if
    # the bytes can't be freed yet.
    try:
        trash = LIBRARY_DIR / f".trash-{doc_id}"
        d.rename(trash)
        return True, (
            "Files are still held open by another process (likely streamlit's "
            "file watcher on Windows). The document was renamed and will be "
            "fully removed the next time the library is opened after a "
            "restart. The doc is hidden from the library now."
        )
    except (OSError, PermissionError):
        return False, f"Could not delete {doc_id}: {last_err!r}"


def _gc_trash() -> None:
    """Best-effort sweep of `.trash-*` dirs left over from prior failed deletes."""
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    for child in LIBRARY_DIR.iterdir():
        if not child.is_dir() or not child.name.startswith(".trash-"):
            continue
        try:
            shutil.rmtree(child)
        except (OSError, PermissionError):
            pass


# ---------- indexing ----------

def _normalize_for_pageindex(src: Path) -> Path:
    """PageIndex's Markdown path expects a real .md. Wrap plain .txt as MD with
    a single top-level heading so md_to_tree has something structural to anchor to."""
    if src.suffix.lower() != ".txt":
        return src
    md = src.with_suffix(".converted.md")
    text = src.read_text(encoding="utf-8", errors="replace")
    md.write_text(f"# Document\n\n{text}", encoding="utf-8")
    return md


def build_indices(
    doc_id: str,
    progress: Callable[[str, dict], None] | None = None,
) -> dict:
    """Build FAISS + PageIndex tree for `doc_id`. Reports progress via callback:
        progress("phase-name", {"detail": ...})
    Phases: 'chunking', 'embedding', 'pageindex', 'done', 'failed'.

    Returns the final meta dict.
    """
    from .rag_pipeline import pdf_to_chunks, build_faiss_index
    from .pageindex_agent import get_client

    def _p(phase: str, **kw):
        if progress:
            progress(phase, kw)

    meta = get_document(doc_id)
    src = source_path(doc_id, ext=meta["ext"])

    # ----- 1. RAG chunks + FAISS -----
    try:
        _write_meta(doc_id, status="rag-indexing", error="")
        _p("chunking", source=str(src))
        chunks = pdf_to_chunks(src)
        _p("embedding", chunks=len(chunks))
        build_faiss_index(chunks, faiss_path=faiss_path(doc_id), chunks_path=chunks_path(doc_id))
    except Exception as e:
        msg = f"RAG indexing failed: {e!r}"
        _write_meta(doc_id, status="failed", error=msg)
        _p("failed", error=msg)
        return get_document(doc_id)

    # ----- 2. PageIndex tree -----
    try:
        _write_meta(doc_id, status="pageindex-indexing")
        _p("pageindex", source=str(src))
        ws = pageindex_workspace(doc_id)
        ws.mkdir(parents=True, exist_ok=True)
        client = get_client(workspace=ws)
        indexing_path = _normalize_for_pageindex(src)
        pi_doc_id = client.index(str(indexing_path))
        pageindex_doc_id_file(doc_id).write_text(pi_doc_id)
        # rough page count
        page_count = None
        try:
            import pymupdf
            if meta["ext"] == ".pdf":
                d = pymupdf.open(src)
                page_count = d.page_count
                d.close()
        except Exception:
            pass
        _write_meta(
            doc_id,
            status="ready",
            error="",
            indexed_at=time.time(),
            pi_doc_id=pi_doc_id,
            page_count=page_count,
        )
        _p("done", pi_doc_id=pi_doc_id, page_count=page_count)
    except Exception as e:
        msg = f"PageIndex tree build failed: {e!r}"
        # The FAISS side already landed, so the doc is partially usable.
        _write_meta(doc_id, status="partial", error=msg)
        _p("failed", error=msg)

    # ----- 3. GraphRAG (knowledge-graph) index — optional / best-effort -----
    if os.environ.get("GRAPHRAG_ENABLED", "true").strip().lower() != "false":
        try:
            from . import graphrag_pipeline
            _p("graphrag", source=str(src))
            _write_meta(doc_id, status="graphrag-indexing")
            # Read the source's text once. For PDFs we already cached pages in
            # the PageIndex workspace JSON, but to keep this independent we
            # re-extract with the table-aware pipeline.
            if meta["ext"] == ".pdf":
                from .extraction import extract_pages_with_tables
                doc_text = "\n\n".join(extract_pages_with_tables(src))
            else:
                doc_text = src.read_text(encoding="utf-8", errors="replace")
            stats = graphrag_pipeline.build_index(
                source_text=doc_text,
                graphrag_dir=graphrag_dir(doc_id),
            )
            _p("graphrag-done", **stats)
            _write_meta(
                doc_id,
                status="ready" if get_document(doc_id).get("status") != "partial" else "partial",
                graphrag=stats,
            )
        except Exception as e:
            msg = f"GraphRAG indexing failed: {e!r}"
            _write_meta(doc_id, graphrag_error=msg)
            _p("graphrag-failed", error=msg)
    else:
        _p("graphrag-skipped", reason="GRAPHRAG_ENABLED=false")

    return get_document(doc_id)


def doc_kwargs(doc_id: str) -> dict:
    """Convenience: returns the kwargs for each pipeline pointed at this doc's
    indices. The Compare page calls answer(**kw["rag"]) / answer(**kw["pi"]) /
    answer(**kw["graphrag"])."""
    meta = get_document(doc_id)
    name = meta.get("name") or doc_id
    return {
        "rag": {
            "faiss_path": faiss_path(doc_id),
            "chunks_path": chunks_path(doc_id),
            "doc_name": name,
        },
        "pi": {
            "workspace": pageindex_workspace(doc_id),
            "doc_id_file": pageindex_doc_id_file(doc_id),
            "doc_name": name,
        },
        "graphrag": {
            "graphrag_dir": graphrag_dir(doc_id),
            "doc_name": name,
        },
    }


def has_graphrag(doc_id: str) -> bool:
    """True if a GraphRAG index has been built and persisted for this doc."""
    return (graphrag_dir(doc_id) / "community_summaries.json").exists()
