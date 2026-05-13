"""Traditional vector RAG: chunk PDF/MD/TXT, embed, query via FAISS.

All public functions optionally accept paths so they can target either the
bundled Tesla-10K indices (the defaults, in config.*) or a user-uploaded doc's
indices stored under the library.
"""
import json
from pathlib import Path

import faiss
import numpy as np
import pymupdf
import tiktoken

from . import config
from .llm import complete
from .embed import get_embedder  # re-exported for callers that imported from here

__all__ = ["get_embedder", "pdf_to_chunks", "build_faiss_index", "load_index", "retrieve", "answer"]


_enc = tiktoken.get_encoding("cl100k_base")


def _read_pages(source_path: Path) -> list[tuple[int, str]]:
    """Return [(page_num, text), …]. For non-PDF formats we treat the whole
    document as 'page 1' so the rest of the chunking logic stays uniform."""
    ext = source_path.suffix.lower()
    if ext == ".pdf":
        doc = pymupdf.open(source_path)
        pages = [(i + 1, page.get_text() or "") for i, page in enumerate(doc)]
        doc.close()
        return pages
    text = source_path.read_text(encoding="utf-8", errors="replace")
    return [(1, text)] if text.strip() else []


def pdf_to_chunks(source_path: Path) -> list[dict]:
    """Split a document into ~CHUNK_TOKENS chunks with overlap, page-aware (when PDF)."""
    pages = _read_pages(Path(source_path))
    chunks: list[dict] = []
    buf_tokens: list[int] = []
    buf_pages: list[int] = []
    for page_num, text in pages:
        if not text.strip():
            continue
        toks = _enc.encode(text)
        i = 0
        while i < len(toks):
            take = config.CHUNK_TOKENS - len(buf_tokens)
            if take <= 0:
                _flush(chunks, buf_tokens, buf_pages)
                buf_tokens, buf_pages = _overlap_tail(buf_tokens, buf_pages)
                take = config.CHUNK_TOKENS - len(buf_tokens)
            slice_toks = toks[i:i + take]
            buf_tokens.extend(slice_toks)
            buf_pages.extend([page_num] * len(slice_toks))
            i += len(slice_toks)
    if buf_tokens:
        _flush(chunks, buf_tokens, buf_pages)
    return chunks


def _flush(chunks: list[dict], buf_tokens: list[int], buf_pages: list[int]) -> None:
    text = _enc.decode(buf_tokens)
    chunks.append({
        "id": len(chunks),
        "text": text,
        "page_start": buf_pages[0] if buf_pages else 0,
        "page_end": buf_pages[-1] if buf_pages else 0,
        "tokens": len(buf_tokens),
    })


def _overlap_tail(buf_tokens: list[int], buf_pages: list[int]) -> tuple[list[int], list[int]]:
    n = min(config.CHUNK_OVERLAP_TOKENS, len(buf_tokens))
    return buf_tokens[-n:], buf_pages[-n:]


def build_faiss_index(chunks: list[dict], faiss_path: Path | None = None, chunks_path: Path | None = None) -> None:
    embedder = get_embedder()
    embeddings = embedder.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    fp = Path(faiss_path) if faiss_path else config.FAISS_PATH
    cp = Path(chunks_path) if chunks_path else config.CHUNKS_PATH
    fp.parent.mkdir(parents=True, exist_ok=True)
    cp.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(fp))
    with open(cp, "w") as f:
        json.dump(chunks, f)
    print(f"FAISS index: {embeddings.shape[0]} vectors, dim={embeddings.shape[1]}")


def load_index(faiss_path: Path | None = None, chunks_path: Path | None = None) -> tuple[faiss.Index, list[dict]]:
    fp = Path(faiss_path) if faiss_path else config.FAISS_PATH
    cp = Path(chunks_path) if chunks_path else config.CHUNKS_PATH
    index = faiss.read_index(str(fp))
    with open(cp) as f:
        chunks = json.load(f)
    return index, chunks


def retrieve(query: str, k: int = config.RAG_TOP_K, faiss_path: Path | None = None, chunks_path: Path | None = None) -> list[dict]:
    """Top-k chunks with similarity scores."""
    index, chunks = load_index(faiss_path=faiss_path, chunks_path=chunks_path)
    embedder = get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q_emb, k)
    out = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0:
            continue
        c = dict(chunks[idx])
        c["score"] = float(score)
        out.append(c)
    return out


def _rag_system_prompt(doc_name: str | None) -> str:
    target = f"about **{doc_name}**" if doc_name else "about the provided document"
    return (
        f"You are a research assistant answering questions {target}.\n"
        "You have been given a small set of retrieved text chunks from the document.\n"
        "Answer the user's question using only those chunks. Cite the page numbers "
        "(e.g. \"[p. 32]\") for every claim. If the chunks do not contain the answer, "
        "say so explicitly — do not guess."
    )


def answer(
    query: str,
    k: int = config.RAG_TOP_K,
    faiss_path: Path | None = None,
    chunks_path: Path | None = None,
    doc_name: str | None = None,
) -> dict:
    """Run the full trad-RAG pipeline: retrieve → generate. Returns trace + answer."""
    hits = retrieve(query, k=k, faiss_path=faiss_path, chunks_path=chunks_path)
    context_blocks = []
    for h in hits:
        page_label = f"p. {h['page_start']}" if h["page_start"] == h["page_end"] else f"pp. {h['page_start']}-{h['page_end']}"
        context_blocks.append(f"[{page_label}, similarity={h['score']:.3f}]\n{h['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    user_prompt = f"Question: {query}\n\nRetrieved chunks:\n{context}\n\nAnswer with citations."
    if doc_name is None:
        doc_name = config.DOC_NAME
    result = complete(
        model=config.ANSWER_MODEL,
        messages=[
            {"role": "system", "content": _rag_system_prompt(doc_name)},
            {"role": "user", "content": user_prompt},
        ],
    )
    return {"hits": hits, "answer": result["content"], "thinking": result["thinking"]}
