"""Traditional vector RAG: chunk PDF, embed with sentence-transformers, query via FAISS."""
import json
from pathlib import Path

import faiss
import numpy as np
import pymupdf
import tiktoken
from sentence_transformers import SentenceTransformer

from . import config
from .llm import complete


_enc = tiktoken.get_encoding("cl100k_base")
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def pdf_to_chunks(pdf_path: Path) -> list[dict]:
    """Split PDF into ~CHUNK_TOKENS chunks with overlap, page-aware."""
    doc = pymupdf.open(pdf_path)
    pages = [(i + 1, page.get_text() or "") for i, page in enumerate(doc)]
    doc.close()

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


def build_faiss_index(chunks: list[dict]) -> None:
    embedder = get_embedder()
    embeddings = embedder.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_PATH))
    with open(config.CHUNKS_PATH, "w") as f:
        json.dump(chunks, f)
    print(f"FAISS index: {embeddings.shape[0]} vectors, dim={embeddings.shape[1]}")


def load_index() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(config.FAISS_PATH))
    with open(config.CHUNKS_PATH) as f:
        chunks = json.load(f)
    return index, chunks


def retrieve(query: str, k: int = config.RAG_TOP_K) -> list[dict]:
    """Top-k chunks with similarity scores."""
    index, chunks = load_index()
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


RAG_SYSTEM_PROMPT = """You are a financial-research assistant answering questions about Tesla's 2024 10-K filing.
You have been given a small set of retrieved text chunks from the document.
Answer the user's question using only those chunks. Cite the page numbers (e.g. "[p. 32]") for every claim.
If the chunks do not contain the answer, say so explicitly — do not guess."""


def answer(query: str, k: int = config.RAG_TOP_K) -> dict:
    """Run the full trad-RAG pipeline: retrieve → generate. Returns trace + answer."""
    hits = retrieve(query, k=k)
    context_blocks = []
    for h in hits:
        page_label = f"p. {h['page_start']}" if h["page_start"] == h["page_end"] else f"pp. {h['page_start']}-{h['page_end']}"
        context_blocks.append(f"[{page_label}, similarity={h['score']:.3f}]\n{h['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    user_prompt = f"Question: {query}\n\nRetrieved chunks:\n{context}\n\nAnswer with citations."
    result = complete(
        model=config.ANSWER_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return {"hits": hits, "answer": result["content"], "thinking": result["thinking"]}
