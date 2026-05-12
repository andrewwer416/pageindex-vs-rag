"""One-time: build both PageIndex tree and FAISS vector index for the Tesla 10-K."""
import sys
import time
from pathlib import Path

# Run from project root: PYTHONPATH=. python scripts/build_index.py
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config
from app.rag_pipeline import build_faiss_index, pdf_to_chunks
from app.pageindex_agent import get_client


def build_pageindex() -> str:
    client = get_client()
    print(f"Indexing {config.PDF_PATH} with PageIndex (model={config.INDEX_MODEL})...")
    t0 = time.time()
    doc_id = client.index(str(config.PDF_PATH))
    print(f"PageIndex done in {time.time() - t0:.1f}s. doc_id={doc_id}")
    config.PAGEINDEX_DOC_ID_FILE.write_text(doc_id)
    return doc_id


def build_rag() -> None:
    print(f"Chunking {config.PDF_PATH} ({config.CHUNK_TOKENS}-token chunks, {config.CHUNK_OVERLAP_TOKENS} overlap)...")
    t0 = time.time()
    chunks = pdf_to_chunks(config.PDF_PATH)
    print(f"Chunked into {len(chunks)} chunks in {time.time() - t0:.1f}s")
    print(f"Embedding with {config.EMBED_MODEL}...")
    t0 = time.time()
    build_faiss_index(chunks)
    print(f"FAISS index built in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    args = set(sys.argv[1:])
    if not args or "rag" in args:
        build_rag()
    if not args or "pageindex" in args:
        build_pageindex()
    print("All done.")
