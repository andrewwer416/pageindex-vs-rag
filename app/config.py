"""Configuration. All LLM-related settings come from environment variables
so the same image runs against any OpenAI-compatible or Ollama endpoint.

Required env vars when running:
  LLM_API_BASE     e.g. http://localhost:11434 (Ollama) or http://localhost:8000/v1 (OpenAI-compat)
  INDEX_MODEL      model string in LiteLLM format, e.g. "ollama_chat/qwen3:14b" or "openai/gemma-3-27b-it"
  ANSWER_MODEL     model for final answer generation in both pipelines
  RETRIEVE_MODEL   model for PageIndex's retrieval-time reasoning (a thinking model shines here)

See .env.example for a starter template.
"""
import os
from pathlib import Path

# In the Docker container the project is at /app; locally it's the repo root.
ROOT = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(ROOT / "index")))

PDF_PATH = DATA_DIR / "tsla-10k-2024.pdf"
DOC_NAME = os.getenv("DOC_NAME", "Tesla 10-K (FY 2024)")

PAGEINDEX_WORKSPACE = INDEX_DIR / "pageindex_workspace"
PAGEINDEX_DOC_ID_FILE = INDEX_DIR / "pageindex_doc_id.txt"

FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

# Single endpoint for all LLM calls. LiteLLM routes based on the model prefix
# (openai/ → uses LLM_API_BASE; ollama_chat/ → uses OLLAMA_API_BASE).
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:11434")
os.environ.setdefault("OLLAMA_API_BASE", LLM_API_BASE)
os.environ.setdefault("OPENAI_API_BASE", LLM_API_BASE)
# A dummy key — many local OpenAI-compatible servers don't enforce auth but LiteLLM expects something set.
os.environ.setdefault("OPENAI_API_KEY", os.getenv("LLM_API_KEY", "not-needed"))

INDEX_MODEL = os.getenv("INDEX_MODEL", "ollama_chat/qwen3:14b")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "ollama_chat/qwen3:8b")
RETRIEVE_MODEL = os.getenv("RETRIEVE_MODEL", "ollama_chat/deepseek-r1:14b")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "450"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

PAGEINDEX_MAX_STEPS = 6
