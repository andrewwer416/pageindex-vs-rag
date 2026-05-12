"""Configuration. All LLM-related settings come from environment variables.

The app talks to a single OpenAI-compatible endpoint via the openai Python
SDK — including local servers that need a custom CA bundle, custom headers,
or an alternate authentication scheme.

Required env vars when running:
  LLM_API_BASE   full URL ending in /v1, e.g. http://localhost:11434/v1
                 (Ollama's OpenAI shim) or https://internal.example.com/.../v1
  MODEL          model name your server expects in the chat-completions request
                 (run `curl $LLM_API_BASE/models` to discover). Used for all
                 three roles below unless individually overridden.

Optional:
  LLM_API_KEY      bearer token (any non-empty string if server doesn't auth)
  LLM_HEADERS      extra request headers: JSON ('{"X-Foo":"bar"}') or "K1:V1;K2:V2"
  LLM_CA_BUNDLE    path to a CA-cert PEM for verifying the server's TLS cert
  LLM_CLIENT_CERT  for mTLS: "cert.pem:key.pem" or a single combined .pem
  LLM_VERIFY_SSL   set to "false" to skip server-cert verification (testing only)
  LLM_TIMEOUT      seconds per call; default 1800

  INDEX_MODEL      model used to build the PageIndex tree (1× heavy run)
  ANSWER_MODEL     model used to generate the final answer in both pipelines
  RETRIEVE_MODEL   model used by PageIndex to reason over the tree at query time
                   (a "thinking" model surfaces a richer trace here)

See .env.example for ready-to-paste templates.
"""
import os
import sys
from pathlib import Path

# On Windows, the default ProactorEventLoop causes spurious
# "RuntimeError: Event loop is closed" warnings at shutdown when AsyncOpenAI /
# httpx connections are garbage-collected after asyncio.run() closes the loop.
# SelectorEventLoop doesn't have this issue and we don't use subprocesses, so
# the trade-off is free.
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load .env BEFORE any env-var reads below. Looks first for an explicit DOTENV_PATH,
# then for .env in the project root regardless of where the process was started.
try:
    from dotenv import load_dotenv
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _dotenv_path = os.getenv("DOTENV_PATH") or str(_PROJECT_ROOT / ".env")
    load_dotenv(_dotenv_path, override=False)
except ImportError:
    pass  # dotenv is optional; env vars set in the shell still work

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

# The base URL of the OpenAI-compatible endpoint. Should end in /v1 for most
# servers; e.g. Ollama's shim is at http://localhost:11434/v1.
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")

# Single model used everywhere by default; override any individual role as needed.
MODEL = os.getenv("MODEL", "qwen3:14b")
INDEX_MODEL = os.getenv("INDEX_MODEL", MODEL)
ANSWER_MODEL = os.getenv("ANSWER_MODEL", MODEL)
RETRIEVE_MODEL = os.getenv("RETRIEVE_MODEL", MODEL)

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "450"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

PAGEINDEX_MAX_STEPS = 6
