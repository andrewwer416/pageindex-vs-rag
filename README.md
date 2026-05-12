# PageIndex vs Traditional RAG

A side-by-side demo of two document-retrieval architectures on the same PDF:

- **Traditional RAG** — chunk → embed → FAISS nearest-neighbor → answer.
- **PageIndex** — build a hierarchical table-of-contents tree once at indexing time, then at query time let an LLM reason over the tree to pick which sections to read, then read only those pages.

The point isn't "X wins." Each architecture wins on a different question shape. The UI shows the reasoning trace from both pipelines side-by-side so the difference is visible.

Default document: **Tesla 10-K (FY 2024)** — 227 pages, fetched from SEC EDGAR.

## How it works

```
┌──────────────────────────────────────────────────┐    ┌─────────────────────────────────┐
│  Streamlit UI                                    │    │  LLM backend                    │
│  ┌────────────────────┬─────────────────────────┐│    │                                 │
│  │ Traditional RAG    │ PageIndex               ││    │  Anything LiteLLM speaks:       │
│  │                    │                         ││    │   • Ollama (local)              │
│  │ FAISS top-k        │ 1. Plan: read tree,     │├───►│   • vLLM / llama.cpp server     │
│  │  → chunks          │    pick page ranges     ││    │     (OpenAI-compatible)         │
│  │  → LLM answer      │ 2. Read: fetch pages    ││    │   • LM Studio                   │
│  │                    │ 3. Answer: cite pages   ││    │   • OpenAI / Anthropic / Groq   │
│  └────────────────────┴─────────────────────────┘│    │                                 │
└──────────────────────────────────────────────────┘    └─────────────────────────────────┘
```

All settings live in env vars — point at any backend by editing `.env`.

## Quick start

### 1. Clone and configure

```bash
git clone https://github.com/<you>/pageindex-vs-rag.git
cd pageindex-vs-rag
cp .env.example .env
# Edit .env: LLM_API_BASE, INDEX_MODEL, ANSWER_MODEL, RETRIEVE_MODEL.
# See .env.example for examples of Ollama, vLLM/llama.cpp, and hosted OpenAI.
```

### 2. Fetch the source PDF

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_tesla_10k.py
```

### 3. Build the indices

The traditional RAG index (FAISS) is committed pre-built. The PageIndex tree is generated locally:

```bash
PYTHONPATH=. python scripts/build_index.py pageindex   # slow: many LLM calls
# or for a quick hand-crafted tree (skips PageIndex's LLM tree-builder):
PYTHONPATH=. python scripts/build_pageindex_manual.py
```

### 4. Run

```bash
docker compose up -d --build      # exposes http://localhost:8501
# or for local dev with hot reload:
PYTHONPATH=. streamlit run app/app.py
```

## LLM backend choices

The default `.env.example` targets Ollama. To swap backends, change the model prefix and `LLM_API_BASE`:

| Backend | `LLM_API_BASE` | model string |
|---|---|---|
| Ollama (local) | `http://localhost:11434` | `ollama_chat/<model>:tag` |
| vLLM / llama.cpp server | `http://localhost:8000/v1` | `openai/<model_name>` |
| LM Studio | `http://localhost:1234/v1` | `openai/<model_name>` |
| OpenAI | `https://api.openai.com/v1` | `openai/gpt-4o` |
| Groq | `https://api.groq.com/openai/v1` | `openai/llama-3.3-70b-versatile` |

If the LLM runs on the same host as the container, use `LLM_API_BASE=http://host.docker.internal:11434` — the compose file maps that to the host gateway.

## Public deployment

For a real subdomain via Traefik + Cloudflare tunnel, see [`deploy/README.md`](deploy/README.md). The `deploy/*.yaml.example` files show two patterns (label-driven and file-driven Traefik) with placeholder hostnames and tunnel IDs.

## Layout

```
pageindex-vs-rag/
├── app/
│   ├── config.py            paths, model names, all env-driven
│   ├── llm.py               LiteLLM wrapper (thinking-tag handling)
│   ├── rag_pipeline.py      chunking, embedding, FAISS retrieval, answer
│   ├── pageindex_agent.py   2-step tree-walk agent (plan → answer)
│   ├── pageindex/           local copy of VectifyAI/PageIndex with patches
│   └── app.py               Streamlit UI
├── scripts/
│   ├── fetch_tesla_10k.py            download + render PDF from SEC EDGAR
│   ├── build_index.py                build FAISS index + PageIndex tree
│   └── build_pageindex_manual.py     hand-craft tree (for when auto-indexing flops)
├── data/                    PDF lives here (gitignored)
├── index/                   generated indices
│   ├── faiss.index          ← committed (pre-built)
│   ├── chunks.json          ← committed (pre-built)
│   └── pageindex_workspace/ ← generated locally
├── deploy/                  Traefik + Cloudflare examples
├── Dockerfile
├── compose.yaml
├── requirements.txt
└── .env.example
```

## Caveats

**PageIndex's auto-indexing is finicky with small open-source models.** The library's prompt chain (~50 LLM calls in series, each demanding strictly-formatted JSON) was designed against GPT-4-class models. With qwen3-8B and even qwen3-14B on consumer hardware, individual calls fail often enough that the end-to-end pipeline breaks. Patches in `app/pageindex/`:

- inject `/no_think` for qwen3 + strip `<think>…</think>` from content
- bump `num_ctx` (Ollama) and timeout for long prompts
- relax `verify_toc`'s page-coverage threshold (default fails when the model can't extrapolate page numbers for back-half items)
- short-circuit `process_no_toc` (the broken-with-small-models fallback) — return partial TOC instead of recursing
- disable `process_large_node_recursively` — large sections stay as single nodes

If auto-indexing still flops on your hardware, `scripts/build_pageindex_manual.py` parses section headings out of the PDF directly and produces a tree that retrieval uses identically. Not "full PageIndex" but the retrieval-side reasoning — the actually interesting half — works the same way.

## Credits

- [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — the document-tree indexing approach.
- [LiteLLM](https://github.com/BerriAI/litellm) — provider-agnostic LLM client.
- [Streamlit](https://streamlit.io/) — UI framework.
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) — embedding model.

## License

MIT
