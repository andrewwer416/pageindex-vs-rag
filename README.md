# PageIndex vs Traditional RAG

A side-by-side demo of two document-retrieval architectures on a document you upload:

- **Traditional RAG** — chunk → embed → FAISS nearest-neighbor → answer.
- **PageIndex** — build a hierarchical table-of-contents tree once at indexing time, then at query time let an LLM reason over the tree to pick which sections to read, then read only those pages.

The point isn't "X wins." Each architecture wins on a different question shape. The UI shows the reasoning trace from both pipelines side by side so the difference is visible, plus the extracted document hierarchy.

## How it works

```
┌──────────────────────────────────────────────────┐    ┌─────────────────────────────────┐
│  Streamlit UI                                    │    │  LLM backend                    │
│  ┌────────────────────┬─────────────────────────┐│    │                                 │
│  │ Traditional RAG    │ PageIndex               ││    │  Anything OpenAI-compatible:    │
│  │                    │                         ││    │   • Ollama (/v1 shim)           │
│  │ FAISS top-k        │ 1. Plan: read tree,     │├───►│   • vLLM / llama.cpp server     │
│  │  → chunks          │    pick page ranges     ││    │   • LM Studio                   │
│  │  → LLM answer      │ 2. Read: fetch pages    ││    │   • OpenAI / Anthropic / Groq   │
│  │                    │ 3. Answer: cite pages   ││    │   • corporate gateways          │
│  └────────────────────┴─────────────────────────┘│    │     (custom CA, headers, mTLS)  │
└──────────────────────────────────────────────────┘    └─────────────────────────────────┘
```

All settings live in env vars — point at any backend by editing `.env`.

## Quick start

### 1. Clone and configure

```bash
git clone https://github.com/<you>/pageindex-vs-rag.git
cd pageindex-vs-rag
cp .env.example .env
# Edit .env: set LLM_API_BASE + MODEL (one model serves all three roles by default).
# Worked examples for Ollama, vLLM, llama.cpp, LM Studio, OpenAI, Groq, and internal
# corporate gateways are inline in .env.example.
```

### 2. Install + run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. streamlit run app/app.py
```

…or with Docker:

```bash
docker compose up -d --build      # exposes http://localhost:8501
```

### 3. Upload + query

The Streamlit app has two pages in the sidebar:

- **app** — Library page. Upload a PDF, Markdown, or plain-text document. Both indices (FAISS for RAG + PageIndex tree) are built once at upload time; expect "many LLM calls" worth of time (tens of minutes on local hardware). Live progress is shown.
- **Compare** — pick an indexed document, see the extracted hierarchy tree, run queries through both pipelines side by side.

## LLM backend choices

| Backend | `LLM_API_BASE` | `MODEL` |
|---|---|---|
| Ollama (local) | `http://localhost:11434/v1` | `qwen3:14b` |
| vLLM / llama.cpp server | `http://localhost:8000/v1` | exact model id from `/models` |
| LM Studio | `http://localhost:1234/v1` | exact model id from `/models` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Internal corporate gateway | `https://<host>/.../v1` | what the server expects |

For internal/enterprise endpoints, `LLM_API_KEY_HEADER`, `LLM_CA_BUNDLE`, `LLM_CLIENT_CERT`, and `LLM_HEADERS` cover custom-auth / private-CA / mTLS setups. See `.env.example` for examples.

If the LLM runs on the same host as the container, use `LLM_API_BASE=http://host.docker.internal:11434/v1` — the compose file maps that to the host gateway.

### Embeddings

`EMBED_PROVIDER` controls how the FAISS pipeline gets its vectors:
- `sentence-transformers` (default) — local HuggingFace model.
- `openai` — call your endpoint's `/v1/embeddings`. Reuses LLM auth by default; override with `EMBED_API_BASE` / `EMBED_API_KEY_HEADER` / `EMBED_CA_BUNDLE` if your embedder lives elsewhere.

### Vision

Optional. Set `VISION_ENABLED=true` and `VISION_MODEL=<your-vision-model>` in `.env` to send each PDF image to a vision-language model at index time. The description is injected into the page text so retrieval finds the image's content. Useful for scanned PDFs, charts, diagrams, and screenshots. All `VISION_*` settings fall back to the corresponding `LLM_*`, so a single endpoint serving chat + vision needs no extra config beyond `VISION_ENABLED=true`.

## Public deployment

For a real subdomain via Traefik + Cloudflare tunnel, see [`deploy/README.md`](deploy/README.md). The `deploy/*.yaml.example` files show two patterns (label-driven and file-driven Traefik) with placeholder hostnames and tunnel IDs.

**Library is currently shared across all visitors.** This app does not yet have per-user isolation — anyone hitting the public URL sees and can delete each other's uploads. Use for personal hosting or behind your own auth; multi-tenant support is on the roadmap.

## Layout

```
pageindex-vs-rag/
├── app/
│   ├── app.py               Library + Upload entry point
│   ├── pages/
│   │   └── 2_Compare.py     Side-by-side comparison + hierarchy view
│   ├── config.py            env-driven config
│   ├── llm.py               openai-SDK wrapper (handles custom CA / headers / mTLS)
│   ├── embed.py             local sentence-transformers OR API embedder
│   ├── vision.py            optional VLM image describer
│   ├── extraction.py        table-aware PDF extraction (markdown tables inline)
│   ├── library.py           per-doc persistence under index/library/
│   ├── rag_pipeline.py      chunking, embedding, FAISS retrieval, answer
│   ├── pageindex_agent.py   2-step tree-walk agent (plan → answer)
│   └── pageindex/           local copy of VectifyAI/PageIndex with patches
├── deploy/                  Traefik + Cloudflare examples
├── Dockerfile
├── compose.yaml
├── requirements.txt
└── .env.example
```

## Caveats

**PageIndex's LLM-driven auto-indexing isn't always reliable with small open models.**
The library's prompt chain (~50 LLM calls in series, each demanding strictly-formatted JSON) was designed against GPT-4-class models. On smaller models, individual calls fail often enough that some docs end up with sparse trees. Patches in `app/pageindex/`:

- inject `/no_think` for qwen3 + strip `<think>…</think>` from content
- bump `num_ctx` and per-call timeout for Ollama
- relax `verify_toc`'s page-coverage threshold so it doesn't early-exit on documents where back-half items can't be extrapolated
- short-circuit the `process_no_toc` fallback — return whatever partial TOC we have instead of recursing into a path small models can't satisfy
- disable `process_large_node_recursively` — large sections stay as single nodes rather than being subdivided

If the PageIndex tree ends up sparse, the document still gets a `partial` status and remains queryable on the Traditional-RAG side. For best PageIndex results, point `INDEX_MODEL` at a GPT-4-class endpoint just for the one-time indexing run.

## Credits

- [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — the document-tree indexing approach.
- [Streamlit](https://streamlit.io/) — UI framework.
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) — default embedding model.

## License

MIT
