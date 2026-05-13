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
# Edit .env: set LLM_API_BASE + MODEL (one model serves all three roles by default).
# Worked examples for Ollama, vLLM, llama.cpp, LM Studio, OpenAI, and Groq inline in the file.
```

### 2. Fetch the source PDF

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_tesla_10k.py
```

### 3. Build the indices

All indices (FAISS + PageIndex tree) are committed pre-built against the
default Tesla 10-K, so you can skip this step entirely if you're not changing
the document. To rebuild them (after swapping the source PDF):

```bash
# FAISS: chunk + embed (fast, ~30s, no LLM calls)
PYTHONPATH=. python scripts/build_index.py rag

# PageIndex tree — RECOMMENDED PATH: hand-craft from section headings
# (works reliably with any model that can write a one-line summary).
PYTHONPATH=. python scripts/build_pageindex_manual.py
```

There's also a full LLM-driven indexer (`scripts/build_index.py pageindex`)
that calls PageIndex's prompt chain end-to-end. **It only works well with
GPT-4-class models.** With smaller open-source models the chain accumulates
per-call errors and the resulting tree typically has hollowed-out sections
even with the patches in this repo. See "Caveats" below.

### 4. Run

```bash
docker compose up -d --build      # exposes http://localhost:8501
# or for local dev with hot reload:
PYTHONPATH=. streamlit run app/app.py
```

The app is a multi-page Streamlit app. The sidebar has:

- **app** (main page) — the side-by-side comparison on the bundled Tesla 10-K.
- **Upload Document** — drop in your own PDF / Markdown / TXT; both indices are built once at upload time.
- **Compare Custom** — pick a doc from your library and run the same side-by-side comparison on questions you write.

PageIndex indexing of a new document takes "many LLM calls" worth of time — typically tens of minutes on local hardware. The upload page blocks while it runs and shows live progress.

### Tables and images

- **Tables** in PDFs are detected via pymupdf and rendered as markdown inline in each page's text. Both pipelines get structured tables; no extra config needed.
- **Images** (scanned pages, charts, diagrams, screenshots) are *optional*. Set `VISION_ENABLED=true` and `VISION_MODEL=<your-vision-model>` in `.env` to send each image to a vision-language model at index time. The description is injected into the page text so retrieval finds the image's content. All `VISION_*` settings fall back to the corresponding `LLM_*` value, so a single endpoint serving chat + vision needs no extra config beyond `VISION_ENABLED=true`.

## LLM backend choices

The default `.env.example` targets Ollama. To swap backends, change the model prefix and `LLM_API_BASE`:

| Backend | `LLM_API_BASE` | `MODEL` |
|---|---|---|
| Ollama (local) | `http://localhost:11434/v1` | `qwen3:14b` |
| vLLM / llama.cpp server | `http://localhost:8000/v1` | exact model id from `/models` |
| LM Studio | `http://localhost:1234/v1` | exact model id from `/models` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Internal corporate gateway | `https://<host>/.../v1` | what the server expects |

For internal/enterprise endpoints, `LLM_API_KEY_HEADER`, `LLM_CA_BUNDLE`, `LLM_CLIENT_CERT`, and `LLM_HEADERS` cover the usual custom-auth / private-CA / mTLS setups. See `.env.example` for examples.

If the LLM runs on the same host as the container, use `LLM_API_BASE=http://host.docker.internal:11434/v1` — the compose file maps that to the host gateway.

### Embeddings

`EMBED_PROVIDER` controls how the FAISS pipeline gets its vectors:
- `sentence-transformers` (default) — local HuggingFace model.
- `openai` — call your endpoint's `/v1/embeddings`. Reuses LLM auth by default; override with `EMBED_API_BASE` / `EMBED_API_KEY_HEADER` / `EMBED_CA_BUNDLE` if your embedder lives elsewhere.

**Important:** the committed `index/faiss.index` was built with `BAAI/bge-small-en-v1.5` (384-dim). Switching `EMBED_MODEL` or `EMBED_PROVIDER` means rebuilding it: `PYTHONPATH=. python scripts/build_index.py rag`.

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
├── index/                   pre-built indices
│   ├── faiss.index          ← committed
│   ├── chunks.json          ← committed
│   └── pageindex_workspace/ ← committed (hand-crafted tree against the default 10-K)
├── deploy/                  Traefik + Cloudflare examples
├── Dockerfile
├── compose.yaml
├── requirements.txt
└── .env.example
```

## Caveats

**PageIndex's LLM-driven auto-indexing isn't viable with small open models — even with patches.**
The library's prompt chain (~50 LLM calls in series, each demanding strictly-formatted JSON) was designed against GPT-4-class models. Empirical results on this hardware (RTX 2080, qwen3-14B Q4 at 32K ctx, ~2 hours):

- TOC parsing, page-offset calculation, and `verify_toc` all completed.
- But ~60% of sections were dropped from the resulting tree — items 1, 1A, 1B, 1C, 2-8 (pp.8-181) didn't make it in.
- The tree also picked up a hallucinated top-level "Preface" node and gave near-identical templated summaries to multiple distinct sections.

Patches in `app/pageindex/` mitigate the crashes (no hangs, no exceptions) but
they can't make a small model reason about 100K tokens of body text reliably enough to build a faithful tree. The patches present:

- inject `/no_think` for qwen3 + strip `<think>…</think>` from content
- bump `num_ctx` and per-call timeout for Ollama
- relax `verify_toc`'s page-coverage threshold so it doesn't early-exit on documents where back-half items can't be extrapolated from the toc_check_page_num window
- short-circuit the `process_no_toc` fallback — return whatever partial TOC we have instead of recursing into a path small models can't satisfy
- disable `process_large_node_recursively` — large sections stay as single nodes rather than being subdivided

**`scripts/build_pageindex_manual.py` is the recommended path on consumer hardware.**
It parses ALL-CAPS section headings out of the PDF (e.g. `ITEM 1A. RISK FACTORS`), builds the tree directly with correct physical page numbers, then makes one ~3000-char LLM call per section to generate a one-line summary. Those short structured-output calls succeed reliably even with 8B models.

The retrieval-side reasoning — the actually interesting half of PageIndex — works identically against either tree. If you want the LLM-driven indexer for the full experience, point `INDEX_MODEL` at `openai/gpt-4o-mini` or similar; a one-time indexing call against a hosted API runs in ~5 minutes for ~$0.50.

## Credits

- [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — the document-tree indexing approach.
- [LiteLLM](https://github.com/BerriAI/litellm) — provider-agnostic LLM client.
- [Streamlit](https://streamlit.io/) — UI framework.
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) — embedding model.

## License

MIT
