"""Query-time entrypoints for collections.

Each function mirrors the per-doc answer() in its respective pipeline module
but pulls from collection-level artifacts and surfaces per-doc attribution
in the response trace.
"""
from __future__ import annotations
import concurrent.futures as _cf
import json
import os
import re
from pathlib import Path

import numpy as np

from . import collections as _col, config, library
from .llm import complete


# ─── RAG over collection ──────────────────────────────────────────────────────

def rag_answer(query: str, faiss_path: Path, chunks_path: Path,
               doc_name: str | None = None, k: int = None) -> dict:
    """RAG over the merged FAISS index. Same shape as rag_pipeline.answer
    but the citations include `[doc_name, p. N]` rather than just page,
    and the trace surfaces which docs contributed."""
    import faiss
    from .embed import get_embedder

    k = k or config.RAG_TOP_K
    index = faiss.read_index(str(faiss_path))
    with open(chunks_path) as f:
        chunks = json.load(f)

    embedder = get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q_emb, k)

    hits: list[dict] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0:
            continue
        c = dict(chunks[idx])
        c["score"] = float(score)
        hits.append(c)

    blocks: list[str] = []
    for h in hits:
        dn = h.get("doc_name") or "?"
        pp = (f"p. {h['page_start']}" if h.get("page_start") == h.get("page_end")
              else f"pp. {h.get('page_start')}-{h.get('page_end')}")
        blocks.append(f"[{dn}, {pp}, sim={h['score']:.3f}]\n{h['text']}")
    context = "\n\n---\n\n".join(blocks)

    sys_prompt = (
        f"You are a research assistant answering questions about the **{doc_name}** "
        "collection (multiple documents).\n"
        "You have been given retrieved text chunks from across the collection.\n"
        "Answer the user's question using only those chunks. Cite the document and "
        "page (e.g. \"[Doc Name, p. 32]\") for every claim. If multiple documents "
        "contribute, attribute each claim to its source document. If the chunks "
        "do not contain the answer, say so explicitly — do not guess."
    )
    user_prompt = f"Question: {query}\n\nRetrieved chunks:\n{context}\n\nAnswer with citations."

    result = complete(
        model=config.ANSWER_MODEL,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
    )

    # Per-doc contribution summary for the UI
    by_doc: dict[str, int] = {}
    for h in hits:
        by_doc[h.get("doc_name") or "?"] = by_doc.get(h.get("doc_name") or "?", 0) + 1

    return {
        "answer": result["content"],
        "thinking": result["thinking"],
        "hits": hits,
        "doc_contributions": by_doc,
    }


# ─── PageIndex over collection: router + per-doc walks + aggregator ──────────

ROUTER_PROMPT = """\
You are routing a user question to the most relevant documents in a collection.

For each document below, output a confidence score from 0.0 to 1.0 reflecting
how likely it is to contain information that answers the question.

Return STRICT JSON in this exact shape:
{{
  "reasoning": "<one sentence on which docs are most relevant and why>",
  "scores": [
    {{"doc_id": "<id>", "confidence": <float in [0,1]>}},
    ...
  ]
}}

User question: {query}

Documents:
{doc_list}

Return only the JSON object."""


AGGREGATOR_PROMPT = """\
You produced per-document answers to the same user question, each one based on
content from a single document in a collection. Combine them into a single,
concise, non-redundant final answer.

Rules:
- If only one document had relevant content, present that answer directly,
  with the document attribution preserved (e.g. "[<doc name>, p. N]").
- If multiple documents had relevant content, integrate them — attribute each
  claim to its source document inline.
- If multiple documents disagree, say so explicitly and present each view.
- If a per-doc answer says the doc didn't contain the answer, do NOT carry
  that forward — only consume useful content.

User question: {query}

Per-document answers:
{per_doc}

Final answer:"""


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = fence.group(1) if fence else text
    start, end = blob.find("{"), blob.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(blob[start:end + 1])
    except json.JSONDecodeError:
        return {}


def pageindex_answer_collection(
    query: str,
    doc_summaries_path: Path,
    doc_ids: list[str],
    doc_name: str | None = None,
) -> dict:
    """Multi-doc PageIndex: route → walk each above-threshold doc → aggregate.

    Quality-over-speed:
      - Threshold defaults to 0.4 (COLLECTION_PI_ROUTER_THRESHOLD).
      - If zero docs clear the threshold, walk ALL docs rather than guess.
      - Per-doc walks run in parallel.
      - Aggregator receives full reasoning traces, not just answers, so it
        can resolve contradictions instead of just concatenating.
    """
    from . import pageindex_agent

    if not doc_summaries_path.exists():
        return {"answer": "(Collection PageIndex router not built.)", "thinking": "",
                "router": {}, "per_doc": [], "selected_doc_ids": []}
    summaries = json.loads(doc_summaries_path.read_text())

    # --- 1. router: per-doc confidence -------------------------------------
    doc_lines = []
    for did in doc_ids:
        s = summaries.get(did, {})
        topics = ", ".join(s.get("topics", []) or [])
        doc_lines.append(
            f"- doc_id: {did}\n"
            f"  name: {s.get('name', '?')}\n"
            f"  summary: {s.get('summary', '(no summary)')}\n"
            f"  topics: {topics}"
        )
    router_prompt = ROUTER_PROMPT.format(query=query, doc_list="\n".join(doc_lines))
    router_result = complete(
        model=config.RETRIEVE_MODEL,
        messages=[{"role": "user", "content": router_prompt}],
        temperature=0.0,
    )
    router_json = _extract_json(router_result["content"])
    scores_in: list[dict] = router_json.get("scores", []) if isinstance(router_json, dict) else []
    router_reasoning = (router_json.get("reasoning", "") if isinstance(router_json, dict) else "").strip()

    confidences: dict[str, float] = {}
    for row in scores_in:
        if not isinstance(row, dict):
            continue
        did = row.get("doc_id")
        try:
            conf = float(row.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        if did in summaries:
            confidences[did] = max(0.0, min(1.0, conf))

    threshold = float(os.environ.get("COLLECTION_PI_ROUTER_THRESHOLD", "0.4"))
    selected = [did for did, c in confidences.items() if c >= threshold]
    fallback_used = False
    if not selected:
        selected = list(doc_ids)
        fallback_used = True

    # --- 2. parallel per-doc walks -----------------------------------------
    def _walk(did: str) -> dict:
        meta = library.get_document(did)
        try:
            return {
                "doc_id": did,
                "doc_name": meta.get("name") or did,
                "confidence": confidences.get(did, 0.0),
                "result": pageindex_agent.answer(
                    query=query,
                    workspace=library.pageindex_workspace(did),
                    doc_id_file=library.pageindex_doc_id_file(did),
                    doc_name=meta.get("name") or did,
                ),
                "error": None,
            }
        except Exception as e:
            return {
                "doc_id": did,
                "doc_name": meta.get("name") or did,
                "confidence": confidences.get(did, 0.0),
                "result": None,
                "error": f"{type(e).__name__}: {e}",
            }

    per_doc: list[dict] = []
    max_workers = min(4, max(1, len(selected)))
    with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in ex.map(_walk, selected):
            per_doc.append(r)

    # --- 3. aggregator ------------------------------------------------------
    blocks = []
    for r in per_doc:
        if r["error"]:
            blocks.append(f"[{r['doc_name']} (conf={r['confidence']:.2f})] ERROR: {r['error']}")
            continue
        res = r["result"] or {}
        ans = (res.get("answer") or "").strip() or "(empty)"
        plan_step = next((s for s in res.get("steps", []) if s.get("name") == "plan"), {})
        plan_reason = plan_step.get("reasoning", "")
        blocks.append(
            f"[{r['doc_name']} (conf={r['confidence']:.2f}, pages read: {res.get('pages_read', '')})]\n"
            f"  plan reasoning: {plan_reason}\n"
            f"  answer: {ans}"
        )
    agg_prompt = AGGREGATOR_PROMPT.format(query=query, per_doc="\n\n".join(blocks))
    agg = complete(
        model=config.ANSWER_MODEL,
        messages=[{"role": "user", "content": agg_prompt}],
        temperature=0.0,
    )

    return {
        "answer": agg["content"],
        "thinking": agg["thinking"],
        "router": {
            "reasoning": router_reasoning,
            "confidences": confidences,
            "threshold": threshold,
            "fallback_used": fallback_used,
            "model_output": router_result["content"],
        },
        "per_doc": per_doc,
        "selected_doc_ids": selected,
    }


# ─── GraphRAG over collection (re-uses graphrag_pipeline.answer) ──────────────

def graphrag_answer_collection(query: str, graphrag_dir: Path,
                                doc_name: str | None = None) -> dict:
    """The collection's merged property graph + community summaries live in
    the same file layout as a single-doc GraphRAG index. Defer to the
    existing answer() — it already handles the empty-communities diagnostic
    path."""
    from . import graphrag_pipeline as gp
    return gp.answer(query=query, graphrag_dir=graphrag_dir, doc_name=doc_name)
