"""Per-pipeline collection builders.

Each pipeline produces a *collection-level* artifact built from the
already-built per-doc indices, so the heavy single-doc work (embedding,
tree-walk indexing, entity extraction) is not repeated:

  build_rag                 — Phase 2: concat FAISS vectors + chunks
  build_pageindex_router    — Phase 3: doc-level summaries for routing
  build_graphrag            — Phase 4: LLM-graded entity resolution +
                                       merged graph + new communities
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

import numpy as np

from . import collections as _col, library


# ─── Phase 2: RAG merge ───────────────────────────────────────────────────────

def build_rag(col_id: str, doc_ids: list[str]) -> dict:
    """Concatenate per-doc FAISS vectors + chunk rows into a single
    collection-level FAISS index. Each output chunk row carries `doc_id`
    and `doc_name` so the answer side can attribute citations across
    multiple documents.

    We do NOT re-embed — the per-doc FAISS already holds the vectors. We
    reconstruct them via faiss.IndexFlatIP.reconstruct_n() and rebuild
    one IndexFlatIP over the union.
    """
    import faiss

    all_vecs: list[np.ndarray] = []
    all_chunks: list[dict] = []
    n_docs = 0
    vec_dim: int | None = None

    for did in doc_ids:
        meta = library.get_document(did)
        doc_name = meta.get("name") or did
        faiss_p = library.faiss_path(did)
        chunks_p = library.chunks_path(did)
        if not faiss_p.exists() or not chunks_p.exists():
            # Skip docs that don't have a RAG index (e.g. failed during
            # single-doc indexing). The doc still counts toward member
            # but contributes 0 vectors.
            continue

        idx = faiss.read_index(str(faiss_p))
        n_vec = idx.ntotal
        if n_vec == 0:
            continue

        if vec_dim is None:
            vec_dim = idx.d
        elif idx.d != vec_dim:
            raise RuntimeError(
                f"Doc {did} ({doc_name}) has vec_dim={idx.d} but collection has "
                f"dim={vec_dim} from earlier docs — mixed embedders are not supported."
            )

        vecs = np.zeros((n_vec, idx.d), dtype=np.float32)
        for i in range(n_vec):
            vecs[i] = idx.reconstruct(i)
        all_vecs.append(vecs)

        with open(chunks_p) as f:
            chunks = json.load(f)
        # Re-key chunks against the union and add attribution. The original
        # chunk["id"] is preserved as `local_id` so we can trace back.
        base = sum(v.shape[0] for v in all_vecs[:-1])
        for j, c in enumerate(chunks):
            c2 = dict(c)
            c2["local_id"] = c2.get("id", j)
            c2["id"] = base + j
            c2["doc_id"] = did
            c2["doc_name"] = doc_name
            all_chunks.append(c2)

        n_docs += 1

    if not all_vecs:
        raise RuntimeError("No member doc has a usable FAISS index — cannot build collection RAG.")

    merged = np.concatenate(all_vecs, axis=0)
    out_index = faiss.IndexFlatIP(merged.shape[1])
    out_index.add(merged)

    out_faiss = _col.rag_faiss_path(col_id)
    out_chunks = _col.rag_chunks_path(col_id)
    out_faiss.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(out_index, str(out_faiss))
    out_chunks.write_text(json.dumps(all_chunks))

    return {
        "n_chunks": merged.shape[0],
        "vec_dim": int(merged.shape[1]),
        "n_docs": n_docs,
        "n_docs_requested": len(doc_ids),
    }


# ─── Phase 3: PageIndex doc-router ────────────────────────────────────────────

DOC_SUMMARY_PROMPT = """\
You are summarizing a document so a downstream routing model can decide
whether the document is relevant to a user's question.

Produce a STRICT JSON object with these fields and nothing else:
{{
  "summary":   "3-4 sentences describing what this document is about, "
               "its scope, and its main contributions or findings",
  "topics":    ["5-8 short topic phrases the document covers"]
}}

Document title: {doc_name}

Document hierarchy (PageIndex tree, JSON):
{tree}

A representative sample of the document text (truncated):
{sample}

Return only the JSON object.
"""


def _summarize_doc_for_routing(doc_id: str, doc_name: str) -> dict:
    """Build a routing summary for one doc. Uses the PageIndex tree as the
    primary signal (it's a hierarchical TOC) and falls back to a text sample
    for docs whose tree is sparse."""
    from . import pageindex_agent, config
    from .llm import complete

    # Tree (best case): structured TOC tells us topics for free.
    try:
        tree = pageindex_agent.get_tree_structure(
            workspace=library.pageindex_workspace(doc_id),
            doc_id_file=library.pageindex_doc_id_file(doc_id),
        )
        tree_str = json.dumps(tree, indent=2)[:6000]
    except Exception:
        tree_str = "(unavailable)"

    # Sample text: first few chunks. RAG chunks are already on disk.
    try:
        chunks_p = library.chunks_path(doc_id)
        with open(chunks_p) as f:
            chunks = json.load(f)
        # Take the first ~3000 chars from the union of the first few chunks.
        sample = "\n\n".join(c.get("text", "") for c in chunks[:4])[:3000]
    except Exception:
        sample = "(unavailable)"

    result = complete(
        model=config.MODEL,
        messages=[
            {"role": "user", "content": DOC_SUMMARY_PROMPT.format(
                doc_name=doc_name, tree=tree_str, sample=sample,
            )},
        ],
        temperature=0.0,
    )
    content = (result.get("content") or "").strip()
    # Loose JSON extraction so we tolerate ```json fences or prose tails.
    import re
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    blob = fence.group(1) if fence else content
    start, end = blob.find("{"), blob.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"summary": "(could not parse summary)", "topics": [], "raw": content}
    try:
        parsed = json.loads(blob[start:end + 1])
    except json.JSONDecodeError:
        return {"summary": "(could not parse summary)", "topics": [], "raw": content}
    return {
        "summary": (parsed.get("summary") or "").strip(),
        "topics":  [str(t).strip() for t in (parsed.get("topics") or []) if str(t).strip()],
    }


def build_pageindex_router(col_id: str, doc_ids: list[str]) -> dict:
    """Generate per-doc routing summaries and persist them. One LLM call
    per member doc — done once at collection-build time and reused for
    every query."""
    out: dict[str, dict] = {}
    for did in doc_ids:
        meta = library.get_document(did)
        doc_name = meta.get("name") or did
        summary = _summarize_doc_for_routing(did, doc_name)
        out[did] = {"name": doc_name, **summary}

    p = _col.doc_summaries_path(col_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return {
        "n_docs_summarized": sum(1 for v in out.values() if v.get("summary")
                                 and not v["summary"].startswith("(")),
        "n_docs": len(doc_ids),
    }


# ─── Phase 4: GraphRAG entity resolution + merge ──────────────────────────────

ENTITY_PAIR_PROMPT = """\
You are deciding whether two named entities refer to the SAME real-world
entity. Answer based on the names and descriptions. If both refer to the
same company / person / concept / place / object, answer YES. If they are
related but distinct (e.g. parent company vs subsidiary, person vs role,
two products from the same maker), answer NO. If you cannot tell, answer NO.

Reply with exactly one token: YES or NO.

Entity A:
  name: {name_a}
  description: {desc_a}
  source doc: {doc_a}

Entity B:
  name: {name_b}
  description: {desc_b}
  source doc: {doc_b}

Are A and B the same real-world entity? Reply YES or NO."""


def _llm_judge_same_entity(a: dict, b: dict) -> bool:
    """One LLM call: are these two entities the same real-world thing?"""
    from .llm import complete
    from . import config

    result = complete(
        model=config.MODEL,
        messages=[{"role": "user", "content": ENTITY_PAIR_PROMPT.format(
            name_a=a["name"], desc_a=a.get("description", "")[:300], doc_a=a.get("doc_name", "?"),
            name_b=b["name"], desc_b=b.get("description", "")[:300], doc_b=b.get("doc_name", "?"),
        )}],
        temperature=0.0,
        max_tokens=8,
    )
    answer = (result.get("content") or "").strip().upper()
    # Accept the first YES/NO token regardless of surrounding fluff.
    for tok in answer.replace(",", " ").replace(".", " ").split():
        if tok in ("YES", "NO"):
            return tok == "YES"
    return False


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_graphrag(col_id: str, doc_ids: list[str], progress: Callable[[str, dict], None] | None = None) -> dict:
    """LLM-graded entity resolution across per-doc graphs, merged graph,
    fresh community detection on the unified graph.

    Pipeline:
      1. Load each per-doc property graph → collect (entity, description, doc_id).
      2. Embedding pass: pair candidates whose name embeddings are similar.
      3. LLM judge: for each candidate pair, "same real-world entity?"
      4. Union-find → canonical name per cluster.
      5. Re-key entities + relations against the canonical names, dedupe.
      6. Build a fresh GraphRAGStore on the merged graph, run
         build_communities(), persist alongside an entity_resolution.json.
    """
    from . import graphrag_pipeline as gp, embed
    from llama_index.core.graph_stores.types import EntityNode, Relation

    def _p(phase: str, **kw):
        # progress here is build_collection's inner _p, signature (phase, **kw)
        # — must unpack, not pass as a positional dict.
        if progress:
            progress(phase, **kw)

    # ----- step 1: gather entities + relations from all per-doc graphs -----
    all_entities: list[dict] = []  # {"name", "description", "doc_id", "doc_name", "node_id"}
    all_relations: list[dict] = []  # {"src", "tgt", "label", "description", "doc_id", "doc_name"}

    for did in doc_ids:
        gdir = library.graphrag_dir(did)
        store, _summaries = gp.load_store(gdir)
        if store is None:
            continue
        meta = library.get_document(did)
        doc_name = meta.get("name") or did

        try:
            nodes = list(store.graph.nodes.values())
        except Exception:
            nodes = []
        for n in nodes:
            name = getattr(n, "name", None) or getattr(n, "id", None)
            if not name:
                continue
            props = getattr(n, "properties", None) or {}
            desc = props.get("description", "") if isinstance(props, dict) else ""
            all_entities.append({
                "name": str(name),
                "description": str(desc or ""),
                "doc_id": did,
                "doc_name": doc_name,
                "node_id": getattr(n, "id", str(name)),
            })

        for src, lbl, tgt, desc in store._iter_triplets_safe():
            all_relations.append({
                "src": src, "tgt": tgt, "label": lbl, "description": desc,
                "doc_id": did, "doc_name": doc_name,
            })

    if not all_entities:
        raise RuntimeError(
            "No entities to merge — none of the member docs have a built "
            "GraphRAG index. Run single-doc indexing on each doc first."
        )

    _p("graphrag-resolution", n_entities=len(all_entities), n_relations=len(all_relations))

    # ----- step 2: embedding-based candidate-pair generation ---------------
    # Embed each entity once as "<name> — <description>". Pair entities
    # whose cosine similarity exceeds the threshold.
    embedder = embed.get_embedder()
    texts = [f"{e['name']} — {e['description'][:200]}" for e in all_entities]
    vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    vecs = np.asarray(vecs, dtype=np.float32)

    threshold = float(__import__("os").environ.get("COLLECTION_ER_EMBED_THRESHOLD", "0.78"))
    candidates: list[tuple[int, int]] = []
    n = vecs.shape[0]
    # O(N^2) pairwise. With a few hundred entities this is fine; if N gets
    # large enough to matter we can swap in faiss/hnsw.
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -1.0)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                # Also gate on string-equality OR cross-doc: same-doc identical
                # names already merged by the per-doc extractor, so skipping
                # those reduces LLM cost without losing recall.
                if (all_entities[i]["doc_id"] != all_entities[j]["doc_id"]
                        or all_entities[i]["name"].lower() != all_entities[j]["name"].lower()):
                    candidates.append((i, j))

    _p("graphrag-resolution", n_entities=len(all_entities), n_candidates=len(candidates))

    # ----- step 3: LLM judges each candidate pair --------------------------
    uf = _UnionFind()
    # Seed every entity into the union-find under its (doc_id, name) key so
    # singletons survive the merge unchanged.
    for idx, e in enumerate(all_entities):
        uf.find(_entity_key(e))

    yes_pairs: list[tuple[int, int]] = []
    n_judged = 0
    for i, j in candidates:
        n_judged += 1
        same = _llm_judge_same_entity(all_entities[i], all_entities[j])
        if same:
            uf.union(_entity_key(all_entities[i]), _entity_key(all_entities[j]))
            yes_pairs.append((i, j))

    # ----- step 4: build the canonical-name map -----------------------------
    # Cluster entities by their union-find root, then pick a canonical name
    # per cluster: the shortest non-empty name (heuristic for "least
    # qualified" form, e.g. "Tesla" over "Tesla, Inc." over "TESLA MOTORS INC").
    clusters: dict[str, list[int]] = {}
    for idx, e in enumerate(all_entities):
        root = uf.find(_entity_key(e))
        clusters.setdefault(root, []).append(idx)

    canonical_for_root: dict[str, str] = {}
    resolution_map: dict[str, list[str]] = {}  # canonical_name → list of aliases
    for root, members in clusters.items():
        names = [all_entities[i]["name"] for i in members]
        canonical = min(names, key=lambda s: (len(s), s.lower()))
        canonical_for_root[root] = canonical
        aliases = sorted(set(n for n in names if n != canonical))
        if aliases:
            resolution_map[canonical] = aliases
        else:
            resolution_map.setdefault(canonical, [])

    _p("graphrag-merge", n_clusters=len(clusters),
       n_merged=sum(1 for v in resolution_map.values() if v))

    # ----- step 5: build the merged GraphRAGStore ---------------------------
    merged = gp.GraphRAGStore()
    # One canonical EntityNode per cluster, description = concatenation of
    # member descriptions (capped) — gives the community-summary step richer
    # text to work with.
    for root, members in clusters.items():
        canonical = canonical_for_root[root]
        descs = [all_entities[i]["description"] for i in members if all_entities[i]["description"]]
        joined = " ".join(descs)[:600]
        merged.graph.add_node(EntityNode(
            name=canonical, label="entity",
            properties={"description": joined,
                        "doc_ids": sorted({all_entities[i]["doc_id"] for i in members})},
        ))

    name_to_canonical: dict[tuple[str, str], str] = {}  # (doc_id, raw_name) → canonical
    for idx, e in enumerate(all_entities):
        root = uf.find(_entity_key(e))
        name_to_canonical[(e["doc_id"], e["name"])] = canonical_for_root[root]

    seen_relations: set[tuple[str, str, str]] = set()
    for r in all_relations:
        c_src = name_to_canonical.get((r["doc_id"], r["src"]), r["src"])
        c_tgt = name_to_canonical.get((r["doc_id"], r["tgt"]), r["tgt"])
        if c_src == c_tgt:
            continue
        key = (c_src, r["label"], c_tgt)
        if key in seen_relations:
            continue
        seen_relations.add(key)
        merged.graph.add_relation(Relation(
            source_id=c_src, target_id=c_tgt, label=r["label"],
            properties={"description": r["description"], "doc_id": r["doc_id"]},
        ))

    _p("graphrag-communities", n_entities=len(clusters), n_relations=len(seen_relations))

    # ----- step 6: communities + persistence --------------------------------
    llm = gp._LocalLLMAdapter()
    cdiag = merged.build_communities(llm=llm)

    out_dir = _col.graphrag_dir(col_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    gp._save_store(merged, out_dir)
    (out_dir / "entity_resolution.json").write_text(json.dumps({
        "threshold": threshold,
        "n_entities_pre": len(all_entities),
        "n_entities_post": len(clusters),
        "n_candidates_judged": n_judged,
        "n_pairs_merged": len(yes_pairs),
        "resolution_map": resolution_map,
    }, indent=2))
    gp._save_debug(out_dir, {
        "n_chunks_pre_merge": len(all_entities),
        "n_chunks_post_merge": len(clusters),
        "total_entities": len(clusters),
        "total_relations": len(seen_relations),
        "n_chunks_with_entities": len(clusters),
        "n_chunks_llm_failed": 0,
        "n_chunks_empty_response": 0,
        "n_chunks_parse_miss": 0,
        "source": "collection-level merge",
    }, samples=[], community_diag=cdiag)

    return {
        "n_entities_pre": len(all_entities),
        "n_entities": len(clusters),
        "n_relations": len(seen_relations),
        "n_candidates_judged": n_judged,
        "n_pairs_merged": len(yes_pairs),
        "n_communities": cdiag.get("n_communities", 0),
        "embed_threshold": threshold,
    }


def _entity_key(e: dict) -> str:
    """Stable key for union-find: keep entities from different docs distinct
    even when their raw names happen to match (they may or may not be the
    same real-world thing; that's what the LLM judge decides)."""
    return f"{e['doc_id']}::{e['name']}"
