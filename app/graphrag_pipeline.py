"""LlamaIndex GraphRAG pipeline, following the v1 cookbook:
https://developers.llamaindex.ai/python/examples/cookbooks/graphrag_v1/

Indexing flow:
  1. SentenceSplitter chunks the document
  2. GraphRAGExtractor uses an LLM to pull (entity, relation, entity) triples
     plus descriptions from each chunk
  3. PropertyGraphIndex stores them in our GraphRAGStore (a
     SimplePropertyGraphStore subclass)
  4. After indexing, build_communities() runs hierarchical_leiden on the
     graph and uses an LLM to summarize each community
  5. Both the property graph and the community-summary dict are persisted
     so the Compare page can re-load them at query time

Query flow:
  - Iterate every community summary
  - Ask the LLM "given THIS community summary, how would you answer the
    user's question?"
  - Aggregate all per-community answers into a final response with one
    more LLM call

Returns a structured trace mirroring the other two pipelines so the
Compare-page tabs can render uniformly.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

# LlamaIndex imports — keep them at module import time so import errors land
# during app startup rather than mid-indexing.
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CustomLLM,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TransformComponent
from pydantic import PrivateAttr

from . import config, llm as _local_llm
from .embed import get_embedder


# ───────────────────────────────────────────────────────────────────────────
# Configuration / transport — same shape as embed.py + vision.py:
# GRAPHRAG_* env vars fall back to LLM_* if unset.
# ───────────────────────────────────────────────────────────────────────────

def _env(name: str, fallback: str | None = None, default: str = "") -> str:
    v = os.environ.get(name, "").strip()
    if v:
        return v
    if fallback:
        return os.environ.get(fallback, "").strip() or default
    return default


class _LocalEmbeddingAdapter(BaseEmbedding):
    """LlamaIndex-compatible adapter that delegates to whatever app.embed
    has configured — sentence-transformers (local) or our OpenAI-compatible
    embed endpoint. Avoids LlamaIndex's default fallback to hosted OpenAI."""

    _embedder: object = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embedder = get_embedder()

    def _get_query_embedding(self, query: str) -> list[float]:
        v = self._embedder.encode([query], normalize_embeddings=True)
        try:
            return v[0].tolist()
        except AttributeError:
            return list(v[0])

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_query_embedding(text)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_query_embedding(text)


class _LocalLLMAdapter(CustomLLM):
    """LlamaIndex-compatible LLM that delegates to `app.llm` — the same
    openai.OpenAI client RAG and PageIndex already use.

    We do NOT use `llama_index.llms.openai.OpenAI` here because it builds its
    own openai client lifecycle that doesn't always honor our custom
    `apikey` header / private CA / mTLS setup on the first request. Routing
    everything through one shared client keeps GraphRAG's transport identical
    to the working pipelines, which matters on proprietary networks where
    every LLM call must hit the user's internal gateway.
    """

    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = _env("GRAPHRAG_MODEL") or config.MODEL or "qwen3:14b"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=int(_env("GRAPHRAG_CONTEXT_WINDOW", "LLM_CONTEXT_WINDOW", default="16384")),
            num_output=self.max_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
        )

    def _to_dict_messages(self, messages: list[ChatMessage]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            role = getattr(m.role, "value", None) or str(m.role)
            content = m.content or ""
            out.append({"role": role, "content": content})
        return out

    @llm_chat_callback()
    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        result = _local_llm.complete(
            model=self.model,
            messages=self._to_dict_messages(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=result.get("content", "")),
            raw=result.get("raw"),
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        result = _local_llm.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return CompletionResponse(text=result.get("content", ""), raw=result.get("raw"))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs):
        # GraphRAG's flow never streams — yield the full completion once.
        yield self.complete(prompt, formatted=formatted, **kwargs)


def _build_llm() -> "_LocalLLMAdapter":
    """Return a LlamaIndex LLM that uses our shared `app.llm` transport. All
    auth/CA/header complexity is inherited from app/llm.py — no separate
    config path for GraphRAG, so what works for RAG works for GraphRAG."""
    return _LocalLLMAdapter()


# ───────────────────────────────────────────────────────────────────────────
# GraphRAGExtractor — entity + relation extraction per chunk
# ───────────────────────────────────────────────────────────────────────────

KG_TRIPLET_EXTRACT_TMPL = """\
You are given a passage of text. Extract a set of (entity, relation, entity) triples
plus a short description of each entity and each relation.

Return strictly the format below — one block per triple, separated by blank lines.

[ENTITY] name: <entity name>
description: <short factual description of the entity, grounded in this passage>

[ENTITY] name: <entity name>
description: <short factual description>

[RELATION] source: <entity name>
target: <entity name>
label: <short relation label, e.g. "employs", "located_in", "produces">
description: <one sentence describing this specific relationship>

Repeat the [ENTITY] / [RELATION] blocks as many times as appropriate. At most {max_paths_per_chunk}
relations per passage. If nothing is extractable, return an empty response.

Passage:
{text}
"""


_ENTITY_BLOCK = re.compile(
    r"\[ENTITY\]\s*name:\s*(?P<name>[^\n]+)\s*\n\s*description:\s*(?P<desc>[^\n]+)",
    re.IGNORECASE,
)
_RELATION_BLOCK = re.compile(
    r"\[RELATION\]\s*source:\s*(?P<src>[^\n]+)\s*\n\s*"
    r"target:\s*(?P<tgt>[^\n]+)\s*\n\s*"
    r"label:\s*(?P<lbl>[^\n]+)\s*\n\s*"
    r"description:\s*(?P<desc>[^\n]+)",
    re.IGNORECASE,
)

# Loose fallbacks — small models (gemma, qwen3:8b, …) frequently drop the
# bracketed [ENTITY]/[RELATION] tags or wrap them in bold markdown. Try these
# only if the strict parse returned nothing.
_LOOSE_HEADER_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*{0,2})\s*\[?(entity|relation|relationship|edge|node)\]?\s*\*{0,2}\s*[:\-]?",
    re.IGNORECASE,
)
_KV_RE = re.compile(r"^\s*(?:[-*]\s*)?\*{0,2}\s*(\w[\w\s]*?)\s*\*{0,2}\s*[:\-]\s*(.+?)\s*$")


def _parse_loose_blocks(text: str) -> tuple[list[dict], list[dict]]:
    """Permissive parser: scan line-by-line, split into blocks on a header
    line ('Entity:', '**Relation**:', '- entity', etc.), and pull key/value
    pairs out of the body. Accepts entities with just a name (no description)
    and relations with just src/tgt/label."""
    entities: list[dict] = []
    relations: list[dict] = []
    if not text:
        return entities, relations

    blocks: list[tuple[str, list[tuple[str, str]]]] = []
    current_kind: str | None = None
    current_kv: list[tuple[str, str]] = []

    def _flush():
        nonlocal current_kind, current_kv
        if current_kind and current_kv:
            blocks.append((current_kind, current_kv))
        current_kind, current_kv = None, []

    for line in text.splitlines():
        h = _LOOSE_HEADER_RE.match(line)
        if h:
            _flush()
            kind = h.group(1).lower()
            current_kind = "entity" if kind in {"entity", "node"} else "relation"
            # The header line may itself carry the name inline: "Entity: Foo"
            tail = line[h.end():].strip().strip("*").strip()
            if tail and ":" not in tail and len(tail) < 200:
                current_kv.append(("name" if current_kind == "entity" else "label", tail))
            continue
        if current_kind is None:
            continue
        m = _KV_RE.match(line)
        if m:
            current_kv.append((m.group(1).strip().lower(), m.group(2).strip()))
        elif not line.strip():
            _flush()
    _flush()

    for kind, kvs in blocks:
        d = {k: v for k, v in kvs}
        if kind == "entity":
            name = d.get("name") or d.get("entity") or d.get("node") or d.get("term")
            if not name:
                continue
            entities.append({"name": name, "description": d.get("description") or d.get("desc") or d.get("definition") or ""})
        else:
            src = d.get("source") or d.get("src") or d.get("from") or d.get("subject")
            tgt = d.get("target") or d.get("tgt") or d.get("to") or d.get("object")
            lbl = d.get("label") or d.get("relation") or d.get("predicate") or d.get("type") or ""
            if not (src and tgt):
                continue
            relations.append({
                "source": src, "target": tgt, "label": lbl or "related_to",
                "description": d.get("description") or d.get("desc") or "",
            })
    return entities, relations


def _parse_extraction(text: str) -> tuple[list[dict], list[dict]]:
    """Return (entities, relations) lists from the LLM's extraction response.

    Tries the strict [ENTITY]/[RELATION] block format first. If that yields
    nothing, falls back to a permissive parser that handles common small-model
    output variants (markdown bold, missing brackets, dropped fields)."""
    entities: list[dict] = []
    relations: list[dict] = []
    if not text:
        return entities, relations
    for m in _ENTITY_BLOCK.finditer(text):
        entities.append({"name": m.group("name").strip(), "description": m.group("desc").strip()})
    for m in _RELATION_BLOCK.finditer(text):
        relations.append({
            "source": m.group("src").strip(),
            "target": m.group("tgt").strip(),
            "label": m.group("lbl").strip(),
            "description": m.group("desc").strip(),
        })
    if not entities and not relations:
        return _parse_loose_blocks(text)
    return entities, relations


class GraphRAGExtractor(TransformComponent):
    """A LlamaIndex TransformComponent that augments each chunk node with
    extracted entities + relations in its metadata, ready for the
    PropertyGraphIndex to fold into the graph store."""
    llm: Any = None
    max_paths_per_chunk: int = 2
    _stats: dict = PrivateAttr(default_factory=dict)
    _samples: list = PrivateAttr(default_factory=list)
    _max_samples: int = PrivateAttr(default=6)

    def __init__(self, llm: Any, max_paths_per_chunk: int = 2, **kwargs):
        super().__init__(llm=llm, max_paths_per_chunk=max_paths_per_chunk, **kwargs)
        self._stats = {
            "n_chunks": 0,
            "n_chunks_with_entities": 0,
            "n_chunks_llm_failed": 0,
            "n_chunks_empty_response": 0,
            "n_chunks_parse_miss": 0,  # LLM responded with content but parser found 0
            "total_entities": 0,
            "total_relations": 0,
        }
        self._samples = []

    def _record_sample(self, reason: str, chunk_text: str, raw: str) -> None:
        if len(self._samples) >= self._max_samples:
            return
        self._samples.append({
            "reason": reason,
            "chunk_preview": chunk_text[:400],
            "llm_response": raw[:2000],
        })

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        for n in nodes:
            # PropertyGraphIndex.insert_nodes() asserts every node has these
            # two keys in metadata — initialize to empty lists so nodes whose
            # extraction returns nothing (empty chunk, LLM failure, parse
            # miss) still satisfy the assert.
            n.metadata[KG_NODES_KEY] = []
            n.metadata[KG_RELATIONS_KEY] = []

            text = (n.get_content() or "").strip()
            if not text:
                continue
            self._stats["n_chunks"] += 1
            prompt = KG_TRIPLET_EXTRACT_TMPL.format(
                text=text, max_paths_per_chunk=self.max_paths_per_chunk
            )
            try:
                resp = self.llm.chat([ChatMessage(role="user", content=prompt)])
                raw = resp.message.content or ""
            except Exception as e:
                self._stats["n_chunks_llm_failed"] += 1
                self._record_sample(f"llm-failed: {type(e).__name__}: {e}", text, "")
                continue
            ents, rels = _parse_extraction(raw)
            if not raw.strip():
                self._stats["n_chunks_empty_response"] += 1
                self._record_sample("empty-response", text, raw)
            elif not ents and not rels:
                self._stats["n_chunks_parse_miss"] += 1
                self._record_sample("parse-miss", text, raw)
            else:
                self._stats["n_chunks_with_entities"] += 1
            self._stats["total_entities"] += len(ents)
            self._stats["total_relations"] += len(rels)
            kg_nodes = [
                EntityNode(name=e["name"], label="entity",
                           properties={"description": e["description"]})
                for e in ents
            ]
            kg_rels = []
            seen_entities = {e["name"] for e in ents}
            for r in rels:
                # Auto-create endpoint EntityNodes for sources/targets the
                # model mentioned in a relation but didn't list as entities.
                for name in (r["source"], r["target"]):
                    if name not in seen_entities:
                        kg_nodes.append(EntityNode(name=name, label="entity",
                                                   properties={"description": ""}))
                        seen_entities.add(name)
                kg_rels.append(Relation(
                    source_id=r["source"], target_id=r["target"],
                    label=r["label"],
                    properties={"description": r["description"]},
                ))
            n.metadata[KG_NODES_KEY] = kg_nodes
            n.metadata[KG_RELATIONS_KEY] = kg_rels
        return nodes


# ───────────────────────────────────────────────────────────────────────────
# GraphRAGStore — adds community detection + summarization
# ───────────────────────────────────────────────────────────────────────────

COMMUNITY_SUMMARY_TMPL = """\
You are given a set of relationships in a knowledge graph that belong to a single
community of related entities. Summarize the community in 2-4 sentences. Capture:
- the main entities involved
- the dominant themes / what this community is about
- any notable facts spelled out in the relationships

Relationships:
{relationships}

Summary:"""


class GraphRAGStore(SimplePropertyGraphStore):
    """SimplePropertyGraphStore extension: detects communities via
    graspologic's hierarchical_leiden and stores per-community summaries."""

    _community_summaries: dict[int, str] = {}
    _community_to_entities: dict[int, set[str]] = {}

    def get_community_summaries(self) -> dict[int, str]:
        return dict(self._community_summaries)

    def get_community_to_entities(self) -> dict[int, list[str]]:
        return {cid: sorted(names) for cid, names in self._community_to_entities.items()}

    def build_communities(self, llm: Any, max_cluster_size: int = 10) -> None:
        """Run hierarchical Leiden over the graph and ask the LLM to summarize
        each community. Idempotent — overwrites previous summaries."""
        import networkx as nx
        from graspologic.partition import hierarchical_leiden

        # Translate the property graph into a NetworkX graph for community
        # detection. Nodes = entity names, edges = relations.
        g = nx.Graph()
        for rel in self.get_triplets():
            try:
                src, _label, tgt, _desc = rel  # depending on simplegraph API
            except (TypeError, ValueError):
                src = getattr(rel, "source_id", None)
                tgt = getattr(rel, "target_id", None)
                if not (src and tgt):
                    continue
            g.add_edge(src, tgt)

        if g.number_of_nodes() == 0:
            self._community_summaries = {}
            self._community_to_entities = {}
            return

        try:
            communities = hierarchical_leiden(g, max_cluster_size=max_cluster_size)
        except Exception:
            # Fall back to NetworkX's louvain if graspologic chokes on the graph.
            from networkx.algorithms.community import louvain_communities
            louv = louvain_communities(g)
            communities = [
                # mimic graspologic's HierarchicalCluster shape: .node, .cluster
                type("_C", (), {"node": n, "cluster": cid, "level": 0})()
                for cid, comm in enumerate(louv) for n in comm
            ]

        # Group entity names by cluster
        comm_to_entities: dict[int, set[str]] = {}
        for item in communities:
            cid = getattr(item, "cluster", None)
            node = getattr(item, "node", None)
            if cid is None or node is None:
                continue
            comm_to_entities.setdefault(cid, set()).add(node)

        # Build per-relation text within each community for the summarizer
        node_to_comm: dict[str, int] = {}
        for cid, names in comm_to_entities.items():
            for name in names:
                node_to_comm.setdefault(name, cid)

        comm_to_relations: dict[int, list[str]] = {}
        for src, _l, tgt, desc in (
            t for t in self._iter_triplets_safe()
        ):
            cid = node_to_comm.get(src) or node_to_comm.get(tgt)
            if cid is None:
                continue
            comm_to_relations.setdefault(cid, []).append(f"- {src} → {tgt}: {desc or _l}")

        # Summarize each community
        summaries: dict[int, str] = {}
        for cid, rels in comm_to_relations.items():
            if not rels:
                continue
            prompt = COMMUNITY_SUMMARY_TMPL.format(
                relationships="\n".join(rels[:200])  # cap to keep prompt sane
            )
            try:
                resp = llm.chat([ChatMessage(role="user", content=prompt)])
                summaries[cid] = (resp.message.content or "").strip()
            except Exception as e:
                summaries[cid] = f"(summary failed: {e!r})"

        self._community_summaries = summaries
        self._community_to_entities = comm_to_entities

    def _iter_triplets_safe(self) -> Iterable[tuple[str, str, str, str]]:
        """Best-effort iterator over (source, label, target, description). The
        underlying property-graph store API has shifted across LlamaIndex
        versions — this normalizes both shapes."""
        try:
            for t in self.get_triplets():
                # Older API: tuple/list of length 3 or 4
                if isinstance(t, (tuple, list)):
                    if len(t) == 4:
                        yield t[0], t[1], t[2], t[3]
                    elif len(t) == 3:
                        yield t[0], t[1], t[2], ""
                    continue
                # Newer API: a Triplet-like object with .source_id, .target_id, .label, .properties
                src = getattr(t, "source_id", None) or getattr(t, "source", None)
                tgt = getattr(t, "target_id", None) or getattr(t, "target", None)
                lbl = getattr(t, "label", "") or ""
                desc = ""
                props = getattr(t, "properties", None)
                if isinstance(props, dict):
                    desc = props.get("description", "")
                if src and tgt:
                    yield str(src), str(lbl), str(tgt), str(desc)
        except Exception:
            return


# ───────────────────────────────────────────────────────────────────────────
# Persistence
# ───────────────────────────────────────────────────────────────────────────

def _summary_path(graphrag_dir: Path) -> Path:
    return graphrag_dir / "community_summaries.json"


def _store_path(graphrag_dir: Path) -> Path:
    return graphrag_dir / "property_graph_store.json"


def _save_store(store: GraphRAGStore, graphrag_dir: Path) -> None:
    graphrag_dir.mkdir(parents=True, exist_ok=True)
    try:
        store.persist(str(_store_path(graphrag_dir)))
    except Exception:
        # Older / newer API: try the alt persist signature
        try:
            store.persist(persist_path=str(_store_path(graphrag_dir)))
        except Exception:
            pass
    payload = {
        "summaries": {str(k): v for k, v in store.get_community_summaries().items()},
        "entities":  {str(k): v for k, v in store.get_community_to_entities().items()},
    }
    _summary_path(graphrag_dir).write_text(json.dumps(payload, indent=2))


def _save_debug(graphrag_dir: Path, stats: dict, samples: list) -> None:
    graphrag_dir.mkdir(parents=True, exist_ok=True)
    (graphrag_dir / "extraction_debug.json").write_text(
        json.dumps({"stats": stats, "samples": samples}, indent=2)
    )


def load_debug(graphrag_dir: Path) -> dict:
    """Return the extraction-debug payload (stats + samples) if persisted."""
    p = Path(graphrag_dir) / "extraction_debug.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_store(graphrag_dir: Path) -> tuple[GraphRAGStore | None, dict]:
    """Load the persisted store + summary dict. Returns (None, {}) if not built."""
    sp = _store_path(graphrag_dir)
    if not sp.exists():
        return None, {}
    try:
        store = GraphRAGStore.from_persist_path(str(sp))
    except Exception:
        store = GraphRAGStore()
        try:
            store.load(str(sp))
        except Exception:
            store = None
    summaries: dict = {}
    if _summary_path(graphrag_dir).exists():
        try:
            payload = json.loads(_summary_path(graphrag_dir).read_text())
            summaries = {
                int(k): v for k, v in payload.get("summaries", {}).items()
            }
            ent = {int(k): v for k, v in payload.get("entities", {}).items()}
        except Exception:
            ent = {}
    else:
        ent = {}
    if store is not None:
        store._community_summaries = summaries
        store._community_to_entities = {k: set(v) for k, v in ent.items()}
    return store, summaries


# ───────────────────────────────────────────────────────────────────────────
# Indexing entry point
# ───────────────────────────────────────────────────────────────────────────

def build_index(source_text: str, graphrag_dir: Path, max_paths_per_chunk: int = 2,
                 max_cluster_size: int = 10) -> dict:
    """Run the full GraphRAG pipeline on `source_text` and persist the results
    under `graphrag_dir`. Returns a small dict of stats for the progress UI.
    """
    llm = _build_llm()
    embed = _LocalEmbeddingAdapter()
    # Force LlamaIndex to use our LLM + embedder for any downstream defaults,
    # otherwise PropertyGraphIndex / its components will silently try to
    # instantiate hosted OpenAI clients on first call.
    Settings.llm = llm
    Settings.embed_model = embed

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    docs = [Document(text=source_text)]
    nodes = splitter.get_nodes_from_documents(docs)

    extractor = GraphRAGExtractor(llm=llm, max_paths_per_chunk=max_paths_per_chunk)

    store = GraphRAGStore()
    PropertyGraphIndex(
        nodes=nodes,
        property_graph_store=store,
        kg_extractors=[extractor],
        embed_model=embed,
        embed_kg_nodes=False,  # we don't need vector retrieval over graph nodes for this flow
        show_progress=False,
    )
    store.build_communities(llm=llm, max_cluster_size=max_cluster_size)

    _save_store(store, graphrag_dir)
    extraction_stats = dict(extractor._stats)
    _save_debug(graphrag_dir, extraction_stats, list(extractor._samples))

    n_communities = len(store.get_community_summaries())
    # If we extracted nothing parseable, surface the dominant failure mode so
    # the upload-page status line says something more useful than "0 entities".
    hint = ""
    if extraction_stats["total_entities"] == 0:
        if extraction_stats["n_chunks_llm_failed"] >= extraction_stats["n_chunks"]:
            hint = " — every extraction LLM call failed; check transport/auth"
        elif extraction_stats["n_chunks_parse_miss"] > 0:
            hint = (" — model returned text but parser found no entities; "
                    "check extraction_debug.json for sample responses")
        else:
            hint = " — extraction returned nothing (empty model responses)"

    return {
        "n_chunks": len(nodes),
        "n_entities": extraction_stats["total_entities"],
        "n_relations": extraction_stats["total_relations"],
        "n_communities": n_communities,
        "hint": hint,
        "extraction_stats": extraction_stats,
    }


# ───────────────────────────────────────────────────────────────────────────
# Query
# ───────────────────────────────────────────────────────────────────────────

PER_COMMUNITY_TMPL = """\
You are answering a user's question using ONLY the following community summary
from a knowledge graph built from a document.

Community summary:
{summary}

User question: {query}

Answer in 2-4 sentences. If the summary clearly doesn't address the question,
say so briefly. Do not invent facts.
"""

AGGREGATE_TMPL = """\
You produced several partial answers to the same user question, each based on
a different community summary from a knowledge graph. Combine them into a single,
concise, non-redundant final answer. If partial answers conflict, prefer concrete,
specific facts. If multiple partials say "this summary doesn't address the question",
do not include that — only carry forward useful content.

User question: {query}

Partial answers:
{partials}

Final answer:"""


def answer(query: str, graphrag_dir: Path, doc_name: str | None = None) -> dict:
    """Per-community answer + aggregate flow.

    Returns the same general shape as the other two pipelines so the Compare
    page can render uniformly: {"answer": str, "thinking": str, "steps": [...]}.
    """
    gd = Path(graphrag_dir)
    store, summaries = load_store(gd)
    if store is None:
        return {
            "answer": "(GraphRAG index not built for this document.)",
            "thinking": "",
            "n_communities": 0,
            "partials": [],
        }
    if not summaries:
        debug = load_debug(gd)
        stats = debug.get("stats", {})
        samples = debug.get("samples", [])
        diag_lines = [
            "GraphRAG indexed without errors but produced **0 communities**, "
            "so there is nothing to answer from on the graph side.",
            "",
            f"- chunks processed: {stats.get('n_chunks', '?')}",
            f"- chunks where extraction yielded ≥1 entity: {stats.get('n_chunks_with_entities', 0)}",
            f"- chunks where LLM call failed: {stats.get('n_chunks_llm_failed', 0)}",
            f"- chunks where LLM returned empty text: {stats.get('n_chunks_empty_response', 0)}",
            f"- chunks where parser found nothing: {stats.get('n_chunks_parse_miss', 0)}",
            f"- total entities extracted: {stats.get('total_entities', 0)}",
            f"- total relations extracted: {stats.get('total_relations', 0)}",
        ]
        if samples:
            diag_lines += ["", "**Sample LLM responses** (first few problematic chunks):", ""]
            for s in samples[:3]:
                diag_lines.append(f"_reason: {s.get('reason', '?')}_")
                resp = s.get("llm_response", "")
                diag_lines.append("```\n" + (resp[:600] or "(empty response)") + "\n```")
        diag_lines += [
            "",
            "**What this means:** entity extraction did not produce enough "
            "structured output for community detection. Likely causes: model "
            "is not following the `[ENTITY]` / `[RELATION]` block format, "
            "context is too short, or auth is silently returning empty bodies. "
            "Try a larger model for `GRAPHRAG_MODEL`, or inspect "
            "`extraction_debug.json` under this doc's `graphrag/` dir.",
        ]
        return {
            "answer": "\n".join(diag_lines),
            "thinking": "",
            "n_communities": 0,
            "partials": [],
            "extraction_stats": stats,
        }

    llm = _build_llm()
    # The query path doesn't touch embeddings, but set them anyway so any
    # internal LlamaIndex Settings reference doesn't fall through to hosted
    # OpenAI as a default.
    Settings.llm = llm
    Settings.embed_model = _LocalEmbeddingAdapter()

    partials: list[dict] = []
    for cid, summary in summaries.items():
        prompt = PER_COMMUNITY_TMPL.format(summary=summary, query=query)
        try:
            resp = llm.chat([ChatMessage(role="user", content=prompt)])
            text = (resp.message.content or "").strip()
        except Exception as e:
            text = f"(community {cid} answer failed: {e!r})"
        partials.append({"community_id": int(cid), "summary": summary, "answer": text})

    # Aggregate
    partials_text = "\n\n".join(
        f"[community {p['community_id']}]\n{p['answer']}" for p in partials
    )
    agg_prompt = AGGREGATE_TMPL.format(query=query, partials=partials_text)
    try:
        agg_resp = llm.chat([ChatMessage(role="user", content=agg_prompt)])
        final = (agg_resp.message.content or "").strip()
    except Exception as e:
        final = f"(aggregation failed: {e!r})"

    return {
        "answer": final,
        "thinking": "",
        "n_communities": len(partials),
        "partials": partials,
    }


# ───────────────────────────────────────────────────────────────────────────
# Diagnostic for the System Status panel
# ───────────────────────────────────────────────────────────────────────────

def describe_graphrag_config() -> dict:
    """GraphRAG now shares the same openai.OpenAI client as `app.llm` — see the
    'LLM transport' section for auth/CA details. Only the model name is
    independently configurable here (GRAPHRAG_MODEL), and it falls back to MODEL.
    """
    return {
        "transport": "shared with LLM (see 'LLM transport' above)",
        "model": _env("GRAPHRAG_MODEL") or config.MODEL,
        "model env override": "GRAPHRAG_MODEL" if _env("GRAPHRAG_MODEL") else "(falls back to MODEL)",
    }
