"""Embedding backend.

Two providers, selected via EMBED_PROVIDER:
  - "sentence-transformers" (default) — local HuggingFace model, e.g.
    BAAI/bge-small-en-v1.5. No network, but requires the model name to
    actually be downloadable from HuggingFace (or already cached locally).
  - "openai" — call an OpenAI-compatible /v1/embeddings endpoint. Reuses
    your LLM auth/CA/headers config by default; override with EMBED_*
    env vars if your embedder lives at a different URL.

Both expose the same interface as sentence_transformers.SentenceTransformer:
    .encode(texts, normalize_embeddings=True, batch_size=N, show_progress_bar=...)
returning a numpy array.
"""
from __future__ import annotations
import json
import os
import threading
from typing import Iterable

import httpx
import numpy as np
from openai import OpenAI

from . import config


_lock = threading.Lock()
_cached = None


def _env(name: str, fallback_name: str | None = None, default: str = "") -> str:
    v = os.environ.get(name, "").strip()
    if v:
        return v
    if fallback_name:
        return os.environ.get(fallback_name, "").strip() or default
    return default


def _parse_embed_headers() -> dict:
    """Same shape as the LLM-side parser but with EMBED_ vars falling back to LLM_."""
    raw = _env("EMBED_HEADERS", "LLM_HEADERS")
    headers: dict[str, str] = {}
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                headers = {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            for pair in raw.split(";"):
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    headers[k.strip()] = v.strip()
    custom_auth_header = _env("EMBED_API_KEY_HEADER", "LLM_API_KEY_HEADER")
    api_key = _env("EMBED_API_KEY", "LLM_API_KEY")
    if custom_auth_header and api_key:
        headers[custom_auth_header] = api_key
    return headers


def _build_api_client() -> OpenAI:
    base = _env("EMBED_API_BASE", "LLM_API_BASE", default=config.LLM_API_BASE)
    api_key = _env("EMBED_API_KEY", "LLM_API_KEY", default="not-needed")
    ca = _env("EMBED_CA_BUNDLE", "LLM_CA_BUNDLE")
    verify: bool | str = ca if ca else True
    if (_env("EMBED_VERIFY_SSL") or _env("LLM_VERIFY_SSL")).lower() == "false":
        verify = False
    timeout = int(_env("EMBED_TIMEOUT", "LLM_TIMEOUT", default="1800"))
    cert = _env("EMBED_CLIENT_CERT", "LLM_CLIENT_CERT")
    http_kwargs: dict = {"verify": verify, "timeout": timeout}
    if cert:
        http_kwargs["cert"] = tuple(cert.split(":", 1)) if ":" in cert else cert
    return OpenAI(
        base_url=base,
        api_key=api_key,
        default_headers=_parse_embed_headers() or None,
        http_client=httpx.Client(**http_kwargs),
        max_retries=2,
    )


class APIEmbedder:
    """Mimics SentenceTransformer's .encode() signature using an OpenAI-compatible endpoint."""

    def __init__(self, model: str):
        self.model = model
        self.client = _build_api_client()

    def encode(
        self,
        texts: list[str] | Iterable[str],
        normalize_embeddings: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        texts = list(texts)
        vectors: list[list[float]] = []
        n = len(texts)
        for i in range(0, n, batch_size):
            batch = texts[i:i + batch_size]
            if show_progress_bar:
                print(f"  embedding {i+len(batch)}/{n}", flush=True)
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in resp.data)
        arr = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr if convert_to_numpy else vectors


class _LazySentenceTransformer:
    """Defer the heavy sentence-transformers import until actually used."""

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer
        self._inner = SentenceTransformer(model)

    def encode(self, *a, **kw):
        return self._inner.encode(*a, **kw)


def get_embedder():
    """Return a singleton embedder selected by EMBED_PROVIDER (default sentence-transformers)."""
    global _cached
    if _cached is not None:
        return _cached
    with _lock:
        if _cached is None:
            provider = os.environ.get("EMBED_PROVIDER", "sentence-transformers").strip().lower()
            model = config.EMBED_MODEL
            if provider in ("openai", "api"):
                _cached = APIEmbedder(model)
            else:
                _cached = _LazySentenceTransformer(model)
    return _cached


def describe_embed_config() -> dict:
    """Surface what actually got loaded — drop into a debug panel."""
    provider = os.environ.get("EMBED_PROVIDER", "sentence-transformers").strip().lower()
    if provider in ("openai", "api"):
        api_key = _env("EMBED_API_KEY", "LLM_API_KEY")
        redacted = (api_key[:4] + "…" + api_key[-2:]) if len(api_key) > 8 else ("set" if api_key else "(empty)")
        return {
            "provider": "openai-compatible API",
            "base_url": _env("EMBED_API_BASE", "LLM_API_BASE"),
            "model": config.EMBED_MODEL,
            "api_key (redacted)": redacted,
            "api_key sent via": _env("EMBED_API_KEY_HEADER", "LLM_API_KEY_HEADER") or "Authorization (SDK default)",
            "ca_bundle": _env("EMBED_CA_BUNDLE", "LLM_CA_BUNDLE") or "(system CAs)",
        }
    return {"provider": "sentence-transformers", "model": config.EMBED_MODEL}
