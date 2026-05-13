"""Optional vision-language pre-processing of PDFs.

When VISION_ENABLED=true, image regions in each PDF page are sent to a
vision-capable LLM (gemma-3-vision, GPT-4o, Claude, etc.) and the
description is injected back into the page text. The result:

  - Scanned/image-only pages → transcribed text from the page render
  - Charts, graphs, diagrams → natural-language descriptions inline
  - Logos / tiny icons → skipped (configurable size threshold)

That enriched text then flows through both pipelines (RAG chunking and
PageIndex tree-building) as normal, so a question like "what's the
revenue mix in the chart on p.32?" finds the page via either retrieval
mode.

Env vars (all VISION_* fall back to LLM_* if unset):
  VISION_ENABLED      "true" to activate (default off — VLM calls are slow + costly)
  VISION_MODEL        model name accepting OpenAI multimodal content format
  VISION_API_BASE     defaults to LLM_API_BASE
  VISION_API_KEY      defaults to LLM_API_KEY
  VISION_API_KEY_HEADER  defaults to LLM_API_KEY_HEADER
  VISION_CA_BUNDLE    defaults to LLM_CA_BUNDLE
  VISION_HEADERS      defaults to LLM_HEADERS
  VISION_MIN_PIXELS   skip images with W*H below this (default 40_000 ≈ 200x200)
  VISION_MAX_PER_PAGE cap describable images per page (default 6)
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
import os
import threading
from pathlib import Path
from typing import Iterable

import httpx
import pymupdf
from openai import OpenAI

from . import config


_lock = threading.Lock()
_client_cache: OpenAI | None = None
_desc_cache: dict[str, str] = {}  # image_hash -> description, in-memory


def is_enabled() -> bool:
    return os.environ.get("VISION_ENABLED", "false").strip().lower() == "true"


def _env(name: str, fallback: str | None = None, default: str = "") -> str:
    v = os.environ.get(name, "").strip()
    if v:
        return v
    if fallback:
        return os.environ.get(fallback, "").strip() or default
    return default


def _parse_headers() -> dict:
    raw = _env("VISION_HEADERS", "LLM_HEADERS")
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
    custom_auth = _env("VISION_API_KEY_HEADER", "LLM_API_KEY_HEADER")
    api_key = _env("VISION_API_KEY", "LLM_API_KEY")
    if custom_auth and api_key:
        headers[custom_auth] = api_key
    return headers


def get_vision_client() -> OpenAI:
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    with _lock:
        if _client_cache is None:
            ca = _env("VISION_CA_BUNDLE", "LLM_CA_BUNDLE")
            verify: bool | str = ca if ca else True
            if (_env("VISION_VERIFY_SSL") or _env("LLM_VERIFY_SSL")).lower() == "false":
                verify = False
            timeout = int(_env("VISION_TIMEOUT", "LLM_TIMEOUT", default="600"))
            base = _env("VISION_API_BASE", "LLM_API_BASE", default=config.LLM_API_BASE)
            key = _env("VISION_API_KEY", "LLM_API_KEY", default="not-needed")
            _client_cache = OpenAI(
                base_url=base,
                api_key=key,
                default_headers=_parse_headers() or None,
                http_client=httpx.Client(verify=verify, timeout=timeout),
                max_retries=2,
            )
    return _client_cache


def _vision_model() -> str:
    """Required when VISION_ENABLED=true. Falls back to MODEL if it looks
    like the user just configured one model for everything."""
    return os.environ.get("VISION_MODEL", "").strip() or config.MODEL


# ─── image extraction ─────────────────────────────────────────────────────────

def _png_bytes_from_xref(doc, xref: int) -> bytes | None:
    """Extract an image as PNG bytes for the model. Returns None on failure."""
    try:
        info = doc.extract_image(xref)
        if not info:
            return None
        # `info["ext"]` is the source format; we re-encode as PNG via pymupdf.Pixmap
        # so the VLM gets a consistent format.
        img_bytes = info.get("image")
        ext = info.get("ext", "png")
        if ext.lower() in ("jpg", "jpeg", "png", "webp"):
            return img_bytes
        # CMYK / weird formats — convert via Pixmap
        pix = pymupdf.Pixmap(doc, xref)
        if pix.n - pix.alpha >= 4:  # CMYK → RGB
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        return pix.tobytes("png")
    except Exception:
        return None


def _page_render_bytes(page) -> bytes:
    """Render the whole page as a PNG (used for scanned-PDF detection)."""
    matrix = pymupdf.Matrix(150 / 72, 150 / 72)  # ~150 DPI
    return page.get_pixmap(matrix=matrix).tobytes("png")


def _looks_like_scanned(page, page_text: str) -> bool:
    """Cheap heuristic: very little extracted text AND at least one large image."""
    if len(page_text.strip()) > 200:
        return False
    images = page.get_images(full=False)
    if not images:
        return False
    page_area = page.rect.width * page.rect.height
    for xref, *_ in images:
        try:
            bbox = page.get_image_bbox(xref) if hasattr(page, "get_image_bbox") else None
        except Exception:
            bbox = None
        if bbox and bbox.get_area() / max(page_area, 1) > 0.5:
            return True
    return False


# ─── VLM call ────────────────────────────────────────────────────────────────

def _describe(image_bytes: bytes, prompt: str) -> str:
    h = hashlib.sha1(image_bytes).hexdigest()
    if h in _desc_cache:
        return _desc_cache[h]
    b64 = base64.b64encode(image_bytes).decode("ascii")
    client = get_vision_client()
    resp = client.chat.completions.create(
        model=_vision_model(),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "").strip()
    _desc_cache[h] = text
    return text


_DESCRIBE_PROMPT = (
    "Describe what this image shows in 2-3 sentences. If it contains text "
    "(chart axis labels, captions, etc.) include the relevant numbers and "
    "labels verbatim. If it's a chart or graph, state the chart type, what "
    "is being compared, and any notable values or trends. Do not add commentary."
)

_TRANSCRIBE_PROMPT = (
    "Transcribe ALL text visible on this page exactly as it appears, "
    "preserving paragraph breaks, headings, and table structure (use "
    "markdown tables for tabular data). Do not summarize or paraphrase."
)


# ─── public API ──────────────────────────────────────────────────────────────

def enrich_page_text(page, raw_text: str) -> str:
    """Append image/page-render descriptions to a page's raw text.

    Returns the original text unchanged if vision is disabled or no
    eligible images are found.
    """
    if not is_enabled():
        return raw_text

    additions: list[str] = []

    # 1) Scanned-page case: VLM-transcribe the whole page.
    if _looks_like_scanned(page, raw_text):
        page_png = _page_render_bytes(page)
        try:
            text = _describe(page_png, _TRANSCRIBE_PROMPT)
            if text:
                additions.append(f"\n\n[PAGE TRANSCRIBED BY VISION MODEL]\n{text}\n[/PAGE]\n")
        except Exception as e:
            additions.append(f"\n\n[VISION_ERROR: {e!r}]\n")

    # 2) Per-image case: describe each embedded image above size threshold.
    min_px = int(os.environ.get("VISION_MIN_PIXELS", "40000"))
    cap = int(os.environ.get("VISION_MAX_PER_PAGE", "6"))
    done = 0
    for xref, *_ in (page.get_images(full=False) or []):
        if done >= cap:
            break
        try:
            info = page.parent.extract_image(xref)
        except Exception:
            continue
        if not info:
            continue
        w, h = info.get("width", 0), info.get("height", 0)
        if w * h < min_px:
            continue
        img = _png_bytes_from_xref(page.parent, xref)
        if not img:
            continue
        try:
            desc = _describe(img, _DESCRIBE_PROMPT)
            if desc:
                additions.append(f"\n\n[IMAGE]\n{desc}\n[/IMAGE]\n")
                done += 1
        except Exception as e:
            additions.append(f"\n\n[VISION_ERROR: {e!r}]\n")

    if not additions:
        return raw_text
    return raw_text + "".join(additions)


def describe_vision_config() -> dict:
    enabled = is_enabled()
    return {
        "enabled": enabled,
        "model": _vision_model() if enabled else "(off)",
        "base_url": _env("VISION_API_BASE", "LLM_API_BASE") if enabled else "(off)",
        "min_pixels": int(os.environ.get("VISION_MIN_PIXELS", "40000")),
        "max_per_page": int(os.environ.get("VISION_MAX_PER_PAGE", "6")),
    }
