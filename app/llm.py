"""Thin wrapper around the OpenAI Python SDK.

We talk to a single OpenAI-compatible endpoint for all roles (indexing,
answering, retrieval reasoning). That endpoint can be:
  - hosted OpenAI itself
  - any local OpenAI-compatible server (vLLM, llama.cpp, LM Studio, etc.)
  - Ollama's /v1 compatibility shim (http://localhost:11434/v1)
  - a corporate gateway behind custom CA + headers

All knobs live in env vars; see .env.example.
"""
from __future__ import annotations
import json
import os
import re
import threading

import httpx
from openai import OpenAI, AsyncOpenAI

from . import config


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_lock = threading.Lock()
_sync_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None


def _parse_headers() -> dict[str, str]:
    """LLM_HEADERS can be either JSON ('{"X-Foo": "bar"}') or "K1:V1;K2:V2".

    Then, if LLM_API_KEY_HEADER is set, send the API key in that header (with
    the value from LLM_API_KEY) instead of the SDK's default Authorization
    header. Many corporate endpoints expect `apikey: …`, `x-api-key: …`, etc.
    Set LLM_API_KEY_HEADER=Authorization to keep the default behavior explicit.
    """
    raw = os.environ.get("LLM_HEADERS", "").strip()
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

    custom_auth_header = os.environ.get("LLM_API_KEY_HEADER", "").strip()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if custom_auth_header and api_key:
        headers[custom_auth_header] = api_key
        # NOTE: we cannot suppress the SDK's auto-generated `Authorization`
        # header from here — httpx rejects None values. The SDK will still
        # send `Authorization: Bearer <api_key>` in addition to your custom
        # header. Real-world internal gateways that use custom auth headers
        # typically ignore Authorization if the custom one is valid, so this
        # is usually fine. If your server is strict about it, set
        # LLM_API_KEY to a value that the gateway also accepts as a Bearer
        # token (often the same value).
    return headers


def _build_http_client(async_mode: bool = False):
    """httpx client respecting LLM_CA_BUNDLE (server-cert CA) and LLM_CLIENT_CERT (mTLS)."""
    ca = os.environ.get("LLM_CA_BUNDLE", "").strip()
    verify: bool | str = ca if ca else True
    if os.environ.get("LLM_VERIFY_SSL", "").lower() == "false":
        verify = False

    kwargs: dict = {"verify": verify, "timeout": int(os.environ.get("LLM_TIMEOUT", "1800"))}
    cert = os.environ.get("LLM_CLIENT_CERT", "").strip()
    if cert:
        if ":" in cert:
            cert_path, key_path = cert.split(":", 1)
            kwargs["cert"] = (cert_path.strip(), key_path.strip())
        else:
            kwargs["cert"] = cert
    return (httpx.AsyncClient if async_mode else httpx.Client)(**kwargs)


def _api_key_for_sdk() -> str:
    """The openai SDK requires `api_key` to be a non-empty string. If the user
    routes auth through a custom header, the SDK's own Authorization header
    is being suppressed in _parse_headers — the value here is irrelevant then
    but must still be set so the constructor doesn't raise."""
    return os.environ.get("LLM_API_KEY", "").strip() or "not-needed"


def get_sync_client() -> OpenAI:
    global _sync_client
    with _lock:
        if _sync_client is None:
            _sync_client = OpenAI(
                base_url=os.environ.get("LLM_API_BASE") or config.LLM_API_BASE,
                api_key=_api_key_for_sdk(),
                default_headers=_parse_headers() or None,
                http_client=_build_http_client(async_mode=False),
                max_retries=2,
            )
    return _sync_client


def get_async_client() -> AsyncOpenAI:
    global _async_client
    with _lock:
        if _async_client is None:
            _async_client = AsyncOpenAI(
                base_url=os.environ.get("LLM_API_BASE") or config.LLM_API_BASE,
                api_key=_api_key_for_sdk(),
                default_headers=_parse_headers() or None,
                http_client=_build_http_client(async_mode=True),
                max_retries=2,
            )
    return _async_client


def describe_client_config() -> dict:
    """For debugging: returns the resolved auth/transport config with the API
    key redacted. Useful to drop into a Streamlit status panel."""
    raw_key = os.environ.get("LLM_API_KEY", "")
    redacted = (raw_key[:4] + "…" + raw_key[-2:]) if len(raw_key) > 8 else ("set" if raw_key else "(empty)")
    headers = _parse_headers()
    custom_header = os.environ.get("LLM_API_KEY_HEADER", "").strip()
    return {
        "base_url": os.environ.get("LLM_API_BASE") or config.LLM_API_BASE,
        "api_key (redacted)": redacted,
        "api_key sent via": custom_header or "Authorization (SDK default)",
        "extra header keys": [k for k in headers if k.lower() != "authorization"],
        "ca_bundle": os.environ.get("LLM_CA_BUNDLE") or "(system CAs)",
        "verify_ssl": os.environ.get("LLM_VERIFY_SSL", "true"),
        "client_cert": os.environ.get("LLM_CLIENT_CERT") or "(none)",
        "timeout_s": int(os.environ.get("LLM_TIMEOUT", "1800")),
    }


def split_thinking(content: str) -> tuple[str, str]:
    """Return (thinking, answer). For models that emit inline <think>…</think> blocks."""
    if not content:
        return "", ""
    matches = _THINK_RE.findall(content)
    thinking = "\n\n".join(m.strip() for m in matches).strip()
    answer = _THINK_RE.sub("", content).strip()
    return thinking, answer


def _is_qwen3(model: str) -> bool:
    return "qwen3" in (model or "").lower()


def _prepare_messages(model: str, messages: list[dict]) -> list[dict]:
    """qwen3 emits a (usually empty) <think>...</think> block even when /no_think
    is set, which downstream JSON extractors choke on. We inject /no_think
    defensively for qwen3 models and strip the tags in split_thinking()."""
    if not _is_qwen3(model):
        return messages
    out = []
    for m in messages:
        c = m.get("content", "")
        if m.get("role") == "user" and "/no_think" not in c:
            c = f"{c}\n/no_think"
        out.append({**m, "content": c})
    return out


def complete(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    api_base: str | None = None,  # kept for signature compat; ignored — client is configured globally
) -> dict:
    """Single completion call. Returns {'content', 'thinking', 'tool_calls', 'raw'}."""
    client = get_sync_client()
    kwargs: dict = {
        "model": model,
        "messages": _prepare_messages(model, messages),
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    content = msg.content or ""
    inline_thinking, content_clean = split_thinking(content)
    provider_thinking = (getattr(msg, "reasoning_content", None) or "").strip()
    thinking = provider_thinking or inline_thinking
    answer = content_clean if inline_thinking else content
    tool_calls = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        try:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })
        except AttributeError:
            tool_calls.append({"raw": str(tc)})
    return {
        "content": answer,
        "thinking": thinking,
        "tool_calls": tool_calls,
        "raw": resp,
    }
