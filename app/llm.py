"""LiteLLM wrappers that handle Qwen3 thinking-mode and DeepSeek-R1 think-tag extraction."""
import os
import re
import litellm

from . import config  # ensures OLLAMA_API_BASE is set


litellm.drop_params = True
litellm.telemetry = False

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _is_qwen3(model: str) -> bool:
    return "qwen3" in (model or "").lower()


def _prepare_messages(model: str, messages: list[dict]) -> list[dict]:
    """Inject /no_think for qwen3 so JSON-only responses don't get padded with thinking."""
    if not _is_qwen3(model):
        return messages
    out = []
    for m in messages:
        c = m.get("content", "")
        if m.get("role") == "user" and "/no_think" not in c:
            c = f"{c}\n/no_think"
        out.append({**m, "content": c})
    return out


def split_thinking(content: str) -> tuple[str, str]:
    """Return (thinking, answer). For models that emit <think>…</think> blocks."""
    if not content:
        return "", ""
    matches = _THINK_RE.findall(content)
    thinking = "\n\n".join(m.strip() for m in matches).strip()
    answer = _THINK_RE.sub("", content).strip()
    return thinking, answer


def complete(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    api_base: str | None = None,
) -> dict:
    """Single completion call. Returns {'content', 'thinking', 'tool_calls', 'raw'}."""
    # LiteLLM picks the right env var based on model prefix; we set both.
    kwargs = dict(
        model=model,
        messages=_prepare_messages(model, messages),
        temperature=temperature,
        api_base=api_base or os.environ.get("LLM_API_BASE") or os.environ.get("OLLAMA_API_BASE"),
    )
    # Ollama-specific: request a large context window so multi-page prompts don't truncate,
    # and bump the timeout to accommodate slower local inference.
    if "ollama" in (model or "").lower():
        kwargs["num_ctx"] = int(os.environ.get("OLLAMA_NUM_CTX", "32768"))
        kwargs["timeout"] = int(os.environ.get("LLM_TIMEOUT", "1800"))
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    resp = litellm.completion(**kwargs)
    msg = resp.choices[0].message
    content = msg.content or ""
    # For models that emit <think>…</think> inline (qwen3 etc.) split here.
    inline_thinking, content_clean = split_thinking(content)
    # For models where the provider separates reasoning into its own field
    # (deepseek-r1 on Ollama → LiteLLM exposes `reasoning_content`).
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
