"""Custom 2-step tree-walk agent over a PageIndex tree.

We call the OpenAI Python SDK directly (rather than the OpenAI Agents SDK) so we have full control
over the reasoning trace and can render it side-by-side with the trad-RAG pipeline.

Flow:
  1. PLAN  : show the model the document tree, ask it to pick page ranges relevant to the question.
  2. ANSWER: fetch those pages, ask the model to answer using only that content.

All public functions optionally accept (workspace, doc_id) so they can target
either the bundled Tesla-10K tree (defaults from config) or any user-uploaded
document's tree under the library.
"""
import json
import re
import sys
from pathlib import Path

from . import config
from .llm import complete


# Add the local pageindex package to sys.path so we can import it without installing.
_APP_DIR = Path(__file__).parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from pageindex import PageIndexClient


# Cache one PageIndexClient per workspace path (different libraries = different clients).
_clients: dict[str, PageIndexClient] = {}


def get_client(workspace: Path | str | None = None) -> PageIndexClient:
    ws = str(workspace) if workspace else str(config.PAGEINDEX_WORKSPACE)
    if ws not in _clients:
        Path(ws).mkdir(parents=True, exist_ok=True)
        _clients[ws] = PageIndexClient(workspace=ws, model=config.INDEX_MODEL)
    return _clients[ws]


def load_doc_id(doc_id_file: Path | str | None = None) -> str:
    p = Path(doc_id_file) if doc_id_file else config.PAGEINDEX_DOC_ID_FILE
    return p.read_text().strip()


def _plan_system_prompt(doc_name: str | None) -> str:
    target = f"**{doc_name}**" if doc_name else "the provided document"
    return (
        f"You are PageIndex, a document QA assistant for {target}.\n"
        "You navigate the document like a human expert: read the table-of-contents tree, identify "
        "which sections are likely to answer the question, then read only those pages.\n\n"
        "Given the document tree, return a JSON object with the page ranges you want to read.\n"
        "Pick TIGHT ranges (rarely more than ~20 pages total). Prefer specific sections over broad ones.\n\n"
        "Reply format (a single JSON object, nothing else):\n"
        "{\n"
        "  \"reasoning\": \"<one or two sentences on which sections are relevant and why>\",\n"
        "  \"pages\": \"5-7,12,18-20\"\n"
        "}"
    )


def _answer_system_prompt(doc_name: str | None) -> str:
    target = f"**{doc_name}**" if doc_name else "the provided document"
    return (
        f"You are PageIndex, a document QA assistant for {target}.\n"
        "You have been given the text of the pages you selected. Answer the user's question using\n"
        "only that content. Cite the page numbers (e.g. \"[p. 32]\") for every claim. If the pages do\n"
        "not contain the answer, say so explicitly — do not guess."
    )


_PAGES_RE = re.compile(r"\d+(?:\s*-\s*\d+)?")


def _extract_json_loose(text: str) -> dict:
    """Find a JSON object in `text`, even if the model rambled."""
    text = text.strip()
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(candidate[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _sanitize_pages(pages: str, max_total: int = 30) -> str:
    """Keep only well-formed `N` or `N-M` ranges, dedupe, cap total pages."""
    if not pages:
        return ""
    parts = []
    total = 0
    for m in _PAGES_RE.finditer(pages):
        s = m.group(0).replace(" ", "")
        if "-" in s:
            a, b = s.split("-")
            try:
                a, b = int(a), int(b)
            except ValueError:
                continue
            if a > b:
                a, b = b, a
            span = min(b - a + 1, max_total - total)
            if span <= 0:
                break
            b = a + span - 1
            parts.append(f"{a}-{b}")
            total += span
        else:
            try:
                a = int(s)
            except ValueError:
                continue
            if total + 1 > max_total:
                break
            parts.append(str(a))
            total += 1
    return ",".join(parts)


def answer(
    query: str,
    workspace: Path | str | None = None,
    doc_id: str | None = None,
    doc_id_file: Path | str | None = None,
    doc_name: str | None = None,
) -> dict:
    """Run the 2-step tree walk and return a structured trace."""
    client = get_client(workspace=workspace)
    if doc_id is None:
        doc_id = load_doc_id(doc_id_file=doc_id_file)
    if doc_name is None:
        doc_name = config.DOC_NAME
    structure_json = client.get_document_structure(doc_id)

    # Step 1: PLAN
    plan_user = (
        f"Question: {query}\n\n"
        f"Document tree (table-of-contents JSON):\n{structure_json}\n\n"
        "Return your JSON plan."
    )
    plan_result = complete(
        model=config.RETRIEVE_MODEL,
        messages=[
            {"role": "system", "content": _plan_system_prompt(doc_name)},
            {"role": "user", "content": plan_user},
        ],
    )
    plan = _extract_json_loose(plan_result["content"])
    pages = _sanitize_pages(plan.get("pages", ""))
    plan_reasoning = plan.get("reasoning", "").strip()

    if not pages:
        return {
            "steps": [{
                "name": "plan",
                "thinking": plan_result["thinking"],
                "model_output": plan_result["content"],
                "parsed_pages": "",
                "reasoning": plan_reasoning,
            }],
            "pages_read": "",
            "page_text": "",
            "answer": "(PageIndex agent could not select page ranges from the tree.)",
            "thinking": "",
        }

    # Step 2: READ
    page_text = client.get_page_content(doc_id, pages)

    # Step 3: ANSWER
    answer_user = (
        f"Question: {query}\n\n"
        f"Pages you selected: {pages}\n\n"
        f"Page content:\n{page_text}\n\n"
        "Answer with citations."
    )
    answer_result = complete(
        model=config.RETRIEVE_MODEL,
        messages=[
            {"role": "system", "content": _answer_system_prompt(doc_name)},
            {"role": "user", "content": answer_user},
        ],
    )

    return {
        "steps": [
            {
                "name": "plan",
                "thinking": plan_result["thinking"],
                "model_output": plan_result["content"],
                "parsed_pages": pages,
                "reasoning": plan_reasoning,
            },
            {
                "name": "answer",
                "thinking": answer_result["thinking"],
                "model_output": answer_result["content"],
            },
        ],
        "pages_read": pages,
        "page_text": page_text,
        "answer": answer_result["content"],
        "thinking": answer_result["thinking"],
    }


def get_tree_structure(
    workspace: Path | str | None = None,
    doc_id: str | None = None,
    doc_id_file: Path | str | None = None,
) -> dict:
    """For the UI: return the parsed tree dict."""
    client = get_client(workspace=workspace)
    if doc_id is None:
        doc_id = load_doc_id(doc_id_file=doc_id_file)
    return json.loads(client.get_document_structure(doc_id))
