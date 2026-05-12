"""Hand-craft a PageIndex-format tree for Tesla's 2024 10-K by scanning the PDF
for Part/Item headers and recording where they actually start in the physical PDF.

Why: PageIndex's LLM-driven auto-indexing is brittle on long, structured filings
when paired with small open-source models (qwen3:8b returns malformed JSON on
some of its prompts, cascading through PageIndex's fallback paths).

The resulting tree is functionally equivalent for the *retrieval* demo — that's
where PageIndex's interesting reasoning behavior happens.

After the structural tree is built, this script makes one small LLM call per
section to generate a one-line summary. Those prompts are short and reliable
even with qwen3:8b.
"""
import asyncio
import json
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf
from app import config
from app.pageindex.utils import llm_acompletion


# Body section headings are ALL CAPS: "ITEM 1. BUSINESS". Mixed case ("Item 9A")
# appears in the TOC and in prose cross-references — we don't want those.
ITEM_RE = re.compile(r"\bITEM\s+(\d{1,2}[A-C]?)\s*\.", re.ASCII)

# Canonical 10-K structure: which Part each Item belongs to and the official title.
CANONICAL_ITEMS = {
    "1":   ("PART I",   "Item 1. Business"),
    "1A":  ("PART I",   "Item 1A. Risk Factors"),
    "1B":  ("PART I",   "Item 1B. Unresolved Staff Comments"),
    "1C":  ("PART I",   "Item 1C. Cybersecurity"),
    "2":   ("PART I",   "Item 2. Properties"),
    "3":   ("PART I",   "Item 3. Legal Proceedings"),
    "4":   ("PART I",   "Item 4. Mine Safety Disclosures"),
    "5":   ("PART II",  "Item 5. Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities"),
    "6":   ("PART II",  "Item 6. [Reserved]"),
    "7":   ("PART II",  "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations"),
    "7A":  ("PART II",  "Item 7A. Quantitative and Qualitative Disclosures about Market Risk"),
    "8":   ("PART II",  "Item 8. Financial Statements and Supplementary Data"),
    "9":   ("PART II",  "Item 9. Changes in and Disagreements with Accountants on Accounting and Financial Disclosure"),
    "9A":  ("PART II",  "Item 9A. Controls and Procedures"),
    "9B":  ("PART II",  "Item 9B. Other Information"),
    "9C":  ("PART II",  "Item 9C. Disclosure Regarding Foreign Jurisdictions that Prevent Inspections"),
    "10":  ("PART III", "Item 10. Directors, Executive Officers and Corporate Governance"),
    "11":  ("PART III", "Item 11. Executive Compensation"),
    "12":  ("PART III", "Item 12. Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters"),
    "13":  ("PART III", "Item 13. Certain Relationships and Related Transactions, and Director Independence"),
    "14":  ("PART III", "Item 14. Principal Accountant Fees and Services"),
    "15":  ("PART IV",  "Item 15. Exhibits, Financial Statement Schedules"),
    "16":  ("PART IV",  "Item 16. Form 10-K Summary"),
}
CANONICAL_ITEMS_KEYS_ORDER = {k: i for i, k in enumerate(CANONICAL_ITEMS.keys())}


def find_item_start_pages(pages: list[dict]) -> dict[str, int]:
    """For each canonical item id, find the first physical page that contains its
    ALL-CAPS section heading. The TOC uses title-case so it naturally doesn't match.
    """
    starts: dict[str, int] = {}
    for page in pages:
        page_num = page["page"]
        for m in ITEM_RE.finditer(page["content"]):
            item_id = m.group(1).upper()
            if item_id in CANONICAL_ITEMS and item_id not in starts:
                starts[item_id] = page_num
    return starts


def build_structure(starts: dict[str, int]) -> list[dict]:
    """Tree: list of Part nodes each containing Item nodes."""
    parts: dict[str, dict] = {}

    for item_id, (part_name, title) in CANONICAL_ITEMS.items():
        if item_id not in starts:
            continue
        page = starts[item_id]
        if part_name not in parts:
            parts[part_name] = {"title": part_name, "physical_index": page, "nodes": []}
        parts[part_name]["nodes"].append({"title": title, "physical_index": page})

    for part_node in parts.values():
        if part_node["nodes"]:
            part_node["physical_index"] = min(n["physical_index"] for n in part_node["nodes"])
        part_node["nodes"].sort(key=lambda n: n["physical_index"])

    ordered_parts = sorted(parts.values(), key=lambda p: p["physical_index"])

    counter = [0]
    def assign(nodes):
        for n in nodes:
            n["node_id"] = str(counter[0]).zfill(4)
            counter[0] += 1
            if n.get("nodes"):
                assign(n["nodes"])
    assign(ordered_parts)
    return ordered_parts


def page_range_for_item(item_id: str, starts: dict[str, int], total_pages: int) -> tuple[int, int]:
    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    for i, (iid, pg) in enumerate(ordered):
        if iid == item_id:
            # Skip past any sibling items that share this same start page.
            j = i + 1
            while j < len(ordered) and ordered[j][1] <= pg:
                j += 1
            next_pg = ordered[j][1] - 1 if j < len(ordered) else total_pages
            return (pg, max(pg, next_pg))
    return (1, total_pages)


async def summarize_section(title: str, body_text: str) -> str:
    prompt = (
        f'Section title: "{title}"\n\n'
        f"Section excerpt (first ~3000 chars):\n{body_text[:3000]}\n\n"
        "Write ONE sentence (max 30 words) describing what THIS specific section discusses. "
        "Do not restate the title verbatim. Be specific about the actual content."
    )
    result = await llm_acompletion(config.INDEX_MODEL, prompt)
    return result.strip().replace("\n", " ")[:300]


async def add_summaries(structure: list[dict], pages: list[dict], starts: dict[str, int], total_pages: int):
    page_map = {p["page"]: p["content"] for p in pages}

    async def walk(nodes):
        for n in nodes:
            if n["title"].startswith("Item "):
                item_id = n["title"].split(".")[0].replace("Item ", "").strip()
                start, end = page_range_for_item(item_id, starts, total_pages)
                body = "\n".join(page_map.get(p, "") for p in range(start, min(start + 3, end + 1)))
                print(f"  summarize {n['title'][:60]}... (pp.{start}-{end})", flush=True)
                t0 = time.time()
                try:
                    summary = await summarize_section(n["title"], body)
                    n["summary"] = summary
                    print(f"    -> {summary[:120]} ({time.time()-t0:.1f}s)", flush=True)
                except Exception as e:
                    print(f"    -> SKIP ({e})", flush=True)
            if n.get("nodes"):
                await walk(n["nodes"])
    await walk(structure)


def main():
    print(f"Loading {config.PDF_PATH}")
    doc = pymupdf.open(config.PDF_PATH)
    pages = [{"page": i + 1, "content": p.get_text() or ""} for i, p in enumerate(doc)]
    total_pages = len(pages)
    doc.close()
    print(f"  {total_pages} pages")

    print("Scanning for Item/Part headers (skipping TOC pages)...")
    starts = find_item_start_pages(pages)
    print(f"  found {len(starts)} items:")
    for item_id in sorted(starts.keys(), key=lambda x: CANONICAL_ITEMS_KEYS_ORDER.get(x, 99)):
        print(f"    Item {item_id} -> p.{starts[item_id]}: {CANONICAL_ITEMS[item_id][1][:60]}")

    print("Building tree structure...")
    structure = build_structure(starts)
    print(f"  {len(structure)} top-level parts, {sum(len(p['nodes']) for p in structure)} items total")

    print(f"Generating per-item summaries via {config.INDEX_MODEL}...")
    asyncio.run(add_summaries(structure, pages, starts, total_pages))

    doc_id = str(uuid.uuid4())
    document = {
        "id": doc_id,
        "type": "pdf",
        "path": str(config.PDF_PATH),
        "doc_name": "Tesla 10-K (FY 2024)",
        "doc_description": "Tesla, Inc. Form 10-K Annual Report for the fiscal year ended December 31, 2024, as filed with the U.S. Securities and Exchange Commission.",
        "page_count": total_pages,
        "structure": structure,
        "pages": pages,
    }

    config.PAGEINDEX_WORKSPACE.mkdir(parents=True, exist_ok=True)
    out = config.PAGEINDEX_WORKSPACE / f"{doc_id}.json"
    out.write_text(json.dumps(document, ensure_ascii=False, indent=2))

    meta = {doc_id: {
        "type": "pdf",
        "doc_name": document["doc_name"],
        "doc_description": document["doc_description"],
        "path": document["path"],
        "page_count": document["page_count"],
    }}
    (config.PAGEINDEX_WORKSPACE / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    config.PAGEINDEX_DOC_ID_FILE.write_text(doc_id)
    print(f"\nWrote tree → {out}")
    print(f"Wrote meta → {config.PAGEINDEX_WORKSPACE / '_meta.json'}")
    print(f"doc_id → {doc_id}")
    print("Done.")


if __name__ == "__main__":
    main()
