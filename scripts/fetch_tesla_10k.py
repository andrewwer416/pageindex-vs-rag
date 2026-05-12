"""Fetch Tesla's 2024 10-K from SEC EDGAR and render to PDF.

SEC requires a User-Agent header with contact info. EDGAR serves the
filing as inline-XBRL HTML; we strip the hidden XBRL noise and render
to PDF via WeasyPrint so PageIndex's PDF parsing has clean text.

Usage:
    python scripts/fetch_tesla_10k.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
HTM = DATA / "tsla-10k-2024.htm"
PDF = DATA / "tsla-10k-2024.pdf"
URL = "https://www.sec.gov/Archives/edgar/data/1318605/000162828025003063/tsla-20241231.htm"
USER_AGENT = "pageindex-vs-rag demo (https://github.com/anthropics/claude-code)"


def fetch():
    import urllib.request
    if HTM.exists() and HTM.stat().st_size > 2_000_000:
        print(f"  HTM cached at {HTM} ({HTM.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  downloading {URL} ...")
    req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    HTM.write_bytes(data)
    print(f"  saved {len(data)/1e6:.1f} MB to {HTM}")


def render():
    import weasyprint
    if PDF.exists() and PDF.stat().st_size > 100_000:
        print(f"  PDF cached at {PDF} ({PDF.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  rendering {HTM} -> {PDF}  (WeasyPrint, ~30s)")
    weasyprint.HTML(filename=str(HTM)).write_pdf(str(PDF))
    print(f"  saved {PDF.stat().st_size/1e6:.1f} MB to {PDF}")


if __name__ == "__main__":
    print("[1/2] Fetching Tesla 10-K HTM from SEC EDGAR")
    fetch()
    print("[2/2] Rendering HTM to PDF (drops inline-XBRL hidden divs)")
    render()
    print("Done.")
