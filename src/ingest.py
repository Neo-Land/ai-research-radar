"""Download recent arXiv AI paper metadata into data/papers.jsonl.

Uses the public arXiv API (https://info.arxiv.org/help/api/), which asks
clients to wait 3 seconds between requests.

    python -m src.ingest --n 20000
"""
import argparse
import json
import re
import time
from pathlib import Path

import feedparser
import requests

API = "https://export.arxiv.org/api/query"
OUT = Path("data/papers.jsonl")
PAGE = 1000


def clean(text: str) -> str:
    """Collapse the hard line-wraps and extra spaces arXiv puts in titles/abstracts."""
    return re.sub(r"\s+", " ", text).strip()


def fetch_page(category: str, start: int) -> list[dict]:
    params = {
        "search_query": f"cat:{category}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": PAGE,
    }
    # Under load the API returns empty pages or refuses connections; back off and retry.
    entries = []
    for attempt in range(6):
        try:
            resp = requests.get(API, params=params, timeout=60)
            resp.raise_for_status()
            entries = feedparser.parse(resp.text).entries
        except requests.RequestException as err:
            print(f"  attempt {attempt + 1} failed: {err.__class__.__name__}")
        if entries:
            break
        time.sleep(30 * (attempt + 1))
    return [
        {
            "id": e.id.rsplit("/abs/", 1)[-1].split("v")[0],  # 2410.12345v2 -> 2410.12345
            "title": clean(e.title),
            "abstract": clean(e.summary),
            "authors": [a.name for a in e.authors],
            "categories": [t.term for t in e.tags],
            "published": e.published[:10],
            "url": e.id,
        }
        for e in entries
    ]


def save(papers: dict[str, dict]) -> None:
    with OUT.open("w") as f:
        for p in papers.values():
            f.write(json.dumps(p) + "\n")


def main(n: int, category: str) -> None:
    OUT.parent.mkdir(exist_ok=True)
    papers: dict[str, dict] = {}
    for start in range(0, n, PAGE):
        page = fetch_page(category, start)
        if not page:
            print(f"no results at {start}, stopping")
            break
        for p in page:
            papers.setdefault(p["id"], p)  # pages can overlap as new papers arrive
        save(papers)  # save every page so a late failure loses nothing
        print(f"{start + len(page):>6} fetched, {len(papers)} unique", flush=True)
        time.sleep(3)
    print(f"wrote {len(papers)} papers to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--category", default="cs.AI")
    args = ap.parse_args()
    main(args.n, args.category)
