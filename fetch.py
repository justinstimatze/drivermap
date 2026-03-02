#!/usr/bin/env python3
"""
fetch.py — Corpus fetcher for the behavioral mechanisms knowledge base.

For each mechanism in seeds.py:
  1. Fetch Wikipedia article via Kiwix HTTP server (localhost:8080)
  2. Fetch 2-3 paper abstracts via Kagi search API
  3. Store raw text to corpus/{mechanism_id}.json

Usage:
    python fetch.py                          # fetch all seeds
    python fetch.py --id loss_aversion       # fetch one mechanism
    python fetch.py --domain status_dominance
    python fetch.py --skip-existing          # skip already-fetched
    python fetch.py --wikipedia-only         # skip Kagi search
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parent
CORPUS_DIR = ROOT / "corpus"

# Load from .env if present
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            v = v.split("#")[0]  # strip inline comments
            os.environ.setdefault(k.strip(), v.strip())

KIWIX_URL = os.environ.get("KIWIX_URL", "http://localhost:8080")
KIWIX_BOOK = os.environ.get("KIWIX_BOOK", "wikipedia_en_all_nopic_2025-12")
KAGI_KEY = os.environ.get("KAGI_API_KEY", "")


# ─── HTML → plain text ────────────────────────────────────────────────────────


class _TextExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "sup", "table"}

    def __init__(self):
        super().__init__()
        self._skip = 0
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip += 1
        if tag in ("p", "h2", "h3", "li"):
            self._buf.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def get_text(self) -> str:
        raw = "".join(self._buf)
        lines = [ln.strip() for ln in raw.splitlines()]
        # collapse blank lines
        out, prev_blank = [], False
        for ln in lines:
            blank = not ln
            if blank and prev_blank:
                continue
            out.append(ln)
            prev_blank = blank
        return "\n".join(out).strip()


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    return p.get_text()


# ─── Kiwix ────────────────────────────────────────────────────────────────────


def _kiwix_get(path: str, timeout: int = 15) -> str | None:
    url = f"{KIWIX_URL}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    [kiwix] {e}", file=sys.stderr)
        return None


def _title_to_path(title: str) -> str:
    return title.replace(" ", "_")


def fetch_wikipedia(article_title: str) -> dict:
    """Fetch a Wikipedia article from the local Kiwix server."""
    slug = _title_to_path(article_title)
    html = _kiwix_get(f"/content/{KIWIX_BOOK}/{slug}")

    if html is None:
        # Try URL-encoded form
        encoded = urllib.parse.quote(slug)
        html = _kiwix_get(f"/content/{KIWIX_BOOK}/{encoded}")

    if html is None:
        return {"title": article_title, "text": "", "found": False, "note": "kiwix fetch failed"}

    text = html_to_text(html)

    # Kiwix 404 pages are short
    if len(text) < 200:
        return {
            "title": article_title,
            "text": text,
            "found": False,
            "note": "article not found or too short",
        }

    # Trim to ~6000 chars at a paragraph boundary
    if len(text) > 6000:
        cutoff = text.rfind("\n", 0, 6000)
        text = text[: cutoff if cutoff > 4000 else 6000]

    return {"title": article_title, "text": text.strip(), "found": True}


# ─── Kagi ─────────────────────────────────────────────────────────────────────


def _kagi_summarize(url: str, timeout: int = 30) -> str:
    """Summarize a URL via the Kagi Summarizer API. Returns text or empty string."""
    if not KAGI_KEY:
        return ""
    params = urllib.parse.urlencode({"url": url, "summary_type": "summary"})
    req = urllib.request.Request(
        f"https://kagi.com/api/v0/summarize?{params}",
        headers={"Authorization": f"Bot {KAGI_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return data.get("data", {}).get("output", "") or ""
    except Exception as e:
        print(f"    [kagi-summarize] {e}", file=sys.stderr)
        return ""


def fetch_kagi(query: str) -> list[dict]:
    """Search Kagi, then summarize top URLs for substantive content."""
    if not KAGI_KEY:
        print("    [kagi] no KAGI_API_KEY set", file=sys.stderr)
        return []

    # Step 1: search
    params = urllib.parse.urlencode({"q": query, "limit": 6})
    url = f"https://kagi.com/api/v0/search?{params}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bot {KAGI_KEY}"})

    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"    [kagi] {e}", file=sys.stderr)
        return []

    candidates = []
    for item in data.get("data", []):
        if item.get("t") != 0:  # type 0 = search result
            continue
        candidates.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            }
        )
        if len(candidates) >= 4:
            break

    # Step 2: summarize top 3
    results = []
    for c in candidates[:3]:
        print(f"      summarizing: {c['url'][:70]}")
        summary = _kagi_summarize(c["url"])
        text = summary if summary else c["snippet"]
        results.append(
            {
                "title": c["title"],
                "url": c["url"],
                "snippet": text,
            }
        )
        time.sleep(0.3)

    return results


# ─── Core fetch logic ─────────────────────────────────────────────────────────


def fetch_mechanism(seed: dict, wikipedia_only: bool = False) -> dict:
    record = {
        "id": seed["id"],
        "name": seed["name"],
        "domain": seed["domain"],
        "sources": [],
    }

    # Wikipedia via Kiwix
    print(f"  → Wikipedia: '{seed['wikipedia']}'")
    wiki = fetch_wikipedia(seed["wikipedia"])
    status = "✓ found" if wiki["found"] else "✗ not found"
    print(f"    {status} ({len(wiki['text'])} chars)")
    record["sources"].append(
        {
            "type": "wikipedia",
            "title": wiki["title"],
            "text": wiki["text"],
            "found": wiki["found"],
            **({} if "note" not in wiki else {"note": wiki["note"]}),
        }
    )

    # Kagi paper search
    if not wikipedia_only:
        print(f"  → Kagi: '{seed['kagi_query']}'")
        results = fetch_kagi(seed["kagi_query"])
        print(f"    ✓ {len(results)} result(s)")
        for r in results:
            record["sources"].append(
                {
                    "type": "kagi_search",
                    "title": r["title"],
                    "url": r["url"],
                    "text": r["snippet"],
                }
            )
        time.sleep(0.5)  # be polite to the Kagi API

    return record


def save_corpus(record: dict) -> Path:
    CORPUS_DIR.mkdir(exist_ok=True)
    path = CORPUS_DIR / f"{record['id']}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Fetch corpus content for behavioral mechanisms")
    parser.add_argument("--id", help="Fetch single mechanism by ID")
    parser.add_argument("--domain", help="Fetch all mechanisms in a domain")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip mechanisms already in corpus/"
    )
    parser.add_argument("--wikipedia-only", action="store_true", help="Skip Kagi search")
    args = parser.parse_args()

    from seeds import SEEDS

    seeds = SEEDS
    if args.id:
        seeds = [s for s in seeds if s["id"] == args.id]
        if not seeds:
            print(f"Unknown ID: {args.id}", file=sys.stderr)
            sys.exit(1)
    elif args.domain:
        seeds = [s for s in seeds if s["domain"] == args.domain]
        if not seeds:
            print(f"Unknown domain: {args.domain}", file=sys.stderr)
            sys.exit(1)

    if args.skip_existing:
        seeds = [s for s in seeds if not (CORPUS_DIR / f"{s['id']}.json").exists()]

    print(f"\nFetching {len(seeds)} mechanism(s)...\n")

    ok = fail = 0
    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] {seed['name']} ({seed['id']})")
        try:
            record = fetch_mechanism(seed, wikipedia_only=args.wikipedia_only)
            path = save_corpus(record)
            print(f"  → saved: {path.name}\n")
            ok += 1
        except Exception as e:
            print(f"  ✗ error: {e}\n", file=sys.stderr)
            fail += 1

    print(f"Done. {ok} saved, {fail} failed.")


if __name__ == "__main__":
    main()
