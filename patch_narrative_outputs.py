#!/usr/bin/env python3
"""
patch_narrative_outputs.py — Add narrative_outputs to extracted records.

For each mechanism, reads its existing extraction and calls Claude to produce
3-5 clinical-register mechanism descriptions suitable for LLM system prompts.

Uses the same claude --print subprocess as extract.py (Max plan, no API key).

Usage:
    python patch_narrative_outputs.py                   # patch all
    python patch_narrative_outputs.py --id loss_aversion  # single mechanism
    python patch_narrative_outputs.py --skip-existing   # skip already-patched
    python patch_narrative_outputs.py --top N           # only top N most-connected
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
EXTRACTED_DIR = ROOT / "extracted"
sys.path.insert(0, str(ROOT))
from extract import call_claude  # noqa: E402

PATCH_PROMPT_TEMPLATE = """\
Generate mechanism-level behavioral descriptions for embedding in an LLM system prompt.

Mechanism: {name}
Domain: {domain}
Definition: {definition}
Plain-language outputs: {plo}

---

Generate 3-5 mechanism-level behavioral descriptions (1-2 sentences each) suitable for
embedding in an LLM system prompt to guide NPC behavior. Use clinical/diagnostic register,
not casual. Describe the active psychological drive, not the character.

Style guide:
  GOOD: "status-maintenance drive active — monitors for challenges to authority"
  GOOD: "loss-frame dominates evaluation — asymmetric weighting of potential losses vs gains"
  GOOD: "in-group boundary enforcement engaged — heightened vigilance toward loyalty signals"
  BAD:  "feels threatened" (too casual, that's PLO territory)
  BAD:  "the character is afraid" (character-level, not mechanism-level)
  BAD:  "loss aversion" (just naming the mechanism, not describing the drive)

Each description should:
- Name the active psychological process/drive
- Describe what it produces in terms of cognition/attention/evaluation
- Use em-dashes to separate the process from its behavioral signature

Output a JSON array of strings only. No commentary.
"""


def patch_one(path: Path, force: bool = False) -> bool:
    """Patch a single extraction file. Returns True if patched."""
    data = json.loads(path.read_text())
    mid = path.stem
    ext = data.get("extraction", {})

    if not force and ext.get("narrative_outputs"):
        return False

    name = data.get("name", mid)
    domain = data.get("domain", "unknown")
    definition = data.get("description") or data.get("summary") or ""
    plo = data.get("plain_language_outputs", [])
    if isinstance(plo, list):
        plo = ", ".join(plo[:8])

    prompt = PATCH_PROMPT_TEMPLATE.format(
        name=name, domain=domain, definition=definition[:500], plo=plo
    )

    result = call_claude(prompt, model="sonnet", timeout=60)
    if not result["ok"]:
        print(f"  ERROR {mid}: {result['error']}", file=sys.stderr)
        return False

    parsed = result.get("parsed")
    if parsed is None:
        # Try to extract JSON array from text
        text = result["text"].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to find array in text
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    print(f"  ERROR {mid}: could not parse response", file=sys.stderr)
                    return False

    if not isinstance(parsed, list) or len(parsed) < 2:
        print(f"  ERROR {mid}: expected list of 3-5 items, got {type(parsed)}", file=sys.stderr)
        return False

    if "extraction" not in data:
        data["extraction"] = {}
    data["extraction"]["narrative_outputs"] = parsed
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  OK {mid}: {len(parsed)} narrative outputs")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Patch a single mechanism by ID")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already-patched")
    parser.add_argument("--top", type=int, help="Only patch top N most-connected mechanisms")
    args = parser.parse_args()

    if args.id:
        path = EXTRACTED_DIR / f"{args.id}.json"
        if not path.exists():
            print(f"Not found: {path}", file=sys.stderr)
            sys.exit(1)
        patch_one(path, force=not args.skip_existing)
        return

    paths = sorted(EXTRACTED_DIR.glob("*.json"))
    if args.top:
        # Sort by number of person_moderators + situation_activators (proxy for "most-connected")
        def connectivity(p):
            d = json.loads(p.read_text())
            return len(d.get("person_moderators", [])) + len(d.get("situation_activators", []))

        paths = sorted(paths, key=connectivity, reverse=True)[: args.top]

    total = 0
    patched = 0
    for path in paths:
        total += 1
        if patch_one(path, force=not args.skip_existing):
            patched += 1
            time.sleep(0.5)  # rate limit

    print(f"\nDone: {patched}/{total} patched")


if __name__ == "__main__":
    main()
