#!/usr/bin/env python3
"""
patch_plain_language.py — Add plain_language_outputs to all extracted records.

For each mechanism, reads its existing extraction and calls Claude to produce
6-10 short everyday English phrases describing what the mechanism produces —
phrased in the commonsense vocabulary that ATOMIC uses (not academic language).

Uses the same claude --print subprocess as extract.py (Max plan, no API key).

Usage:
    python patch_plain_language.py                   # patch all
    python patch_plain_language.py --id gratitude    # single mechanism
    python patch_plain_language.py --skip-existing   # skip already-patched
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
EXTRACTED_DIR = ROOT / "extracted"
sys.path.insert(0, str(ROOT))
from extract import call_claude

PATCH_PROMPT_TEMPLATE = """\
Add a vocabulary bridge field to this psychological mechanism record.

Mechanism: {name}
Domain: {domain}
Definition: {definition}
Behavioral outputs (academic language): {outputs}

---

Generate a list of 6-10 short, everyday English phrases (1-4 words each) that
describe what happens when this mechanism activates. Phrase them as an ordinary
person would — matching the vocabulary of a commonsense knowledge base like ATOMIC.

Style guide:
  GOOD: "feels grateful", "helps others", "avoids the situation", "cries",
        "gets angry", "apologizes", "pays it forward", "seeks reassurance",
        "blames others", "withdraws", "feels proud", "takes risks",
        "seeks approval", "loses interest", "bonds with others"
  BAD:  "prosocial behavioral output", "reparatory cognition", "inhibition
        activation", "differential encoding of valenced stimuli"

Cover both internal states ("feels relieved") and external actions
("avoids conflict"). Include effects on the person and on others where relevant.

Output a JSON array only — no explanation, no prose, just the array.

Let me repeat the task:

Generate 6-10 plain everyday English phrases for: {name}
Academic outputs for context: {outputs}

JSON array only.
"""


def patch_mechanism(mid: str) -> bool:
    path = EXTRACTED_DIR / f"{mid}.json"
    if not path.exists():
        print(f"  {mid}: no extracted file, skipping")
        return False

    record = json.loads(path.read_text())
    ext = record.get("extraction", {})

    name = record.get("name") or ext.get("mechanism") or mid
    domain = record.get("domain") or ext.get("domain") or ""
    definition = (
        ext.get("definition")
        or ext.get("core_definition")
        or ext.get("core_claim")
        or ext.get("description")
        or ""
    )
    outputs = (
        ext.get("behavioral_outputs")
        or ext.get("outputs")
        or ext.get("behavioral_outcomes")
        or ext.get("resulting_behaviors")
        or ext.get("downstream_effects")
        or ""
    )
    if isinstance(outputs, dict):
        outputs = "; ".join(f"{k}: {v}" for k, v in outputs.items())
    elif isinstance(outputs, list):
        outputs = "; ".join(str(x) for x in outputs)

    prompt = PATCH_PROMPT_TEMPLATE.format(
        name=name,
        domain=domain,
        definition=str(definition)[:400],
        outputs=str(outputs)[:600],
    )

    result = call_claude(prompt, model="haiku", timeout=60)
    if not result["ok"]:
        print(f"  {mid}: error — {result.get('error')}")
        return False

    phrases = result["parsed"]
    if phrases is None:
        # Try parsing from raw text if JSON parsing failed
        text = result.get("text", "")
        try:
            phrases = json.loads(text)
        except Exception:
            print(f"  {mid}: could not parse output: {text[:100]!r}")
            return False

    if not isinstance(phrases, list) or not phrases:
        print(f"  {mid}: unexpected output type: {type(phrases)}")
        return False

    # Keep only strings, deduplicate, limit to 12
    phrases = list(dict.fromkeys(p for p in phrases if isinstance(p, str)))[:12]

    record["extraction"]["plain_language_outputs"] = phrases
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"  {mid}: {phrases[:6]}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add plain_language_outputs to extracted mechanism records"
    )
    parser.add_argument("--id", help="Patch single mechanism by ID")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip mechanisms that already have plain_language_outputs",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Seconds between calls (default 1.0)"
    )
    args = parser.parse_args()

    if args.id:
        ids = [args.id]
    else:
        ids = sorted(p.stem for p in EXTRACTED_DIR.glob("*.json"))

    if args.skip_existing:

        def has_field(mid):
            p = EXTRACTED_DIR / f"{mid}.json"
            if not p.exists():
                return False
            d = json.loads(p.read_text())
            return "plain_language_outputs" in d.get("extraction", {})

        ids = [i for i in ids if not has_field(i)]

    if not ids:
        print("Nothing to patch.")
        return

    print(f"\nPatching {len(ids)} mechanism(s) (using haiku for speed)...\n")
    ok = fail = 0
    for i, mid in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] {mid}")
        if patch_mechanism(mid):
            ok += 1
        else:
            fail += 1
        if i < len(ids):
            time.sleep(args.delay)

    print(f"\nDone. {ok} patched, {fail} failed.")


if __name__ == "__main__":
    main()
