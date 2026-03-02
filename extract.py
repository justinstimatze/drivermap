#!/usr/bin/env python3
"""
extract.py — LLM extraction loop for the behavioral mechanisms knowledge base.

Phase 2 (blind): Extract with minimal prompt; let schema emerge.
Phase 3 (guided): Extract using discovered schema (schema/v1.json).
Phase 3v (verify): Verifier pass — LLM checks extraction accuracy.

Uses `claude --print` via subprocess (Max plan, no API key needed).

Usage:
    python extract.py --blind                    # phase 2: blind extraction
    python extract.py --blind --limit 5          # first 5 only
    python extract.py --guided                   # phase 3: schema-guided
    python extract.py --verify                   # verifier pass on extracted/
    python extract.py --id loss_aversion --blind # single mechanism
    python extract.py --id loss_aversion --guided
    python extract.py --rerun-flagged            # re-extract low-quality records
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
CORPUS_DIR = ROOT / "corpus"
EXTRACTED_DIR = ROOT / "extracted"
SCHEMA_DIR = ROOT / "schema"


# ─── Claude CLI invocation ────────────────────────────────────────────────────


def call_claude(
    prompt: str, json_schema: dict = None, model: str = "sonnet", timeout: int = 120
) -> dict:
    """
    Call claude CLI in --print mode and return parsed JSON result.

    Returns {"ok": bool, "text": str, "parsed": dict|None, "error": str|None}
    """
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
    ]
    if json_schema:
        cmd.extend(["--json-schema", json.dumps(json_schema)])

    try:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": "", "parsed": None, "error": f"timeout after {timeout}s"}
    except FileNotFoundError:
        return {"ok": False, "text": "", "parsed": None, "error": "claude CLI not found in PATH"}

    if result.returncode != 0:
        return {
            "ok": False,
            "text": result.stderr,
            "parsed": None,
            "error": f"claude exited {result.returncode}: {result.stderr[:200]}",
        }

    # --output-format json wraps the result in a JSON envelope
    parsed = None
    try:
        envelope = json.loads(result.stdout)
        # When --json-schema is used, Claude puts structured output here
        if "structured_output" in envelope and envelope["structured_output"]:
            parsed = envelope["structured_output"]
            text = json.dumps(parsed)
        else:
            text = envelope.get("result", "") or result.stdout
    except json.JSONDecodeError:
        text = result.stdout

    # Try to parse text as JSON if not already parsed
    if parsed is None and text:
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            clean = inner.strip()
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            pass

    return {"ok": True, "text": text, "parsed": parsed, "error": None}


# ─── Prompts ──────────────────────────────────────────────────────────────────

BLIND_PROMPT_TEMPLATE = """\
You are analyzing research text about a behavioral/psychological mechanism.

Mechanism: {name} (domain: {domain})

Source material:
{source_text}

---

Extract whatever structured information you can find about this behavioral mechanism.
Do NOT follow a fixed schema — just capture what's actually present in the text.
Be specific to this mechanism; don't pad with generic filler.

Think about: what triggers this mechanism, what behaviors result, what moderates it,
what evidence quality exists, how it varies between individuals, what it interacts with.
But only include fields you actually have information for.

Output valid JSON only. No explanation outside the JSON.
"""

GUIDED_PROMPT_TEMPLATE = """\
You are extracting structured data about a behavioral/psychological mechanism.

Mechanism: {name} (domain: {domain})

Source material:
{source_text}

---

Schema guide (use as a starting point, not a constraint):
{schema_str}

Extract a JSON record for this mechanism following the schema where the source material
supports it. For fields with no information in the source, omit them rather than guessing.
Use the "notes" field for anything important that doesn't fit the schema.
For interactions[], include only those explicitly supported by the text.

Additionally, always include these two fields using ONLY the dimension keys defined below —
do not invent dimension names:

"person_moderators": which person-level dimensions amplify (+), dampen (-), or
  shift the mechanism. Use only keys from this vocabulary:
  TRAIT: big_five_O, big_five_C, big_five_E, big_five_A, big_five_N,
         dark_triad_narcissism, dark_triad_machiavellianism, dark_triad_psychopathy,
         attachment_anxious, attachment_avoidant, bis_sensitivity, bas_sensitivity,
         need_for_cognition, need_for_closure, disgust_sensitivity,
         sensory_processing_sensitivity, alexithymia, social_dominance_orientation,
         hexaco_H
  STATE: cognitive_load, affective_valence, affective_arousal, resource_scarcity_state,
         power_state, threat_appraisal, in_group_salience, relationship_history,
         fatigue_depletion
  Format each entry as: {{"dimension": "<key>", "direction": "+|-|mixed",
    "strength": "weak|moderate|strong", "note": "<1-sentence evidence basis>"}}
  Only include dimensions where the source text or well-established research supports
  the moderation. Do not encode demographic stereotypes.

"situation_activators": which situational features trigger or modulate this mechanism.
  Use only keys from this vocabulary:
  stakes, social_visibility, time_pressure, ambiguity, out_group_salience,
  power_differential, resource_availability, novelty, relationship_type,
  conflict_present, anonymity, social_norms_clarity, physical_threat,
  group_context, outcome_reversibility, prior_commitment, surveillance
  Format each entry as: {{"feature": "<key>", "effect": "activates|amplifies|dampens|required",
    "note": "<1-sentence basis>"}}

"plain_language_outputs": a JSON array of 6-10 short everyday English phrases (1-4 words each)
  describing what this mechanism produces in observable behavior — how a friend would describe
  the behavior, not how a textbook would.
  GOOD: "feels grateful", "helps others", "avoids the situation", "cries", "gets angry",
        "apologizes", "takes bigger risks", "steps back"
  BAD:  "prosocial behavioral output", "reparatory cognition", "behavioral inhibition response"
  Always include this field.

"narrative_outputs": a JSON array of 3-5 mechanism-level behavioral descriptions (1-2 sentences
  each) suitable for embedding in an LLM system prompt to guide NPC behavior. Use clinical/
  diagnostic register, not casual. Describe the active psychological drive, not the character.
  GOOD: "status-maintenance drive active — monitors for challenges to authority"
  GOOD: "loss-frame dominates evaluation — asymmetric weighting of potential losses vs gains"
  BAD:  "feels threatened" (too casual, that's PLO territory)
  BAD:  "the character is afraid" (character-level, not mechanism-level)
  Always include this field.

Output valid JSON only. No explanation outside the JSON.
"""

VERIFIER_PROMPT_TEMPLATE = """\
You are a verifier checking whether an extracted record accurately represents its source.

Mechanism: {name}

Source material:
{source_text}

Extracted record:
{extracted_json}

---

Check: does the extracted record accurately represent what the source says?
Flag any fields that are:
- Inaccurate (contradicts the source)
- Hallucinated (not in the source)
- Missing (clearly present in source but absent from record)

IMPORTANT EXCEPTION: Do NOT flag "person_moderators", "situation_activators", or
"plain_language_outputs" as hallucinated. These are intentional synthesis fields — the first
two draw on established cross-mechanism research; the last is a vocabulary translation of the
behavioral outputs into everyday English. Only flag person_moderators/situation_activators if
they actively contradict the source.

Output JSON:
{{
  "passes": true/false,
  "accuracy_score": 0.0-1.0,
  "issues": [
    {{"field": "...", "type": "inaccurate|hallucinated|missing", "detail": "..."}}
  ],
  "notes": "..."
}}
"""


# ─── Prompt repetition ────────────────────────────────────────────────────────
# Based on Leviathan et al. 2025 "Prompt Repetition Improves Non-Reasoning LLMs":
# causal LLMs only attend to prior tokens; repeating the prompt lets the second
# copy attend to all tokens in the first, approximating bidirectional attention.


def with_prompt_repetition(text: str) -> str:
    return f"{text}\n\nLet me repeat the task:\n{text}"


# ─── Source text assembly ─────────────────────────────────────────────────────


def assemble_source_text(corpus_record: dict, max_chars: int = 6000) -> str:
    """Combine Wikipedia + Kagi snippets into a single text block."""
    parts = []
    for src in corpus_record.get("sources", []):
        text = src.get("text", "").strip()
        if not text:
            continue
        title = src.get("title", src.get("type", "source"))
        src_type = src.get("type", "")
        if src_type == "wikipedia":
            parts.append(f"=== Wikipedia: {title} ===\n{text}")
        elif src_type == "kagi_search":
            url = src.get("url", "")
            parts.append(f"=== Paper/Source: {title}\nURL: {url}\n{text}")
        else:
            parts.append(f"=== Source: {title} ===\n{text}")

    combined = "\n\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n\n[... truncated ...]"
    return combined


# ─── Extraction ───────────────────────────────────────────────────────────────


def load_schema() -> dict | None:
    """Load the latest schema version if it exists."""
    schema_files = sorted(SCHEMA_DIR.glob("v*.json"))
    if not schema_files:
        return None
    return json.loads(schema_files[-1].read_text())


def extract_blind(corpus_record: dict) -> dict:
    """Phase 2: blind extraction with minimal prompt."""
    source_text = assemble_source_text(corpus_record)
    prompt = BLIND_PROMPT_TEMPLATE.format(
        name=corpus_record["name"],
        domain=corpus_record["domain"],
        source_text=source_text,
    )
    return call_claude(with_prompt_repetition(prompt))


def extract_guided(corpus_record: dict, schema: dict) -> dict:
    """Phase 3: schema-guided extraction."""
    source_text = assemble_source_text(corpus_record)
    schema_str = json.dumps(schema, indent=2)
    prompt = GUIDED_PROMPT_TEMPLATE.format(
        name=corpus_record["name"],
        domain=corpus_record["domain"],
        source_text=source_text,
        schema_str=schema_str,
    )
    return call_claude(with_prompt_repetition(prompt))


def verify_extraction(corpus_record: dict, extracted: dict) -> dict:
    """Verifier pass: check extraction accuracy."""
    source_text = assemble_source_text(corpus_record)
    extracted_json = json.dumps(extracted.get("extraction", {}), indent=2)

    verifier_schema = {
        "type": "object",
        "properties": {
            "passes": {"type": "boolean"},
            "accuracy_score": {"type": "number"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "type": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["passes", "accuracy_score"],
    }

    prompt = VERIFIER_PROMPT_TEMPLATE.format(
        name=corpus_record["name"],
        source_text=source_text,
        extracted_json=extracted_json,
    )
    return call_claude(prompt, json_schema=verifier_schema)


FIX_PROMPT_TEMPLATE = """\
You are fixing a previously extracted record for a behavioral/psychological mechanism.
A verifier found issues with the original extraction. Fix ONLY the flagged issues — keep
everything else identical.

Mechanism: {name} (domain: {domain})

Source material:
{source_text}

Previous extraction:
{extracted_json}

Verifier issues:
{issues_text}

---

Return the COMPLETE corrected JSON record. For each flagged issue:
- "hallucinated": remove the hallucinated content or replace with source-supported content
- "inaccurate": correct to match the source
- "missing": add the missing information from the source

Do NOT change fields that were not flagged. Keep person_moderators, situation_activators,
plain_language_outputs, and narrative_outputs as-is unless they were specifically flagged.

Output valid JSON only. No explanation outside the JSON.
"""


def fix_extraction(corpus_record: dict, extracted: dict) -> dict:
    """Fix a flagged extraction by feeding verifier issues back to the LLM."""
    source_text = assemble_source_text(corpus_record)
    extraction = extracted.get("extraction", {})
    extracted_json = json.dumps(extraction, indent=2)
    verification = extracted.get("verification", {})
    issues = verification.get("issues", [])

    issues_text = "\n".join(
        f"- [{issue['type']}] {issue['field']}: {issue['detail']}" for issue in issues
    )

    prompt = FIX_PROMPT_TEMPLATE.format(
        name=corpus_record["name"],
        domain=corpus_record["domain"],
        source_text=source_text,
        extracted_json=extracted_json,
        issues_text=issues_text,
    )
    return call_claude(with_prompt_repetition(prompt))


# ─── Save / load ─────────────────────────────────────────────────────────────


def save_extracted(mechanism_id: str, record: dict) -> Path:
    EXTRACTED_DIR.mkdir(exist_ok=True)
    path = EXTRACTED_DIR / f"{mechanism_id}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


def load_extracted(mechanism_id: str) -> dict | None:
    path = EXTRACTED_DIR / f"{mechanism_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_corpus(mechanism_id: str) -> dict | None:
    path = CORPUS_DIR / f"{mechanism_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


# ─── CLI ─────────────────────────────────────────────────────────────────────


def get_corpus_ids() -> list[str]:
    return sorted(p.stem for p in CORPUS_DIR.glob("*.json"))


def main():
    parser = argparse.ArgumentParser(description="LLM extraction for behavioral mechanisms")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--blind", action="store_true", help="Phase 2: blind extraction (no schema)")
    mode.add_argument("--guided", action="store_true", help="Phase 3: schema-guided extraction")
    mode.add_argument("--verify", action="store_true", help="Verifier pass on extracted records")
    mode.add_argument(
        "--fix", action="store_true", help="Fix flagged records using verifier feedback"
    )

    parser.add_argument("--id", help="Process single mechanism by ID")
    parser.add_argument("--domain", help="Process one domain only")
    parser.add_argument("--limit", type=int, help="Max number to process")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip already-extracted mechanisms"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Seconds between LLM calls (default 2.0)"
    )
    args = parser.parse_args()

    # Determine which corpus IDs to process
    if args.id:
        ids = [args.id]
    else:
        ids = get_corpus_ids()

    if args.domain:
        from seeds import SEEDS

        domain_ids = {s["id"] for s in SEEDS if s["domain"] == args.domain}
        ids = [i for i in ids if i in domain_ids]

    if args.skip_existing and not args.verify and not args.fix:
        ids = [i for i in ids if not (EXTRACTED_DIR / f"{i}.json").exists()]

    if args.limit:
        ids = ids[: args.limit]

    if not ids:
        print("No mechanisms to process.")
        return

    # Load schema for guided mode
    schema = None
    if args.guided:
        schema = load_schema()
        if schema is None:
            print("ERROR: No schema found in schema/. Run discover.py first.", file=sys.stderr)
            sys.exit(1)
        print(f"Using schema: {sorted(SCHEMA_DIR.glob('v*.json'))[-1].name}")

    print(f"\nProcessing {len(ids)} mechanism(s)...\n")

    passed = 0
    flagged = 0

    for i, mid in enumerate(ids, 1):
        corpus = load_corpus(mid)
        if corpus is None:
            print(f"[{i}] {mid}: no corpus file, skipping")
            continue

        print(f"[{i}/{len(ids)}] {corpus['name']} ({mid})")

        # ── Verify mode ──
        if args.verify:
            extracted = load_extracted(mid)
            if extracted is None:
                print("  → no extraction yet, skipping")
                continue
            print("  → verifying...")
            result = verify_extraction(corpus, extracted)
            if not result["ok"]:
                print(f"  ✗ claude error: {result['error']}")
                continue
            v = result["parsed"]
            if v:
                passes = v.get("passes", False)
                score = v.get("accuracy_score", 0)
                issues = v.get("issues", [])
                print(f"  → passes={passes}, score={score:.2f}, issues={len(issues)}")
                if issues:
                    for issue in issues[:3]:
                        print(f"    [{issue['type']}] {issue['field']}: {issue['detail'][:80]}")
                # Save verification result into the extracted record
                extracted["verification"] = v
                save_extracted(mid, extracted)
                if passes:
                    passed += 1
                else:
                    flagged += 1
                    print("  ⚑ flagged for re-extraction")
            else:
                print(f"  ? verifier output not parseable: {result['text'][:200]}")
            time.sleep(args.delay)
            continue

        # ── Fix flagged mode ──
        if args.fix:
            extracted = load_extracted(mid)
            if extracted is None:
                continue
            v = extracted.get("verification", {})
            if v.get("passes", True):
                continue  # not flagged
            issues = v.get("issues", [])
            print(f"  → fixing {len(issues)} issue(s)...")
            for issue in issues[:5]:
                print(f"    [{issue['type']}] {issue['field']}: {issue['detail'][:80]}")

            result = fix_extraction(corpus, extracted)
            if not result["ok"]:
                print(f"  ✗ error: {result['error']}")
                continue

            fixed = result["parsed"] or {"raw_text": result["text"]}

            # Preserve fields the fix prompt shouldn't have touched
            prev = extracted.get("extraction", {})
            for keep_field in (
                "person_moderators",
                "situation_activators",
                "plain_language_outputs",
                "narrative_outputs",
            ):
                if keep_field in prev and keep_field not in fixed:
                    fixed[keep_field] = prev[keep_field]

            record = {
                "id": mid,
                "name": corpus["name"],
                "domain": corpus["domain"],
                "phase": "fix",
                "extraction": fixed,
                "previous_verification": v,
            }
            path = save_extracted(mid, record)
            print(f"  → saved fix: {path.name}")
            flagged += 1
            time.sleep(args.delay)
            continue

        # ── Extraction (blind or guided) ──
        if args.guided:
            result = extract_guided(corpus, schema)
            phase = "guided"
        else:
            result = extract_blind(corpus)
            phase = "blind"

        if not result["ok"]:
            print(f"  ✗ error: {result['error']}")
            continue

        extraction = result["parsed"] or {"raw_text": result["text"]}

        record = {
            "id": mid,
            "name": corpus["name"],
            "domain": corpus["domain"],
            "phase": phase,
            "extraction": extraction,
        }
        path = save_extracted(mid, record)
        print(f"  → saved: {path.name}")
        if result["parsed"]:
            fields = list(result["parsed"].keys())
            print(f"  → fields: {fields}")
        else:
            print("  ⚠ extraction not parseable as JSON (saved as raw_text)")

        time.sleep(args.delay)

    print("\nDone.")
    if args.verify:
        total = passed + flagged
        print(f"Verification: {passed}/{total} passed, {flagged} flagged")
        if total > 0:
            rate = passed / total
            print(f"Agreement rate: {rate:.1%}")
            if rate < 0.8:
                print("⚠ Agreement <80% — consider refining extraction prompt")
    elif args.fix:
        print(f"Fixed: {flagged} record(s). Run --verify again to check.")


if __name__ == "__main__":
    main()
