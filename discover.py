#!/usr/bin/env python3
"""
discover.py — Schema discovery from blind extractions.

Analyzes extracted/*.json to find fields that emerged consistently.
  >80% presence  → required
  30-80%         → optional
  <30%           → escape hatch / notes

Outputs schema/v1.json (or schema/v2.json etc. if v1 already exists).

Usage:
    python discover.py              # analyze all extracted/
    python discover.py --show       # print schema without writing
    python discover.py --min 5      # minimum records to proceed (default 10)
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
EXTRACTED_DIR = ROOT / "extracted"
SCHEMA_DIR = ROOT / "schema"


# ─── Field walking ────────────────────────────────────────────────────────────


def walk_fields(obj, prefix="") -> list[str]:
    """
    Recursively extract field paths from a JSON object.
    Lists produce their element's fields with [] suffix.
    """
    fields = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            fields.append(path)
            fields.extend(walk_fields(v, path))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                fields.extend(walk_fields(item, f"{prefix}[]"))
    return fields


# ─── Type inference ───────────────────────────────────────────────────────────


def infer_type(values: list) -> str:
    """Guess the JSON type from observed values."""
    types = set()
    for v in values:
        if v is None:
            types.add("null")
        elif isinstance(v, bool):
            types.add("boolean")
        elif isinstance(v, int | float):
            types.add("number")
        elif isinstance(v, str):
            types.add("string")
        elif isinstance(v, list):
            types.add("array")
        elif isinstance(v, dict):
            types.add("object")
    if len(types) == 1:
        return types.pop()
    if types == {"string", "null"}:
        return "string?"
    return "|".join(sorted(types))


def collect_values(records: list[dict], field_path: str) -> list:
    """Extract values for a dot-path field from a list of records."""
    values = []
    for rec in records:
        obj = rec.get("extraction", rec)
        parts = field_path.split(".")
        cur = obj
        for part in parts:
            if part.endswith("[]"):
                part = part[:-2]
            if isinstance(cur, dict):
                cur = cur.get(part)
            elif isinstance(cur, list):
                cur = [item.get(part) for item in cur if isinstance(item, dict)]
            else:
                cur = None
                break
        if cur is not None:
            if isinstance(cur, list):
                values.extend(cur)
            else:
                values.append(cur)
    return values


# ─── Discovery ────────────────────────────────────────────────────────────────


def discover_schema(records: list[dict]) -> dict:
    """
    Analyze extracted records to discover the emergent schema.

    Returns a schema dict with required, optional, and notes categories.
    """
    n = len(records)
    if n == 0:
        return {}

    # Count field presence across records
    field_counts = Counter()
    for rec in records:
        extraction = rec.get("extraction", rec)
        if isinstance(extraction, dict):
            seen = set()
            for field in walk_fields(extraction):
                # Normalize: strip [] from array element paths for top-level counting
                top = field.split(".")[0].rstrip("[]")
                seen.add(top)
                # Also track full path
                seen.add(field)
            for f in seen:
                field_counts[f] += 1

    # Categorize
    required = {}
    optional = {}
    escape_hatch = {}

    for field, count in field_counts.items():
        presence = count / n
        # Skip deeply nested paths for the top-level schema (keep as structure hints)
        depth = field.count(".")
        if depth > 1:
            continue

        values = collect_values(records, field)
        type_str = infer_type(values)

        entry = {
            "presence": round(presence, 2),
            "count": count,
            "total": n,
            "inferred_type": type_str,
        }

        if presence >= 0.80:
            required[field] = entry
        elif presence >= 0.30:
            optional[field] = entry
        else:
            escape_hatch[field] = entry

    # Collect nested structure hints for complex fields
    structure_hints = defaultdict(set)
    for rec in records:
        extraction = rec.get("extraction", rec)
        if isinstance(extraction, dict):
            for field in walk_fields(extraction):
                parts = field.split(".")
                if len(parts) == 2:
                    parent = parts[0].rstrip("[]")
                    child = parts[1].rstrip("[]")
                    structure_hints[parent].add(child)

    # Build schema output
    schema = {
        "_meta": {
            "records_analyzed": n,
            "total_fields_seen": len(field_counts),
            "thresholds": {
                "required": ">= 80%",
                "optional": "30-80%",
                "escape_hatch": "< 30%",
            },
        },
        "required": required,
        "optional": optional,
        "escape_hatch": escape_hatch,
        "structure_hints": {k: sorted(v) for k, v in structure_hints.items()},
    }

    return schema


def print_schema(schema: dict):
    meta = schema.get("_meta", {})
    n = meta.get("records_analyzed", 0)
    print(f"\nSchema discovered from {n} records")
    print(f"Total unique fields seen: {meta.get('total_fields_seen', '?')}\n")

    print("─── REQUIRED (≥80%) ───────────────────────────────────")
    for field, info in sorted(schema.get("required", {}).items()):
        pct = f"{info['presence'] * 100:.0f}%"
        print(f"  {field:<35} {pct:>5}  ({info['inferred_type']})")

    print("\n─── OPTIONAL (30-80%) ─────────────────────────────────")
    for field, info in sorted(schema.get("optional", {}).items()):
        pct = f"{info['presence'] * 100:.0f}%"
        print(f"  {field:<35} {pct:>5}  ({info['inferred_type']})")

    print("\n─── RARE / ESCAPE HATCH (<30%) ────────────────────────")
    for field, info in sorted(schema.get("escape_hatch", {}).items()):
        pct = f"{info['presence'] * 100:.0f}%"
        print(f"  {field:<35} {pct:>5}  ({info['inferred_type']})")

    print("\n─── NESTED STRUCTURE HINTS ────────────────────────────")
    for parent, children in sorted(schema.get("structure_hints", {}).items()):
        print(f"  {parent}: {children}")


def save_schema(schema: dict) -> Path:
    SCHEMA_DIR.mkdir(exist_ok=True)
    # Find next version number
    existing = sorted(SCHEMA_DIR.glob("v*.json"))
    if existing:
        last = existing[-1].stem  # e.g. "v1"
        version = int(last[1:]) + 1
    else:
        version = 1
    path = SCHEMA_DIR / f"v{version}.json"
    path.write_text(json.dumps(schema, indent=2))
    return path


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Discover schema from blind extractions")
    parser.add_argument("--show", action="store_true", help="Print schema without writing to file")
    parser.add_argument("--min", type=int, default=10, help="Minimum records required (default 10)")
    parser.add_argument("--domain", help="Analyze only one domain")
    args = parser.parse_args()

    # Load all extracted records
    paths = sorted(EXTRACTED_DIR.glob("*.json"))
    if not paths:
        print("No extracted records found. Run extract.py --blind first.", file=sys.stderr)
        sys.exit(1)

    records = []
    for p in paths:
        try:
            rec = json.loads(p.read_text())
            # Only use blind extractions for discovery
            if rec.get("phase") not in ("blind", None):
                continue
            if args.domain and rec.get("domain") != args.domain:
                continue
            # Skip records with only raw_text (unparseable extractions)
            extraction = rec.get("extraction", {})
            if "raw_text" in extraction and len(extraction) == 1:
                print(f"  skipping {p.stem}: extraction was not parseable JSON")
                continue
            records.append(rec)
        except json.JSONDecodeError:
            print(f"  skipping {p.stem}: invalid JSON")

    print(f"Loaded {len(records)} parseable blind extractions")

    if len(records) < args.min:
        print(f"Only {len(records)} records — need at least {args.min}.", file=sys.stderr)
        print("Run more blind extractions first: python extract.py --blind", file=sys.stderr)
        sys.exit(1)

    schema = discover_schema(records)
    print_schema(schema)

    if not args.show:
        path = save_schema(schema)
        print(f"\n✓ Schema saved to: {path}")
        print("  Run guided extraction next:")
        print("  python extract.py --guided --skip-existing")
    else:
        print("\n(--show: schema not written)")


if __name__ == "__main__":
    main()
