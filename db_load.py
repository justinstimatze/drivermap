#!/usr/bin/env python3
"""
db_load.py — Load extracted records into SQLite.

Three tables:
  mechanisms           — one row per mechanism, scalar fields
  mechanism_properties — key/value for variable-density optional fields
  interactions         — typed edges between mechanisms

Usage:
    python db_load.py              # load all extracted/ into db/mechanisms.sqlite
    python db_load.py --rebuild    # drop and recreate (full reload)
    python db_load.py --id loss_aversion  # load/update single record
"""

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent
EXTRACTED_DIR = ROOT / "extracted"
DB_PATH = ROOT / "db" / "mechanisms.sqlite"

# Fields that go into the mechanisms table as columns
SCALAR_FIELDS = [
    "name",
    "domain",
    "description",
    "summary",
    "mechanism_type",
    "class",
    # triggers / outputs as JSON arrays (stored as JSON strings)
    "triggers",
    "behavioral_outputs",
    "outputs",
    "plain_language_outputs",  # everyday-English effect phrases for vocabulary bridging
    "narrative_outputs",  # clinical/diagnostic register mechanism descriptions for LLM prompts
    # evidence
    "effect_size",
    "replication",
    "replication_status",
    # individual variation
    "individual_variation",
    "variation",
    # cross-cultural
    "cross_cultural",
    "cross_cultural_status",
    # phase
    "phase",
    # verification quality score
    "accuracy_score",
    # raw notes
    "notes",
]

# Fields that go into mechanism_properties as key/value
# Everything not in SCALAR_FIELDS and not in EXCLUDED_FIELDS
EXCLUDED_FIELDS = {
    "id",
    "name",
    "domain",
    "phase",
    "sources",
    "interactions",
    "person_moderators",
    "situation_activators",  # normalized into own tables
}


# ─── DB setup ─────────────────────────────────────────────────────────────────


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS mechanisms (
            id                   TEXT PRIMARY KEY,
            name                 TEXT NOT NULL,
            domain               TEXT,
            description          TEXT,
            summary              TEXT,
            mechanism_type       TEXT,
            class                TEXT,
            triggers             TEXT,   -- JSON array
            behavioral_outputs   TEXT,   -- JSON array
            outputs              TEXT,   -- JSON array (alt field name)
            plain_language_outputs TEXT, -- JSON array of everyday-English effect phrases
            narrative_outputs    TEXT,   -- JSON array of clinical-register mechanism descriptions
            effect_size          TEXT,
            replication          TEXT,
            replication_status   TEXT,
            individual_variation TEXT,   -- JSON object or string
            variation            TEXT,   -- JSON object or string (alt)
            cross_cultural       TEXT,
            cross_cultural_status TEXT,
            accuracy_score       REAL,   -- verifier score 0.0-1.0
            phase                TEXT,
            notes                TEXT,
            loaded_at            TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS mechanism_properties (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mechanism_id TEXT NOT NULL REFERENCES mechanisms(id) ON DELETE CASCADE,
            key         TEXT NOT NULL,
            value       TEXT,           -- JSON-serialized
            value_type  TEXT            -- 'string'|'number'|'boolean'|'array'|'object'|'null'
        );

        -- Normalized person-level moderators: queryable by dimension
        CREATE TABLE IF NOT EXISTS person_moderators (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            mechanism_id TEXT NOT NULL REFERENCES mechanisms(id) ON DELETE CASCADE,
            dimension    TEXT NOT NULL,  -- key from dimensions.json person_trait/state vocab
            direction    TEXT NOT NULL,  -- '+' | '-' | 'mixed'
            strength     TEXT,           -- 'weak' | 'moderate' | 'strong'
            note         TEXT
        );

        -- Normalized situation activators: queryable by feature
        CREATE TABLE IF NOT EXISTS situation_activators (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            mechanism_id TEXT NOT NULL REFERENCES mechanisms(id) ON DELETE CASCADE,
            feature      TEXT NOT NULL,  -- key from dimensions.json situation_dimensions vocab
            effect       TEXT NOT NULL,  -- 'activates' | 'amplifies' | 'dampens' | 'required'
            note         TEXT
        );

        CREATE TABLE IF NOT EXISTS interactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            mechanism_a     TEXT NOT NULL REFERENCES mechanisms(id) ON DELETE CASCADE,
            relationship    TEXT NOT NULL,  -- amplifies|suppresses|correlates_with|prerequisite_for|confounded_with
            mechanism_b     TEXT NOT NULL,  -- may not exist in DB yet
            strength        TEXT,           -- weak|moderate|strong or numeric
            direction       TEXT,           -- bidirectional|unidirectional
            notes           TEXT,
            source_record   TEXT            -- which extraction this came from
        );

        CREATE INDEX IF NOT EXISTS idx_mech_domain     ON mechanisms(domain);
        CREATE INDEX IF NOT EXISTS idx_mech_score      ON mechanisms(accuracy_score);
        CREATE INDEX IF NOT EXISTS idx_prop_mech       ON mechanism_properties(mechanism_id);
        CREATE INDEX IF NOT EXISTS idx_prop_key        ON mechanism_properties(key);
        CREATE INDEX IF NOT EXISTS idx_pm_mech         ON person_moderators(mechanism_id);
        CREATE INDEX IF NOT EXISTS idx_pm_dimension    ON person_moderators(dimension);
        CREATE INDEX IF NOT EXISTS idx_pm_dir          ON person_moderators(direction);
        CREATE INDEX IF NOT EXISTS idx_sa_mech         ON situation_activators(mechanism_id);
        CREATE INDEX IF NOT EXISTS idx_sa_feature      ON situation_activators(feature);
        CREATE INDEX IF NOT EXISTS idx_sa_effect       ON situation_activators(effect);
        CREATE INDEX IF NOT EXISTS idx_inter_a         ON interactions(mechanism_a);
        CREATE INDEX IF NOT EXISTS idx_inter_b         ON interactions(mechanism_b);
        CREATE INDEX IF NOT EXISTS idx_inter_rel       ON interactions(relationship);
    """)
    conn.commit()


def drop_tables(conn: sqlite3.Connection):
    conn.executescript("""
        DROP TABLE IF EXISTS interactions;
        DROP TABLE IF EXISTS situation_activators;
        DROP TABLE IF EXISTS person_moderators;
        DROP TABLE IF EXISTS mechanism_properties;
        DROP TABLE IF EXISTS mechanisms;
    """)
    conn.commit()


# ─── Value helpers ────────────────────────────────────────────────────────────


def json_or_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, dict | list):
        return json.dumps(v)
    return str(v)


def value_type(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int | float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "string"


# ─── Load record ─────────────────────────────────────────────────────────────


def flatten_evidence(extraction: dict) -> dict:
    """Pull evidence sub-fields up to the top level for scalar storage."""
    result = dict(extraction)
    evidence = extraction.get("evidence", {})
    if isinstance(evidence, dict):
        for k in (
            "effect_size",
            "replication",
            "replication_status",
            "cross_cultural",
            "cross_cultural_status",
            "caveats",
        ):
            if k in evidence and k not in result:
                result[k] = evidence[k]
    return result


def load_record(conn: sqlite3.Connection, rec: dict):
    """Insert or replace one extracted record into the DB."""
    mid = rec["id"]
    extraction = rec.get("extraction", {})
    if not isinstance(extraction, dict):
        extraction = {}

    extraction = flatten_evidence(extraction)

    # ── mechanisms row ──
    mech_row = {
        "id": mid,
        "name": rec.get("name", extraction.get("name", mid)),
        "domain": rec.get("domain", extraction.get("domain", "")),
        "phase": rec.get("phase", ""),
        "accuracy_score": rec.get("verification", {}).get("accuracy_score"),
    }
    for field in SCALAR_FIELDS:
        if field not in mech_row and field in extraction:
            mech_row[field] = json_or_str(extraction[field])

    cols = list(mech_row.keys())
    placeholders = ", ".join("?" for _ in cols)
    col_list = ", ".join(cols)
    update_set = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "id")

    conn.execute(
        f"INSERT INTO mechanisms ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT(id) DO UPDATE SET {update_set}",
        [mech_row.get(c) for c in cols],
    )

    # ── mechanism_properties ──
    conn.execute("DELETE FROM mechanism_properties WHERE mechanism_id=?", (mid,))
    known = set(SCALAR_FIELDS) | EXCLUDED_FIELDS
    for k, v in extraction.items():
        if k in known:
            continue
        if isinstance(v, dict) and k == "evidence":
            # Expand evidence sub-keys as properties
            for ek, ev in v.items():
                conn.execute(
                    "INSERT INTO mechanism_properties (mechanism_id, key, value, value_type) "
                    "VALUES (?, ?, ?, ?)",
                    (mid, f"evidence.{ek}", json_or_str(ev), value_type(ev)),
                )
        else:
            conn.execute(
                "INSERT INTO mechanism_properties (mechanism_id, key, value, value_type) "
                "VALUES (?, ?, ?, ?)",
                (mid, k, json_or_str(v), value_type(v)),
            )

    # ── person_moderators ──
    conn.execute("DELETE FROM person_moderators WHERE mechanism_id=?", (mid,))
    pm = extraction.get("person_moderators", [])
    if isinstance(pm, list):
        for item in pm:
            if not isinstance(item, dict):
                continue
            dim = item.get("dimension", "").strip()
            direction = item.get("direction", "").strip()
            if not dim or not direction:
                continue
            conn.execute(
                "INSERT INTO person_moderators (mechanism_id, dimension, direction, strength, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (mid, dim, direction, item.get("strength"), item.get("note")),
            )

    # ── situation_activators ──
    conn.execute("DELETE FROM situation_activators WHERE mechanism_id=?", (mid,))
    sa = extraction.get("situation_activators", [])
    if isinstance(sa, list):
        for item in sa:
            if not isinstance(item, dict):
                continue
            feature = item.get("feature", "").strip()
            effect = item.get("effect", "").strip()
            if not feature or not effect:
                continue
            conn.execute(
                "INSERT INTO situation_activators (mechanism_id, feature, effect, note) "
                "VALUES (?, ?, ?, ?)",
                (mid, feature, effect, item.get("note")),
            )

    # ── interactions ──
    conn.execute("DELETE FROM interactions WHERE mechanism_a=?", (mid,))
    interactions = extraction.get("interactions", [])
    if isinstance(interactions, list):
        for item in interactions:
            if not isinstance(item, dict):
                continue
            rel = item.get("relationship") or item.get("type") or item.get("relation", "")
            target = item.get("mechanism") or item.get("target") or item.get("mechanism_b", "")
            if not rel or not target:
                continue
            conn.execute(
                "INSERT INTO interactions "
                "(mechanism_a, relationship, mechanism_b, strength, direction, notes, source_record) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    mid,
                    rel,
                    target,
                    item.get("strength"),
                    item.get("direction"),
                    item.get("notes"),
                    mid,
                ),
            )
    elif isinstance(interactions, dict):
        # Some extractions use {"amplifies": [...], "suppresses": [...]}
        for rel, targets in interactions.items():
            if not isinstance(targets, list):
                targets = [targets]
            for target in targets:
                if isinstance(target, dict):
                    tname = target.get("mechanism") or target.get("name", "")
                    notes = target.get("notes")
                    strength = target.get("strength")
                else:
                    tname = str(target)
                    notes = None
                    strength = None
                if tname:
                    conn.execute(
                        "INSERT INTO interactions "
                        "(mechanism_a, relationship, mechanism_b, strength, notes, source_record) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (mid, rel, tname, strength, notes, mid),
                    )


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Load extracted records into SQLite")
    parser.add_argument(
        "--rebuild", action="store_true", help="Drop and recreate all tables before loading"
    )
    parser.add_argument("--id", help="Load/update single mechanism by ID")
    args = parser.parse_args()

    conn = get_conn()

    if args.rebuild:
        print("Dropping tables...")
        drop_tables(conn)

    create_tables(conn)

    if args.id:
        paths = [EXTRACTED_DIR / f"{args.id}.json"]
    else:
        paths = sorted(EXTRACTED_DIR.glob("*.json"))

    loaded = 0
    skipped = 0

    for path in paths:
        if not path.exists():
            print(f"  ✗ {path.name}: not found")
            skipped += 1
            continue
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"  ✗ {path.name}: JSON error: {e}")
            skipped += 1
            continue

        # Skip records with only raw_text
        extraction = rec.get("extraction", {})
        if isinstance(extraction, dict) and "raw_text" in extraction and len(extraction) == 1:
            print(f"  ⚠ {path.stem}: skipping (extraction was not parseable JSON)")
            skipped += 1
            continue

        load_record(conn, rec)
        loaded += 1
        print(f"  ✓ {path.stem}")

    conn.commit()
    conn.close()

    print(f"\nDone: {loaded} loaded, {skipped} skipped → {DB_PATH}")


if __name__ == "__main__":
    main()
