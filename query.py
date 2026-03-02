#!/usr/bin/env python3
"""
query.py — CLI query interface for the behavioral mechanisms knowledge base.

Usage:
    python query.py --mechanism loss_aversion
    python query.py --domain status_dominance
    python query.py --domain status_dominance --filter "replication=strong"
    python query.py --interaction amplifies --target status_threat_response
    python query.py --interaction amplifies --source loss_aversion
    python query.py --list                          # list all mechanisms
    python query.py --list --domain ingroup_outgroup
    python query.py --search "reference point"      # full-text search in name/description
    python query.py --stats                         # DB summary

    # Prediction: score mechanisms against a profile + situation
    python query.py --dim big_five_N:+ --dim bis_sensitivity:+ --feature stakes --feature social_visibility
    python query.py --scenario "high-stakes public negotiation with a rival team"
    python query.py --scenario "alone at night, unfamiliar city, low on money" --top 5
    python query.py --dim dark_triad_narcissism:+ --export json

    # Comparison: side-by-side two mechanisms
    python query.py --compare shame_response guilt
    python query.py --compare loss_aversion prospect_theory
"""

import argparse
import csv
import io
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "db" / "mechanisms.sqlite"


# ─── DB connection ────────────────────────────────────────────────────────────


def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}", file=sys.stderr)
        print("Run: python db_load.py", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


# ─── Scoring (mirrors mcp_server._score_mechanisms) ──────────────────────────

STRENGTH_WEIGHT = {"strong": 1.5, "moderate": 1.0, "weak": 0.5}
SITUATION_MULTIPLIER = 0.5


def score_mechanisms(
    conn: sqlite3.Connection, profile: dict, situation: list[str], top_n: int = 10
) -> list[dict]:
    """Score all mechanisms against a profile and situation. Returns sorted list."""
    mechs = conn.execute(
        "SELECT id, name, domain, description, summary, "
        "behavioral_outputs, outputs, plain_language_outputs, narrative_outputs, "
        "accuracy_score, effect_size, replication, replication_status "
        "FROM mechanisms"
    ).fetchall()

    situation_set = set(situation)
    results = []

    for mech in mechs:
        mid = mech["id"]
        person_score = 0.0
        situation_score = 0.0
        person_matches: list[dict] = []
        situation_matches: list[dict] = []
        excluded = False

        pms = conn.execute(
            "SELECT dimension, direction, strength, note FROM person_moderators WHERE mechanism_id=?",
            (mid,),
        ).fetchall()
        for pm in pms:
            dim = pm["dimension"]
            if dim not in profile:
                continue
            w = STRENGTH_WEIGHT.get(pm["strength"] or "moderate", 1.0)
            user_dir = profile[dim]
            mech_dir = pm["direction"]
            if mech_dir == "mixed":
                person_score += 0.25
                person_matches.append({"dimension": dim, "direction": "mixed", "weight": 0.25})
            elif user_dir == mech_dir:
                person_score += w
                person_matches.append(
                    {"dimension": dim, "direction": mech_dir, "weight": w, "effect": "amplifies"}
                )
            else:
                person_score -= w * 0.5
                person_matches.append(
                    {
                        "dimension": dim,
                        "direction": mech_dir,
                        "weight": -w * 0.5,
                        "effect": "dampens",
                    }
                )

        sas = conn.execute(
            "SELECT feature, effect, note FROM situation_activators WHERE mechanism_id=?", (mid,)
        ).fetchall()
        for sa in sas:
            feat = sa["feature"]
            effect = sa["effect"]
            if effect == "required":
                if feat in situation_set:
                    situation_score += 2.0
                    situation_matches.append(
                        {"feature": feat, "effect": "required+present", "weight": 2.0}
                    )
                else:
                    excluded = True
                    break
            elif feat in situation_set:
                if effect == "activates":
                    situation_score += 2.0
                    situation_matches.append(
                        {"feature": feat, "effect": "activates", "weight": 2.0}
                    )
                elif effect == "amplifies":
                    situation_score += 1.0
                    situation_matches.append(
                        {"feature": feat, "effect": "amplifies", "weight": 1.0}
                    )
                elif effect == "dampens":
                    situation_score -= 1.0
                    situation_matches.append({"feature": feat, "effect": "dampens", "weight": -1.0})

        if excluded:
            continue

        if person_score <= 0:
            continue

        total = person_score * (1 + situation_score * SITUATION_MULTIPLIER)
        if total <= 0:
            continue

        plo = mech["plain_language_outputs"]
        outputs = plo or mech["behavioral_outputs"] or mech["outputs"]
        try:
            outputs_parsed = json.loads(outputs) if outputs else None
        except (json.JSONDecodeError, TypeError):
            outputs_parsed = outputs

        narr = mech["narrative_outputs"]
        try:
            narr_parsed = json.loads(narr) if narr else None
        except (json.JSONDecodeError, TypeError):
            narr_parsed = narr

        results.append(
            {
                "id": mid,
                "name": mech["name"],
                "domain": mech["domain"],
                "score": round(total, 2),
                "person_score": round(person_score, 2),
                "situation_score": round(situation_score, 2),
                "person_matches": person_matches,
                "situation_matches": situation_matches,
                "outputs": outputs_parsed,
                "narrative_outputs": narr_parsed,
                "description": mech["description"] or mech["summary"],
                "evidence": {
                    "effect_size": mech["effect_size"],
                    "replication": mech["replication"] or mech["replication_status"],
                    "accuracy_score": mech["accuracy_score"],
                },
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


# ─── Feature extraction from natural language ─────────────────────────────────

_FEATURE_KEYWORDS: dict[str, list[str]] = {
    "stakes": [
        "high stakes",
        "important",
        "risky",
        "consequential",
        "critical",
        "matter",
        "significant",
        "big decision",
    ],
    "social_visibility": [
        "public",
        "watched",
        "audience",
        "observed",
        "visible",
        "everyone",
        "crowd",
        "on display",
        "in front of",
    ],
    "time_pressure": [
        "urgent",
        "rushed",
        "deadline",
        "hurry",
        "quickly",
        "time pressure",
        "fast",
        "immediate",
        "no time",
    ],
    "ambiguity": [
        "uncertain",
        "unclear",
        "ambiguous",
        "confusing",
        "vague",
        "unpredictable",
        "don't know",
        "unknown",
        "mixed signals",
        "not sure",
        "unsure",
    ],
    "out_group_salience": [
        "them",
        "outgroup",
        "enemy",
        "competitor",
        "rival",
        "other group",
        "us vs them",
        "opposition",
        "outsider",
    ],
    "power_differential": [
        "boss",
        "authority",
        "powerful",
        "hierarchy",
        "subordinate",
        "unequal",
        "superior",
        "inferior",
    ],
    "power_holder": [
        "in charge",
        "has authority",
        "holds power",
        "the boss",
        "leader",
        "in command",
        "in control",
    ],
    "power_low": [
        "outranked",
        "under authority",
        "low status",
        "subordinate",
        "powerless",
        "answering to",
        "junior",
    ],
    "resource_availability": [
        "scarce",
        "limited",
        "not enough",
        "shortage",
        "lack",
        "plenty",
        "abundant",
        "running out",
        "constrained",
    ],
    "novelty": [
        "new",
        "unfamiliar",
        "novel",
        "first time",
        "never before",
        "strange",
        "unexpected",
        "unusual",
        "foreign",
    ],
    "relationship_type": [
        "friend",
        "partner",
        "colleague",
        "stranger",
        "family",
        "relationship",
        "close",
        "intimate",
        "acquaintance",
    ],
    "anonymity": [
        "anonymous",
        "private",
        "no one watching",
        "secret",
        "hidden",
        "unknown",
        "incognito",
        "unidentified",
    ],
    "conflict_present": [
        "conflict",
        "argument",
        "disagreement",
        "fight",
        "dispute",
        "confrontation",
        "tension",
        "clash",
        "angry",
        "upset",
        "rude",
        "mad",
        "hostile",
        "insult",
        "yell",
        "bully",
        "harsh",
        "offend",
        "heated",
    ],
    "group_context": [
        "group",
        "team",
        "crowd",
        "social",
        "community",
        "together",
        "collective",
        "meeting",
        "committee",
    ],
    "outcome_reversibility": [
        "irreversible",
        "permanent",
        "can't undo",
        "final",
        "irrevocable",
        "no going back",
        "committed",
        "locked in",
    ],
    "physical_threat": [
        "danger",
        "threat",
        "physical",
        "pain",
        "harm",
        "violence",
        "attack",
        "unsafe",
        "injury",
        "fear",
    ],
    "social_norms_clarity": [
        "rules",
        "norms",
        "expectations",
        "standards",
        "appropriate",
        "proper",
        "protocol",
        "etiquette",
    ],
    "surveillance": [
        "monitored",
        "surveillance",
        "being watched",
        "tracked",
        "evaluated",
        "assessed",
        "recorded",
    ],
    "prior_commitment": [
        "committed",
        "promised",
        "already decided",
        "invested",
        "obligation",
        "pledge",
        "agreed to",
        "signed up",
    ],
}


def _text_to_features(text: str) -> list[str]:
    """Extract situation features from natural language text via keyword matching."""
    text_lower = text.lower()
    return [feat for feat, kws in _FEATURE_KEYWORDS.items() if any(kw in text_lower for kw in kws)]


# ─── Print prediction results ─────────────────────────────────────────────────


def print_predictions(results: list[dict], profile: dict, situation: list[str], export: str = None):
    if export == "json":
        print(json.dumps(results, indent=2))
        return

    if export == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(
            ["rank", "id", "name", "domain", "score", "person_score", "situation_score", "outputs"]
        )
        for i, r in enumerate(results, 1):
            outs = r["outputs"]
            if isinstance(outs, list):
                outs = "; ".join(str(o) for o in outs)
            w.writerow(
                [
                    i,
                    r["id"],
                    r["name"],
                    r["domain"],
                    r["score"],
                    r["person_score"],
                    r["situation_score"],
                    outs or "",
                ]
            )
        print(buf.getvalue())
        return

    if profile:
        print(f"\nProfile: {', '.join(f'{k}:{v}' for k, v in profile.items())}")
    if situation:
        print(f"Situation: {', '.join(situation)}")
    print(f"\nTop {len(results)} mechanisms:\n")

    for i, r in enumerate(results, 1):
        outs = r["outputs"]
        if isinstance(outs, list):
            outs_str = ", ".join(str(o) for o in outs[:4])
            if len(outs) > 4:
                outs_str += f"  +{len(outs) - 4}"
        else:
            outs_str = _truncate(str(outs or ""), 80)

        evid = r["evidence"]
        evid_parts = []
        if evid.get("effect_size"):
            evid_parts.append(f"effect={evid['effect_size']}")
        if evid.get("replication"):
            evid_parts.append(f"rep={evid['replication']}")
        evid_str = f"  [{', '.join(evid_parts)}]" if evid_parts else ""

        print(
            f"  {i:2}. {r['id']:<38}  score={r['score']:5.2f}"
            f"  (person={r['person_score']:+.1f}, sit={r['situation_score']:+.1f}){evid_str}"
        )
        print(f"      {r['name']}  [{r['domain']}]")
        if outs_str:
            print(f"      → {outs_str}")

        amp = [m["dimension"] for m in r["person_matches"] if m.get("effect") == "amplifies"]
        dmp = [m["dimension"] for m in r["person_matches"] if m.get("effect") == "dampens"]
        if amp:
            print(f"      ✓ person: {', '.join(amp)}")
        if dmp:
            print(f"      ✗ person: {', '.join(dmp)}")
        sit_pos = [
            m["feature"] for m in r["situation_matches"] if "dampens" not in m.get("effect", "")
        ]
        if sit_pos:
            print(f"      ✓ sit: {', '.join(sit_pos)}")
        print()


# ─── Compare two mechanisms ────────────────────────────────────────────────────


def compare_mechanisms(conn: sqlite3.Connection, id_a: str, id_b: str):
    def _get(mid: str):
        row = conn.execute("SELECT * FROM mechanisms WHERE id=?", (mid,)).fetchone()
        if row is None:
            rows = conn.execute(
                "SELECT * FROM mechanisms WHERE id LIKE ?", (f"%{mid}%",)
            ).fetchall()
            if len(rows) == 1:
                row = rows[0]
            elif len(rows) > 1:
                print(f"Multiple matches for '{mid}': {', '.join(r['id'] for r in rows)}")
                return None
            else:
                print(f"Not found: {mid}")
                return None
        pms = conn.execute(
            "SELECT dimension, direction, strength, note FROM person_moderators WHERE mechanism_id=?",
            (row["id"],),
        ).fetchall()
        sas = conn.execute(
            "SELECT feature, effect, note FROM situation_activators WHERE mechanism_id=?",
            (row["id"],),
        ).fetchall()
        return row, list(pms), list(sas)

    ra = _get(id_a)
    rb = _get(id_b)
    if ra is None or rb is None:
        return

    row_a, pms_a, sas_a = ra
    row_b, pms_b, sas_b = rb

    pm_a = {r["dimension"]: r for r in pms_a}
    pm_b = {r["dimension"]: r for r in pms_b}
    sa_a = {r["feature"]: r for r in sas_a}
    sa_b = {r["feature"]: r for r in sas_b}

    def evid_str(r):
        parts = []
        if r["effect_size"]:
            parts.append(f"effect={r['effect_size']}")
        rep = r["replication"] or r["replication_status"]
        if rep:
            parts.append(f"rep={rep}")
        if r["accuracy_score"]:
            parts.append(f"score={r['accuracy_score']:.2f}")
        return ", ".join(parts) or "—"

    def parse_outputs(r):
        plo = r["plain_language_outputs"]
        out = plo or r["behavioral_outputs"] or r["outputs"]
        if not out:
            return []
        try:
            o = json.loads(out)
            if isinstance(o, list):
                return [str(x) for x in o]
            if isinstance(o, dict):
                return [f"{k}: {v}" for k, v in list(o.items())[:5]]
        except (json.JSONDecodeError, TypeError):
            pass
        return [str(out)[:80]]

    print(f"\n{'═' * 70}")
    print("  COMPARISON")
    print(f"  A: {row_a['name']}  [{row_a['id']}]")
    print(f"  B: {row_b['name']}  [{row_b['id']}]")
    print(f"{'─' * 70}")

    print(f"\n  Domain     A: {row_a['domain']}")
    print(f"             B: {row_b['domain']}")
    print(f"\n  Evidence   A: {evid_str(row_a)}")
    print(f"             B: {evid_str(row_b)}")

    # Shared person moderators
    shared_dims = sorted(set(pm_a) & set(pm_b))
    if shared_dims:
        print(f"\n  Shared person moderators ({len(shared_dims)}):")
        for dim in shared_dims:
            da = pm_a[dim]["direction"]
            db = pm_b[dim]["direction"]
            sa_ = (pm_a[dim]["strength"] or "mod")[:3]
            sb_ = (pm_b[dim]["strength"] or "mod")[:3]
            conflict = "  ← CONFLICT" if (da != db and "mixed" not in (da, db)) else ""
            same = "  ← SAME" if da == db else ""
            print(f"    {dim:<35}  A:{da}({sa_})  B:{db}({sb_}){conflict}{same}")

    only_a_dims = sorted(set(pm_a) - set(pm_b))
    only_b_dims = sorted(set(pm_b) - set(pm_a))
    if only_a_dims:
        print(f"\n  Only A's moderators: {', '.join(only_a_dims)}")
    if only_b_dims:
        print(f"\n  Only B's moderators: {', '.join(only_b_dims)}")

    # Shared situation activators
    shared_feats = sorted(set(sa_a) & set(sa_b))
    if shared_feats:
        print(f"\n  Shared situation features ({len(shared_feats)}):")
        for feat in shared_feats:
            ea = sa_a[feat]["effect"]
            eb = sa_b[feat]["effect"]
            conflict = "  ← CONFLICT" if ea != eb else "  ← SAME"
            print(f"    {feat:<30}  A:{ea}  B:{eb}{conflict}")

    only_a_feats = sorted(set(sa_a) - set(sa_b))
    only_b_feats = sorted(set(sa_b) - set(sa_a))
    if only_a_feats:
        print(f"\n  Only A's situation: {', '.join(only_a_feats)}")
    if only_b_feats:
        print(f"\n  Only B's situation: {', '.join(only_b_feats)}")

    # Outputs
    oa = parse_outputs(row_a)
    ob = parse_outputs(row_b)
    print(f"\n  Outputs A: {', '.join(oa[:5])}")
    print(f"  Outputs B: {', '.join(ob[:5])}")
    print()


# ─── Formatters ───────────────────────────────────────────────────────────────


def fmt_json_field(val: str | None, label: str, indent: int = 2) -> str:
    if val is None:
        return ""
    try:
        obj = json.loads(val)
        if isinstance(obj, list):
            lines = [f"{'  ' * indent}{label}:"]
            for item in obj:
                lines.append(f"{'  ' * (indent + 1)}- {item}")
            return "\n".join(lines)
        elif isinstance(obj, dict):
            lines = [f"{'  ' * indent}{label}:"]
            for k, v in obj.items():
                lines.append(f"{'  ' * (indent + 1)}{k}: {v}")
            return "\n".join(lines)
    except (json.JSONDecodeError, TypeError):
        pass
    return f"{'  ' * indent}{label}: {val}"


def format_mechanism(
    row: sqlite3.Row,
    props: list[sqlite3.Row],
    interactions: list[sqlite3.Row],
    pms: list[sqlite3.Row] = None,
    sas: list[sqlite3.Row] = None,
) -> str:
    lines = []
    lines.append(f"\n{'═' * 60}")
    lines.append(f"  {row['name']}  [{row['id']}]")
    lines.append(f"  Domain: {row['domain'] or '—'}")
    if row["accuracy_score"]:
        lines.append(f"  Accuracy: {row['accuracy_score']:.2f}")
    lines.append(f"{'─' * 60}")

    if row["description"] or row["summary"]:
        desc = row["description"] or row["summary"]
        lines.append(f"\n  {desc}")

    # Evidence
    evid_parts = []
    if row["effect_size"]:
        evid_parts.append(f"effect_size={row['effect_size']}")
    if row["replication"] or row["replication_status"]:
        evid_parts.append(f"replication={row['replication'] or row['replication_status']}")
    if row["cross_cultural"] or row["cross_cultural_status"]:
        evid_parts.append(f"cross_cultural={row['cross_cultural'] or row['cross_cultural_status']}")
    if evid_parts:
        lines.append(f"\n  Evidence: {', '.join(evid_parts)}")

    # Triggers
    t = fmt_json_field(row["triggers"], "triggers")
    if t:
        lines.append(f"\n{t}")

    # Outputs (prefer plain_language_outputs if present)
    plo = row["plain_language_outputs"]
    out = plo or row["behavioral_outputs"] or row["outputs"]
    label = "plain_language_outputs" if plo else "behavioral_outputs"
    o = fmt_json_field(out, label)
    if o:
        lines.append(f"\n{o}")

    # Individual variation
    iv = row["individual_variation"] or row["variation"]
    if iv:
        lines.append(f"\n  individual_variation: {_truncate(iv, 120)}")

    # Person moderators
    if pms:
        lines.append("\n  Person moderators:")
        for pm in pms:
            strength = f" ({pm['strength']})" if pm["strength"] else ""
            note = f" — {_truncate(pm['note'] or '', 80)}" if pm["note"] else ""
            lines.append(f"    {pm['direction']:4} {pm['dimension']}{strength}{note}")

    # Situation activators
    if sas:
        lines.append("\n  Situation activators:")
        for sa in sas:
            note = f" — {_truncate(sa['note'] or '', 80)}" if sa["note"] else ""
            lines.append(f"    {sa['effect']:10} {sa['feature']}{note}")

    # Properties (optional fields)
    if props:
        lines.append("\n  Properties:")
        for p in props:
            val = _truncate(p["value"] or "", 100)
            lines.append(f"    {p['key']}: {val}")

    # Interactions
    if interactions:
        lines.append("\n  Interactions:")
        for ix in interactions:
            strength = f" ({ix['strength']})" if ix["strength"] else ""
            notes = f" — {ix['notes']}" if ix["notes"] else ""
            lines.append(f"    {ix['relationship']:20} → {ix['mechanism_b']}{strength}{notes}")

    if row["notes"]:
        lines.append(f"\n  Notes: {_truncate(row['notes'], 200)}")

    return "\n".join(lines)


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "…"


# ─── Queries ──────────────────────────────────────────────────────────────────


def query_mechanism(conn: sqlite3.Connection, mid: str):
    row = conn.execute("SELECT * FROM mechanisms WHERE id=?", (mid,)).fetchone()
    if row is None:
        rows = conn.execute("SELECT * FROM mechanisms WHERE id LIKE ?", (f"%{mid}%",)).fetchall()
        if not rows:
            print(f"Mechanism not found: {mid}")
            return
        if len(rows) > 1:
            print(f"Multiple matches for '{mid}':")
            for r in rows:
                print(f"  {r['id']}  {r['name']}")
            return
        row = rows[0]

    props = conn.execute(
        "SELECT * FROM mechanism_properties WHERE mechanism_id=? ORDER BY key",
        (row["id"],),
    ).fetchall()
    interactions = conn.execute(
        "SELECT * FROM interactions WHERE mechanism_a=? ORDER BY relationship, mechanism_b",
        (row["id"],),
    ).fetchall()
    pms = conn.execute(
        "SELECT dimension, direction, strength, note FROM person_moderators "
        "WHERE mechanism_id=? ORDER BY strength DESC, dimension",
        (row["id"],),
    ).fetchall()
    sas = conn.execute(
        "SELECT feature, effect, note FROM situation_activators "
        "WHERE mechanism_id=? ORDER BY effect, feature",
        (row["id"],),
    ).fetchall()

    print(format_mechanism(row, props, interactions, pms=pms, sas=sas))


def query_domain(conn: sqlite3.Connection, domain: str, filters: list[str] = None):
    rows = conn.execute(
        "SELECT * FROM mechanisms WHERE domain LIKE ? ORDER BY name",
        (f"%{domain}%",),
    ).fetchall()

    if not rows:
        print(f"No mechanisms found for domain: {domain}")
        return

    if filters:
        filtered = []
        for row in rows:
            match = True
            for f in filters:
                if "=" not in f:
                    continue
                k, _, v = f.partition("=")
                k = k.strip()
                v = v.strip().lower()
                row_val = (row[k] or "").lower() if k in row.keys() else ""
                if v not in row_val:
                    match = False
                    break
            if match:
                filtered.append(row)
        rows = filtered

    print(f"\nDomain: {domain} ({len(rows)} mechanism(s))\n")
    for row in rows:
        evid = []
        if row["effect_size"]:
            evid.append(f"effect={row['effect_size']}")
        if row["replication"] or row["replication_status"]:
            evid.append(f"rep={row['replication'] or row['replication_status']}")
        evid_str = f"  [{', '.join(evid)}]" if evid else ""
        print(f"  {row['id']:<40} {row['name'][:40]}{evid_str}")


def query_interaction(
    conn: sqlite3.Connection, relationship: str, source: str = None, target: str = None
):
    where = ["relationship LIKE ?"]
    params = [f"%{relationship}%"]

    if source:
        where.append("mechanism_a LIKE ?")
        params.append(f"%{source}%")
    if target:
        where.append("mechanism_b LIKE ?")
        params.append(f"%{target}%")

    sql = f"SELECT * FROM interactions WHERE {' AND '.join(where)} ORDER BY mechanism_a"
    rows = conn.execute(sql, params).fetchall()

    if not rows:
        print(
            f"No interactions found for: relationship={relationship}"
            + (f", source={source}" if source else "")
            + (f", target={target}" if target else "")
        )
        return

    print(f"\nInteractions ({relationship}) — {len(rows)} result(s)\n")
    for row in rows:
        strength = f" ({row['strength']})" if row["strength"] else ""
        notes = f"\n      {row['notes']}" if row["notes"] else ""
        print(
            f"  {row['mechanism_a']:<35} {row['relationship']:20} → {row['mechanism_b']}{strength}{notes}"
        )


def list_mechanisms(conn: sqlite3.Connection, domain: str = None):
    if domain:
        rows = conn.execute(
            "SELECT * FROM mechanisms WHERE domain LIKE ? ORDER BY domain, name",
            (f"%{domain}%",),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM mechanisms ORDER BY domain, name").fetchall()

    current_domain = None
    for row in rows:
        if row["domain"] != current_domain:
            current_domain = row["domain"]
            print(f"\n── {current_domain or 'unknown'} ──")
        evid = []
        if row["effect_size"]:
            evid.append(row["effect_size"])
        if row["replication"] or row["replication_status"]:
            evid.append(row["replication"] or row["replication_status"])
        tag = f" [{', '.join(evid)}]" if evid else ""
        print(f"  {row['id']:<42} {row['name'][:50]}{tag}")


def search_mechanisms(conn: sqlite3.Connection, query: str):
    q = f"%{query}%"
    rows = conn.execute(
        """SELECT m.*, GROUP_CONCAT(p.key || '=' || p.value, '; ') as props
           FROM mechanisms m
           LEFT JOIN mechanism_properties p ON m.id = p.mechanism_id
           WHERE m.name LIKE ? OR m.description LIKE ? OR m.summary LIKE ?
              OR m.notes LIKE ? OR m.id LIKE ?
           GROUP BY m.id
           ORDER BY m.name""",
        (q, q, q, q, q),
    ).fetchall()

    if not rows:
        print(f"No results for: '{query}'")
        return

    print(f"\nSearch results for '{query}' — {len(rows)} match(es)\n")
    for row in rows:
        desc = _truncate(row["description"] or row["summary"] or "", 120)
        print(f"  {row['id']}")
        print(f"    {row['name']}  [{row['domain']}]")
        if desc:
            print(f"    {desc}")
        print()


def show_stats(conn: sqlite3.Connection):
    n_mech = conn.execute("SELECT COUNT(*) FROM mechanisms").fetchone()[0]
    n_props = conn.execute("SELECT COUNT(*) FROM mechanism_properties").fetchone()[0]
    n_inter = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    n_pm = conn.execute("SELECT COUNT(*) FROM person_moderators").fetchone()[0]
    n_sa = conn.execute("SELECT COUNT(*) FROM situation_activators").fetchone()[0]
    n_plo = conn.execute(
        "SELECT COUNT(*) FROM mechanisms WHERE plain_language_outputs IS NOT NULL"
    ).fetchone()[0]

    print(f"\nDatabase: {DB_PATH}\n")
    print(f"  Mechanisms:              {n_mech}")
    print(f"  plain_language_outputs:  {n_plo}/{n_mech}")
    print(f"  Properties:              {n_props}")
    print(f"  Interactions:            {n_inter}")
    print(f"  Person moderators:       {n_pm}")
    print(f"  Situation activators:    {n_sa}")

    print("\nBy domain:")
    rows = conn.execute(
        "SELECT domain, COUNT(*) as n FROM mechanisms GROUP BY domain ORDER BY domain"
    ).fetchall()
    for r in rows:
        print(f"  {r['domain'] or 'unknown':<45} {r['n']}")

    print("\nRelationship types:")
    rows = conn.execute(
        "SELECT relationship, COUNT(*) as n FROM interactions GROUP BY relationship ORDER BY n DESC"
    ).fetchall()
    for r in rows:
        print(f"  {r['relationship']:<30} {r['n']}")

    print("\nTop optional fields:")
    rows = conn.execute(
        "SELECT key, COUNT(*) as n FROM mechanism_properties GROUP BY key ORDER BY n DESC LIMIT 15"
    ).fetchall()
    for r in rows:
        print(f"  {r['key']:<30} {r['n']}")

    print("\nDimension coverage (person_moderators):")
    rows = conn.execute(
        "SELECT dimension, COUNT(DISTINCT mechanism_id) as n FROM person_moderators "
        "GROUP BY dimension ORDER BY n DESC LIMIT 15"
    ).fetchall()
    for r in rows:
        print(f"  {r['dimension']:<35} {r['n']} mechanisms")

    print("\nSituation feature coverage:")
    rows = conn.execute(
        "SELECT feature, effect, COUNT(DISTINCT mechanism_id) as n FROM situation_activators "
        "GROUP BY feature, effect ORDER BY feature, n DESC"
    ).fetchall()
    current_feat = None
    for r in rows:
        if r["feature"] != current_feat:
            current_feat = r["feature"]
            print(f"  {r['feature']}")
        print(f"    {r['effect']:<15} {r['n']} mechanisms")


# ─── Rationalization verbalization ───────────────────────────────────────────


def verbalize_behavior(
    conn: sqlite3.Connection,
    hidden_id: str,
    action: str,
    profile: dict = None,
    situation: list[str] = None,
    framing: str = "first_person",
):
    """
    Given a hidden mechanism and an action, generate surface rationalizations.
    Prints results to stdout.
    """
    # Look up hidden mechanism
    row = conn.execute(
        "SELECT id, name, domain, description, summary, plain_language_outputs "
        "FROM mechanisms WHERE id=?",
        (hidden_id,),
    ).fetchone()
    if row is None:
        print(f"Error: mechanism '{hidden_id}' not found.", file=sys.stderr)
        return

    plo_raw = row["plain_language_outputs"]
    try:
        hidden_plo = json.loads(plo_raw) if plo_raw else []
    except (json.JSONDecodeError, TypeError):
        hidden_plo = []

    # description may be in mechanism_properties if not in top-level columns
    desc = row["description"] or row["summary"] or ""
    if not desc:
        prop = conn.execute(
            "SELECT value FROM mechanism_properties WHERE mechanism_id=? AND key='definition'",
            (hidden_id,),
        ).fetchone()
        if prop:
            desc = prop["value"] or ""

    hidden = {
        "id": row["id"],
        "name": row["name"],
        "domain": row["domain"],
        "description": desc,
        "plain_language_outputs": hidden_plo,
    }

    # Score posthoc_rationalization mechanisms
    profile = profile or {}
    situation = situation or []
    situation_set = set(situation)

    rat_mechs = conn.execute(
        "SELECT id, name, description, summary, plain_language_outputs "
        "FROM mechanisms WHERE domain='posthoc_rationalization'"
    ).fetchall()

    rat_scored = []
    for mech in rat_mechs:
        mid = mech["id"]
        person_score = 0.0
        situation_score = 0.0
        excluded = False

        pms = conn.execute(
            "SELECT dimension, direction, strength FROM person_moderators WHERE mechanism_id=?",
            (mid,),
        ).fetchall()
        for pm in pms:
            dim = pm["dimension"]
            if dim not in profile:
                continue
            w = STRENGTH_WEIGHT.get(pm["strength"] or "moderate", 1.0)
            if pm["direction"] == "mixed":
                person_score += 0.25
            elif profile[dim] == pm["direction"]:
                person_score += w
            else:
                person_score -= w * 0.5

        sas = conn.execute(
            "SELECT feature, effect FROM situation_activators WHERE mechanism_id=?", (mid,)
        ).fetchall()
        for sa in sas:
            feat, effect = sa["feature"], sa["effect"]
            if effect == "required":
                if feat not in situation_set:
                    excluded = True
                    break
                situation_score += 2.0
            elif feat in situation_set:
                situation_score += (
                    2.0 if effect == "activates" else (1.0 if effect == "amplifies" else -1.0)
                )

        if excluded:
            continue

        plo_r = mech["plain_language_outputs"]
        try:
            plo_parsed = json.loads(plo_r) if plo_r else []
        except (json.JSONDecodeError, TypeError):
            plo_parsed = []

        desc_r = mech["description"] or mech["summary"] or ""
        if not desc_r:
            prop_r = conn.execute(
                "SELECT value FROM mechanism_properties WHERE mechanism_id=? AND key='definition'",
                (mid,),
            ).fetchone()
            if prop_r:
                desc_r = prop_r["value"] or ""

        rat_scored.append(
            {
                "id": mid,
                "name": mech["name"],
                "score": person_score * (1 + situation_score * SITUATION_MULTIPLIER),
                "description": desc_r,
                "plain_language_outputs": plo_parsed,
            }
        )

    rat_scored.sort(key=lambda x: x["score"], reverse=True)

    # Pick template: top scorer, or self_serving_bias as universal fallback
    # (motivated_reasoning has a required activator so it gets excluded with no situation)
    if rat_scored and rat_scored[0]["score"] > 0:
        template = rat_scored[0]
    else:
        template = next(
            (r for r in rat_scored if r["id"] == "self_serving_bias"),
            rat_scored[0] if rat_scored else None,
        )

    if template is None:
        print("No rationalization template found.", file=sys.stderr)
        return

    # Build prompt
    framing_instr = {
        "first_person": "Generate 4-5 first-person statements (I/me/my) this person would say out loud. They sound sincere and self-justifying.",
        "third_person": "Generate 4-5 statements in third person. Use 'She told herself…', 'He felt that…', etc.",
        "dialogue": 'Generate 4-5 dialogue lines. Format each as: Character: "..."',
    }.get(framing, "Generate 4-5 first-person statements.")

    hidden_vocab = ", ".join(f'"{p}"' for p in hidden["plain_language_outputs"][:6])
    rat_vocab = ", ".join(f'"{p}"' for p in template["plain_language_outputs"][:6])

    prompt = f"""You are generating dialogue for a character study in behavioral psychology.

ACTUAL HIDDEN DRIVER
Mechanism: {hidden["name"]}
What it produces: {hidden_vocab}
Core dynamic: {hidden["description"][:280]}

ACTION TAKEN: {action}

RATIONALIZATION TEMPLATE
The character processes this via: {template["name"]}
Verbal patterns: {rat_vocab}
Core pattern: {template["description"][:200]}

TASK
{framing_instr}

Rules:
- Do NOT name or reference the actual psychological mechanism
- The character is NOT self-aware about the rationalization
- Statements sound genuine, not rehearsed
- Draw on the verbal pattern vocabulary above
- Each statement 1-2 sentences max

Output: a JSON array of strings only. No commentary."""

    # Print structured analysis
    print(f"\nVerbalize: {hidden['name']}  [{hidden['domain']}]")
    print(f"Action:    {action}")
    print(f"Rationalization type: {template['name']}")
    if profile or situation:
        print(f"Template fit score: {template['score']:.1f}")
    print()
    print("Hidden driver vocabulary:")
    for p in hidden["plain_language_outputs"][:6]:
        print(f"  · {p}")
    print()
    print(f"Rationalization vocabulary ({template['name']}):")
    for p in template["plain_language_outputs"][:6]:
        print(f"  · {p}")
    print()
    print("─" * 60)
    print("Verbalization prompt (pipe to claude or paste):")
    print("─" * 60)
    print(prompt)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Query the behavioral mechanisms knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Look up a mechanism (now includes moderators + activators)
  python query.py --mechanism loss_aversion

  # Predict active mechanisms for a profile + situation
  python query.py --dim big_five_N:+ --dim bis_sensitivity:+ --feature stakes
  python query.py --scenario "high-stakes negotiation watched by colleagues"
  python query.py --dim dark_triad_narcissism:+ --scenario "rival challenges status" --top 5

  # Compare two mechanisms side by side
  python query.py --compare shame_response guilt
  python query.py --compare loss_aversion prospect_theory

  # Export prediction results
  python query.py --dim big_five_N:+ --feature stakes --export json
  python query.py --scenario "job interview" --export csv

  # Classic queries
  python query.py --domain status_dominance
  python query.py --search "reference point"
  python query.py --stats
""",
    )

    # Classic flags
    parser.add_argument("--mechanism", "-m", help="Look up a mechanism by ID")
    parser.add_argument("--domain", "-d", help="List mechanisms in a domain")
    parser.add_argument(
        "--filter",
        "-f",
        action="append",
        dest="filters",
        help="Filter: field=value (use with --domain)",
    )
    parser.add_argument("--interaction", "-i", help="Query interactions by relationship type")
    parser.add_argument("--source", help="Interaction source mechanism")
    parser.add_argument("--target", "-t", help="Interaction target mechanism")
    parser.add_argument("--list", "-l", action="store_true", help="List all mechanisms")
    parser.add_argument("--search", "-s", help="Full-text search")
    parser.add_argument("--stats", action="store_true", help="Show DB stats")

    # Prediction flags
    parser.add_argument(
        "--dim",
        action="append",
        dest="dims",
        metavar="DIM:+/-",
        help="Person dimension (repeatable). E.g. --dim big_five_N:+",
    )
    parser.add_argument(
        "--feature",
        action="append",
        dest="features",
        metavar="FEATURE",
        help="Situation feature (repeatable). E.g. --feature stakes",
    )
    parser.add_argument(
        "--scenario", metavar="TEXT", help="Natural-language situation; auto-extracts features"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="Number of results for --predict (default 10)",
    )
    parser.add_argument(
        "--export", choices=["json", "csv"], help="Export prediction results as json or csv"
    )
    parser.add_argument(
        "--viz", metavar="PATH", help="Save prediction result as a PNG visualization"
    )

    # Comparison flag
    parser.add_argument(
        "--compare", nargs=2, metavar=("A", "B"), help="Compare two mechanisms side by side"
    )

    # Verbalization (rationalization meta-layer)
    parser.add_argument(
        "--verbalize",
        metavar="MECHANISM_ID",
        help="Generate surface rationalizations for a hidden mechanism",
    )
    parser.add_argument(
        "--action", metavar="TEXT", help="What the person did (use with --verbalize)"
    )
    parser.add_argument(
        "--framing",
        choices=["first_person", "third_person", "dialogue"],
        default="first_person",
        help="Verbalization framing (default: first_person)",
    )

    args = parser.parse_args()

    # Determine if prediction mode is active
    predict_mode = bool(args.dims or args.features or args.scenario)

    if not any(
        [
            args.mechanism,
            args.domain,
            args.interaction,
            args.list,
            args.search,
            args.stats,
            predict_mode,
            args.compare,
            args.verbalize,
        ]
    ):
        parser.print_help()
        return

    conn = get_conn()

    if args.mechanism:
        query_mechanism(conn, args.mechanism)

    if args.domain and not args.list:
        query_domain(conn, args.domain, filters=args.filters or [])

    if args.interaction:
        query_interaction(conn, args.interaction, source=args.source, target=args.target)

    if args.list:
        list_mechanisms(conn, domain=args.domain)

    if args.search:
        search_mechanisms(conn, args.search)

    if args.stats:
        show_stats(conn)

    if predict_mode:
        # Build profile dict from --dim flags
        profile: dict[str, str] = {}
        for dim_spec in args.dims or []:
            if ":" in dim_spec:
                dim, _, val = dim_spec.rpartition(":")
                if val in ("+", "-"):
                    profile[dim] = val
                else:
                    print(
                        f"  Warning: ignoring malformed --dim '{dim_spec}' (expected DIM:+ or DIM:-)",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"  Warning: ignoring malformed --dim '{dim_spec}' (expected DIM:+ or DIM:-)",
                    file=sys.stderr,
                )

        # Build situation list from --feature flags + --scenario extraction
        situation = list(args.features or [])
        if args.scenario:
            extracted = _text_to_features(args.scenario)
            new_feats = [f for f in extracted if f not in situation]
            if new_feats:
                print(f'\nScenario: "{args.scenario}"')
                print(f"Extracted features: {', '.join(extracted)}")
            situation.extend(new_feats)

        results = score_mechanisms(conn, profile, situation, top_n=args.top)

        if not results:
            print("\nNo mechanisms scored above zero for this profile/situation.")
            print("Try adding more --dim or --feature flags, or broaden --scenario.")
        else:
            print_predictions(results, profile, situation, export=args.export)
            if args.viz:
                from visualize import render_prediction

                title_parts = []
                if profile:
                    title_parts.append(", ".join(f"{k}:{v}" for k, v in profile.items()))
                if situation:
                    title_parts.append("· " + ", ".join(situation))
                render_prediction(
                    results,
                    profile,
                    situation,
                    output_path=args.viz,
                    title="Prediction — " + "  ".join(title_parts),
                )

    if args.compare:
        compare_mechanisms(conn, args.compare[0], args.compare[1])

    if args.verbalize:
        if not args.action:
            print("Error: --verbalize requires --action TEXT", file=sys.stderr)
        else:
            # Build profile/situation if prediction flags were also provided
            profile: dict[str, str] = {}
            for dim_spec in args.dims or []:
                if ":" in dim_spec:
                    dim, _, val = dim_spec.rpartition(":")
                    if val in ("+", "-"):
                        profile[dim] = val
            situation = list(args.features or [])
            if args.scenario:
                situation.extend(_text_to_features(args.scenario))
            verbalize_behavior(
                conn,
                args.verbalize,
                args.action,
                profile=profile or None,
                situation=situation or None,
                framing=args.framing,
            )

    conn.close()


if __name__ == "__main__":
    main()
