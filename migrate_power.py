#!/usr/bin/env python3
"""
One-time migration: add power_holder / power_low situation_activator rows
for mechanisms that currently use power_differential.

Run once, then discard:
    python migrate_power.py
    python migrate_power.py --dry-run   # preview only
"""

import sqlite3
import sys

DB = "db/mechanisms.sqlite"

# Mechanism → which directional features to add, and their effect
# Based on mechanism semantics (see plan for rationale)
POWER_HOLDER = {
    "dominance_hierarchy": "required",
    "power_effects": "required",
    "status_threat_response": "required",
    "institutional_role_adoption": "required",
    "prestige_dominance": "activates",
    "testosterone_status": "amplifies",
    "dark_triad": "amplifies",
    "social_dominance_orientation": "amplifies",
    "dehumanization": "amplifies",
    "moral_exclusion": "amplifies",
    "scapegoating": "amplifies",
    "collective_narcissism": "amplifies",
}

POWER_LOW = {
    "obedience_authority": "required",
    "sycophancy": "required",
    "impression_management": "amplifies",
    "conformity_social_influence": "amplifies",
    "shame_response": "amplifies",
    "reactance": "activates",
    "envy_jealousy": "activates",
    "intrinsic_motivation_sdt": "dampens",
}

BOTH = {
    "coalition_formation": "activates",
    "trust_formation": "amplifies",
    "attachment_styles": "amplifies",
    "costly_signaling": "activates",
    "significance_quest": "activates",
    "pride": "amplifies",
    "social_comparison": "amplifies",
    "social_identity_theory": "amplifies",
    "gossip_reputation": "amplifies",
    "proxemics_personal_space": "amplifies",
    "contact_hypothesis": "dampens",
}


def main():
    dry_run = "--dry-run" in sys.argv

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    # Verify all referenced mechanisms exist
    all_mechs = {r["id"] for r in conn.execute("SELECT id FROM mechanisms").fetchall()}
    all_referenced = set(POWER_HOLDER) | set(POWER_LOW) | set(BOTH)
    missing = all_referenced - all_mechs
    if missing:
        print(f"WARNING: mechanisms not in DB (skipping): {missing}")

    # Check existing situation_activators with power_holder/power_low
    existing = conn.execute(
        "SELECT mechanism_id, feature FROM situation_activators "
        "WHERE feature IN ('power_holder', 'power_low')"
    ).fetchall()
    existing_set = {(r["mechanism_id"], r["feature"]) for r in existing}
    if existing_set:
        print(f"Already have {len(existing_set)} power_holder/power_low rows, skipping those")

    inserts = []

    def add(mid, feature, effect):
        if mid not in all_mechs:
            return
        if (mid, feature) in existing_set:
            return
        inserts.append((mid, feature, effect))

    for mid, effect in POWER_HOLDER.items():
        add(mid, "power_holder", effect)

    for mid, effect in POWER_LOW.items():
        add(mid, "power_low", effect)

    for mid, effect in BOTH.items():
        add(mid, "power_holder", effect)
        add(mid, "power_low", effect)

    print(f"{'Would insert' if dry_run else 'Inserting'} {len(inserts)} rows:")
    for mid, feature, effect in sorted(inserts):
        print(f"  {mid:40s}  {feature:15s}  {effect}")

    if not dry_run:
        conn.executemany(
            "INSERT INTO situation_activators (mechanism_id, feature, effect) VALUES (?,?,?)",
            inserts,
        )
        conn.commit()
        print(f"\nDone. {len(inserts)} rows inserted.")

    conn.close()


if __name__ == "__main__":
    main()
