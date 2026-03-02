"""
DB integrity tests — verify that the real mechanisms.sqlite is consistent.

These tests guard against accidental data corruption or schema drift.
Requires the DB to be present (skip gracefully if not).
"""

import json
import sqlite3
from pathlib import Path

import pytest

DB_PATH = Path(__file__).parent.parent / "db" / "mechanisms.sqlite"

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="mechanisms.sqlite not present — run db_load.py first",
)


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


# ─── Schema ───────────────────────────────────────────────────────────────────


def test_tables_exist(conn):
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {
        "mechanisms",
        "mechanism_properties",
        "interactions",
        "person_moderators",
        "situation_activators",
    } <= tables


def test_mechanisms_has_expected_columns(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(mechanisms)").fetchall()}
    expected = {
        "id",
        "name",
        "domain",
        "description",
        "summary",
        "behavioral_outputs",
        "plain_language_outputs",
        "narrative_outputs",
        "accuracy_score",
        "effect_size",
    }
    assert expected <= cols


# ─── Count invariants ─────────────────────────────────────────────────────────


def test_mechanism_count(conn):
    n = conn.execute("SELECT COUNT(*) FROM mechanisms").fetchone()[0]
    assert n >= 122, f"Expected ≥122 mechanisms, got {n}"


def test_all_mechanisms_have_plo(conn):
    missing = conn.execute(
        "SELECT id FROM mechanisms WHERE plain_language_outputs IS NULL OR plain_language_outputs=''"
    ).fetchall()
    assert missing == [], f"Mechanisms without PLO: {[r['id'] for r in missing]}"


def test_all_mechanisms_have_name_and_domain(conn):
    bad = conn.execute("SELECT id FROM mechanisms WHERE name IS NULL OR domain IS NULL").fetchall()
    assert bad == [], f"Mechanisms missing name/domain: {[r['id'] for r in bad]}"


def test_person_moderator_count(conn):
    n = conn.execute("SELECT COUNT(*) FROM person_moderators").fetchone()[0]
    assert n >= 700, f"Expected ≥700 person_moderators, got {n}"


def test_situation_activator_count(conn):
    n = conn.execute("SELECT COUNT(*) FROM situation_activators").fetchone()[0]
    assert n >= 700, f"Expected ≥700 situation_activators, got {n}"


# ─── Data quality ─────────────────────────────────────────────────────────────


def test_plo_are_valid_json_lists(conn):
    rows = conn.execute("SELECT id, plain_language_outputs FROM mechanisms").fetchall()
    errors = []
    for r in rows:
        raw = r["plain_language_outputs"]
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                errors.append(f"{r['id']}: PLO is not a JSON list")
            elif len(parsed) < 3:
                errors.append(f"{r['id']}: PLO has only {len(parsed)} items (expected ≥3)")
        except json.JSONDecodeError as e:
            errors.append(f"{r['id']}: PLO JSON parse error: {e}")
    assert errors == [], "\n".join(errors)


def test_person_moderator_directions(conn):
    """All directions are +, -, or mixed."""
    bad = conn.execute(
        "SELECT DISTINCT direction FROM person_moderators "
        "WHERE direction NOT IN ('+', '-', 'mixed')"
    ).fetchall()
    assert bad == [], f"Invalid directions: {[r[0] for r in bad]}"


def test_person_moderator_strengths(conn):
    """All strengths are strong/moderate/weak/NULL."""
    bad = conn.execute(
        "SELECT DISTINCT strength FROM person_moderators "
        "WHERE strength NOT IN ('strong', 'moderate', 'weak') AND strength IS NOT NULL"
    ).fetchall()
    assert bad == [], f"Invalid strengths: {[r[0] for r in bad]}"


def test_situation_activator_effects(conn):
    """All effects are required/activates/amplifies/dampens."""
    valid = {"required", "activates", "amplifies", "dampens"}
    bad = conn.execute("SELECT DISTINCT effect FROM situation_activators").fetchall()
    invalid = [r[0] for r in bad if r[0] not in valid]
    assert invalid == [], f"Invalid effects: {invalid}"


def test_accuracy_scores_in_range(conn):
    bad = conn.execute(
        "SELECT id, accuracy_score FROM mechanisms "
        "WHERE accuracy_score IS NOT NULL AND (accuracy_score < 0 OR accuracy_score > 1)"
    ).fetchall()
    assert bad == [], (
        f"Accuracy scores out of [0,1]: {[(r['id'], r['accuracy_score']) for r in bad]}"
    )


def test_no_duplicate_mechanism_ids(conn):
    dups = conn.execute(
        "SELECT id, COUNT(*) as n FROM mechanisms GROUP BY id HAVING n > 1"
    ).fetchall()
    assert dups == [], f"Duplicate mechanism IDs: {[r['id'] for r in dups]}"


def test_person_moderator_fk(conn):
    """All person_moderators reference existing mechanisms."""
    orphans = conn.execute(
        "SELECT DISTINCT pm.mechanism_id FROM person_moderators pm "
        "LEFT JOIN mechanisms m ON pm.mechanism_id=m.id WHERE m.id IS NULL"
    ).fetchall()
    assert orphans == [], f"Orphaned person_moderators: {[r[0] for r in orphans]}"


def test_situation_activator_fk(conn):
    """All situation_activators reference existing mechanisms."""
    orphans = conn.execute(
        "SELECT DISTINCT sa.mechanism_id FROM situation_activators sa "
        "LEFT JOIN mechanisms m ON sa.mechanism_id=m.id WHERE m.id IS NULL"
    ).fetchall()
    assert orphans == [], f"Orphaned situation_activators: {[r[0] for r in orphans]}"


# ─── Domain coverage ──────────────────────────────────────────────────────────


def test_all_seven_domains_present(conn):
    domains = {r[0] for r in conn.execute("SELECT DISTINCT domain FROM mechanisms").fetchall()}
    expected_domains = {
        "threat_affective_priming",
        "status_dominance",
        "posthoc_rationalization",
        "ingroup_outgroup",
        "social_influence_compliance",
        "individual_variation",
        "loss_aversion_reference",
    }
    assert expected_domains <= domains, f"Missing domains: {expected_domains - domains}"


def test_posthoc_rationalization_has_mechanisms(conn):
    n = conn.execute(
        "SELECT COUNT(*) FROM mechanisms WHERE domain='posthoc_rationalization'"
    ).fetchone()[0]
    assert n >= 10, f"Expected ≥10 posthoc_rationalization mechanisms, got {n}"
