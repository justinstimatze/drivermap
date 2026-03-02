"""
Tests for verbalize_behavior() template selection logic in query.py.

Uses an in-memory DB with a minimal posthoc_rationalization domain.
"""

import io
import json
import sqlite3
import sys

import pytest

from query import verbalize_behavior

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE mechanisms (
            id TEXT PRIMARY KEY,
            name TEXT,
            domain TEXT,
            description TEXT,
            summary TEXT,
            plain_language_outputs TEXT
        );
        CREATE TABLE mechanism_properties (
            mechanism_id TEXT,
            key TEXT,
            value TEXT
        );
        CREATE TABLE person_moderators (
            mechanism_id TEXT,
            dimension TEXT,
            direction TEXT,
            strength TEXT,
            note TEXT
        );
        CREATE TABLE situation_activators (
            mechanism_id TEXT,
            feature TEXT,
            effect TEXT,
            note TEXT
        );
    """)
    return conn


def _add_mech(conn, mid, domain, name=None, plo=None, desc=None):
    plo_json = json.dumps(plo) if plo else None
    conn.execute(
        "INSERT INTO mechanisms (id, name, domain, description, plain_language_outputs) "
        "VALUES (?,?,?,?,?)",
        (mid, name or mid, domain, desc, plo_json),
    )


def _add_pm(conn, mid, dimension, direction, strength="moderate"):
    conn.execute(
        "INSERT INTO person_moderators (mechanism_id, dimension, direction, strength) VALUES (?,?,?,?)",
        (mid, dimension, direction, strength),
    )


def _add_sa(conn, mid, feature, effect):
    conn.execute(
        "INSERT INTO situation_activators (mechanism_id, feature, effect) VALUES (?,?,?)",
        (mid, feature, effect),
    )


def _capture(func, *args, **kwargs) -> str:
    """Capture stdout from a function call."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue()


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_missing_hidden_mechanism_prints_error(capsys):
    conn = _make_db()
    verbalize_behavior(conn, "nonexistent_id", "did something")
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()


def test_selects_highest_scoring_template():
    """Template with matching profile dim wins over zero-score ones."""
    conn = _make_db()
    _add_mech(
        conn,
        "hidden_mech",
        "status_dominance",
        plo=["acts tough", "shows off"],
        desc="A status mechanism",
    )
    # Two rationalization templates
    _add_mech(
        conn,
        "self_serving_bias",
        "posthoc_rationalization",
        plo=["takes credit", "blames others"],
        desc="Self-serving bias",
    )
    _add_mech(
        conn,
        "high_score_template",
        "posthoc_rationalization",
        plo=["justified it", "had good reasons"],
        desc="High score template",
    )
    # Give high_score_template a matching person dim
    _add_pm(conn, "high_score_template", "big_five_N", "+")

    output = _capture(
        verbalize_behavior, conn, "hidden_mech", "grabbed credit", profile={"big_five_N": "+"}
    )
    assert "high_score_template" in output or "High score" in output


def test_fallback_to_self_serving_bias_when_no_profile():
    """With no profile/situation, self_serving_bias is selected as fallback."""
    conn = _make_db()
    _add_mech(
        conn,
        "hidden_mech",
        "status_dominance",
        plo=["dominates", "controls"],
        desc="Dominance mechanism",
    )
    _add_mech(
        conn,
        "motivated_reasoning",
        "posthoc_rationalization",
        plo=["rationalized it"],
        desc="Motivated reasoning",
    )
    _add_sa(conn, "motivated_reasoning", "prior_commitment", "required")
    _add_mech(
        conn,
        "self_serving_bias",
        "posthoc_rationalization",
        plo=["took credit", "blames context"],
        desc="Self-serving bias",
    )

    output = _capture(verbalize_behavior, conn, "hidden_mech", "took credit")
    # motivated_reasoning should be excluded (required activator missing)
    # self_serving_bias should be selected as fallback
    assert "self_serving_bias" in output or "Self-serving" in output


def test_motivated_reasoning_excluded_without_required_activator():
    """motivated_reasoning requires prior_commitment — excluded when situation is empty."""
    conn = _make_db()
    _add_mech(
        conn, "hidden_mech", "status_dominance", plo=["acts dominant"], desc="Status mechanism"
    )
    _add_mech(
        conn,
        "motivated_reasoning",
        "posthoc_rationalization",
        plo=["was already committed"],
        desc="Motivated reasoning",
    )
    _add_sa(conn, "motivated_reasoning", "prior_commitment", "required")
    _add_mech(
        conn,
        "self_serving_bias",
        "posthoc_rationalization",
        plo=["externalized blame"],
        desc="Self-serving bias",
    )

    output = _capture(verbalize_behavior, conn, "hidden_mech", "some action")
    # self_serving_bias wins because motivated_reasoning is excluded
    assert (
        "motivated_reasoning" not in output.lower().split("rationalization")[0]
        or "self_serving_bias" in output
    )


def test_motivated_reasoning_included_with_required_activator():
    """motivated_reasoning IS selected when prior_commitment is in situation and profile matches."""
    conn = _make_db()
    _add_mech(
        conn, "hidden_mech", "status_dominance", plo=["acts dominant"], desc="Status mechanism"
    )
    _add_mech(
        conn,
        "motivated_reasoning",
        "posthoc_rationalization",
        plo=["was already committed"],
        desc="Motivated reasoning",
    )
    _add_sa(conn, "motivated_reasoning", "prior_commitment", "required")
    _add_pm(conn, "motivated_reasoning", "need_for_closure", "+")
    _add_mech(
        conn,
        "self_serving_bias",
        "posthoc_rationalization",
        plo=["externalized blame"],
        desc="Self-serving bias",
    )

    output = _capture(
        verbalize_behavior,
        conn,
        "hidden_mech",
        "some action",
        profile={"need_for_closure": "+"},
        situation=["prior_commitment"],
    )
    assert "motivated_reasoning" in output or "Motivated" in output


def test_output_contains_prompt(capsys):
    conn = _make_db()
    _add_mech(
        conn,
        "hidden_mech",
        "status_dominance",
        plo=["dominates", "controls"],
        desc="Dominance mechanism",
    )
    _add_mech(
        conn,
        "self_serving_bias",
        "posthoc_rationalization",
        plo=["took credit"],
        desc="Self-serving bias",
    )

    verbalize_behavior(conn, "hidden_mech", "pushed colleague aside")
    captured = capsys.readouterr()
    # Should print the verbalization prompt
    assert "ACTUAL HIDDEN DRIVER" in captured.out
    assert "RATIONALIZATION TEMPLATE" in captured.out
    assert "TASK" in captured.out


def test_framing_first_person(capsys):
    conn = _make_db()
    _add_mech(conn, "hidden_mech", "status_dominance", plo=["shows off"], desc="Status mech")
    _add_mech(conn, "self_serving_bias", "posthoc_rationalization", plo=["justified"], desc="SSB")

    verbalize_behavior(conn, "hidden_mech", "stole credit", framing="first_person")
    captured = capsys.readouterr()
    assert "first-person" in captured.out.lower() or "I/me/my" in captured.out


def test_framing_dialogue(capsys):
    conn = _make_db()
    _add_mech(conn, "hidden_mech", "status_dominance", plo=["shows off"], desc="Status mech")
    _add_mech(conn, "self_serving_bias", "posthoc_rationalization", plo=["justified"], desc="SSB")

    verbalize_behavior(conn, "hidden_mech", "stole credit", framing="dialogue")
    captured = capsys.readouterr()
    assert "dialogue" in captured.out.lower() or "Character:" in captured.out


def test_description_fallback_from_properties(capsys):
    """If description is NULL, should fall back to mechanism_properties.definition."""
    conn = _make_db()
    # Add mechanism with no description
    conn.execute(
        "INSERT INTO mechanisms (id, name, domain, description, plain_language_outputs) "
        "VALUES (?,?,?,?,?)",
        ("hidden_mech", "Hidden Mech", "status_dominance", None, json.dumps(["acts tough"])),
    )
    # Add definition in properties
    conn.execute(
        "INSERT INTO mechanism_properties (mechanism_id, key, value) VALUES (?,?,?)",
        ("hidden_mech", "definition", "A mechanism about hidden dominance drives."),
    )
    _add_mech(
        conn,
        "self_serving_bias",
        "posthoc_rationalization",
        plo=["justified it"],
        desc="Self-serving bias",
    )

    verbalize_behavior(conn, "hidden_mech", "took credit")
    captured = capsys.readouterr()
    # The description fallback text should appear in the prompt
    assert "hidden dominance" in captured.out
