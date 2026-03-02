"""
Unit tests for score_mechanisms() and _text_to_features() in query.py.

Uses an in-memory SQLite DB so tests are fully isolated from the real DB.
"""

import json
import sqlite3

import pytest

from query import SITUATION_MULTIPLIER, STRENGTH_WEIGHT, _text_to_features, score_mechanisms

# ─── Fixtures ────────────────────────────────────────────────────────────────


def _make_db() -> sqlite3.Connection:
    """Return an in-memory SQLite connection with the project schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE mechanisms (
            id TEXT PRIMARY KEY,
            name TEXT,
            domain TEXT,
            description TEXT,
            summary TEXT,
            behavioral_outputs TEXT,
            outputs TEXT,
            plain_language_outputs TEXT,
            narrative_outputs TEXT,
            accuracy_score REAL,
            effect_size TEXT,
            replication TEXT,
            replication_status TEXT
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


def _add_mech(conn, mid, name="Test Mech", domain="test", plo=None):
    plo_json = json.dumps(plo) if plo else None
    conn.execute(
        "INSERT INTO mechanisms (id, name, domain, plain_language_outputs) VALUES (?,?,?,?)",
        (mid, name, domain, plo_json),
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


# ─── STRENGTH_WEIGHT sanity ───────────────────────────────────────────────────


def test_strength_weight_values():
    assert STRENGTH_WEIGHT["strong"] > STRENGTH_WEIGHT["moderate"] > STRENGTH_WEIGHT["weak"]
    assert STRENGTH_WEIGHT["strong"] == 1.5
    assert STRENGTH_WEIGHT["moderate"] == 1.0
    assert STRENGTH_WEIGHT["weak"] == 0.5


# ─── Scoring: person moderators ──────────────────────────────────────────────


def test_person_match_amplifies():
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+")
    results = score_mechanisms(conn, {"big_five_N": "+"}, [])
    assert len(results) == 1
    assert results[0]["id"] == "m1"
    assert results[0]["person_score"] == pytest.approx(1.0)  # moderate weight


def test_person_match_dampens():
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+")
    # Profile is opposite direction → person_score negative, filtered out
    results = score_mechanisms(conn, {"big_five_N": "-"}, [])
    assert len(results) == 0


def test_person_match_strong_amplifies():
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+", strength="strong")
    results = score_mechanisms(conn, {"big_five_N": "+"}, [])
    assert results[0]["person_score"] == pytest.approx(1.5)


def test_person_match_mixed_direction():
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "mixed")
    results = score_mechanisms(conn, {"big_five_N": "+"}, [])
    assert results[0]["person_score"] == pytest.approx(0.25)


def test_irrelevant_dimension_ignored():
    """Profile dim not in mechanism's moderators → no score change."""
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+")
    # Supply a dimension the mechanism doesn't care about → still scores via person match
    results = score_mechanisms(conn, {"big_five_N": "+", "dark_triad_M": "-"}, [])
    assert results[0]["person_score"] == pytest.approx(1.0)


def test_no_profile_match_excluded():
    """Mechanism requires person dims but profile is empty → score=0 → excluded."""
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+")
    results = score_mechanisms(conn, {}, [])
    assert len(results) == 0


# ─── Scoring: situation activators ───────────────────────────────────────────


def test_situation_activates():
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+")  # ensure base score > 0
    _add_sa(conn, "m1", "stakes", "activates")
    results = score_mechanisms(conn, {"big_five_N": "+"}, ["stakes"])
    assert results[0]["situation_score"] == pytest.approx(2.0)


def test_situation_amplifies():
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+")
    _add_sa(conn, "m1", "ambiguity", "amplifies")
    results = score_mechanisms(conn, {"big_five_N": "+"}, ["ambiguity"])
    assert results[0]["situation_score"] == pytest.approx(1.0)


def test_situation_dampens():
    conn = _make_db()
    _add_mech(conn, "m1")
    # Strong person match (1.5) so total stays positive after dampening (-1.0)
    _add_pm(conn, "m1", "big_five_N", "+", strength="strong")
    _add_sa(conn, "m1", "anonymity", "dampens")
    results = score_mechanisms(conn, {"big_five_N": "+"}, ["anonymity"])
    assert len(results) == 1
    assert results[0]["situation_score"] == pytest.approx(-1.0)
    # multiplicative: 1.5 * (1 + (-1.0) * 0.5) = 1.5 * 0.5 = 0.75
    assert results[0]["score"] == pytest.approx(0.75)


def test_situation_dampens_survives():
    """Dampening reduces but no longer zeros out a single-moderator mechanism."""
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+")  # moderate = 1.0
    _add_sa(conn, "m1", "anonymity", "dampens")  # -1.0
    results = score_mechanisms(conn, {"big_five_N": "+"}, ["anonymity"])
    # multiplicative: 1.0 * (1 + (-1.0) * 0.5) = 1.0 * 0.5 = 0.5
    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(0.5)


def test_situation_required_present():
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+")
    _add_sa(conn, "m1", "prior_commitment", "required")
    results = score_mechanisms(conn, {"big_five_N": "+"}, ["prior_commitment"])
    assert results[0]["situation_score"] == pytest.approx(2.0)


def test_situation_required_absent_excludes():
    """Required feature not in situation → mechanism excluded entirely."""
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+")
    _add_sa(conn, "m1", "prior_commitment", "required")
    results = score_mechanisms(conn, {"big_five_N": "+"}, [])
    assert len(results) == 0


def test_situation_feature_not_in_situation_ignored():
    """Situation activator not triggered → adds 0."""
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+")
    _add_sa(conn, "m1", "stakes", "activates")
    results = score_mechanisms(conn, {"big_five_N": "+"}, [])
    assert results[0]["situation_score"] == pytest.approx(0.0)


# ─── Scoring: ranking and top_n ──────────────────────────────────────────────


def test_ranking_order():
    conn = _make_db()
    _add_mech(conn, "weak_one")
    _add_pm(conn, "weak_one", "big_five_N", "+", strength="weak")
    _add_mech(conn, "strong_one")
    _add_pm(conn, "strong_one", "big_five_N", "+", strength="strong")
    results = score_mechanisms(conn, {"big_five_N": "+"}, [])
    assert results[0]["id"] == "strong_one"
    assert results[1]["id"] == "weak_one"


def test_top_n_limits_results():
    conn = _make_db()
    for i in range(5):
        _add_mech(conn, f"m{i}")
        _add_pm(conn, f"m{i}", "big_five_N", "+")
    results = score_mechanisms(conn, {"big_five_N": "+"}, [], top_n=3)
    assert len(results) == 3


def test_empty_profile_and_situation():
    """No profile, no situation → no mechanism scores above 0 → empty results."""
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+")
    results = score_mechanisms(conn, {}, [])
    assert len(results) == 0


# ─── Scoring: multiplicative formula ─────────────────────────────────────────


def test_multiplicative_formula():
    """person=1.5, sit=4.0 → 1.5 * (1 + 4.0 * 0.5) = 4.5"""
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+", strength="strong")  # 1.5
    _add_sa(conn, "m1", "stakes", "activates")  # +2.0
    _add_sa(conn, "m1", "social_visibility", "activates")  # +2.0
    results = score_mechanisms(conn, {"big_five_N": "+"}, ["stakes", "social_visibility"])
    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(4.5)


def test_person_zero_excludes_despite_situation():
    """person_score=0 + high situation → excluded (situation can't rescue personality mismatch)."""
    conn = _make_db()
    _add_mech(conn, "m1")
    _add_pm(conn, "m1", "big_five_N", "+")
    _add_sa(conn, "m1", "stakes", "activates")
    _add_sa(conn, "m1", "social_visibility", "activates")
    # Profile has no matching dimensions → person_score = 0
    results = score_mechanisms(conn, {"big_five_E": "+"}, ["stakes", "social_visibility"])
    assert len(results) == 0


def test_situation_multiplier_constant():
    assert SITUATION_MULTIPLIER == 0.5


# ─── _text_to_features ────────────────────────────────────────────────────────


def test_text_to_features_stakes():
    feats = _text_to_features("This is a high stakes decision")
    assert "stakes" in feats


def test_text_to_features_time_pressure():
    feats = _text_to_features("There's a tight deadline and we need to decide quickly")
    assert "time_pressure" in feats


def test_text_to_features_multiple():
    feats = _text_to_features("public negotiation under time pressure with a rival group")
    assert "social_visibility" in feats
    assert "time_pressure" in feats
    assert "out_group_salience" in feats


def test_text_to_features_case_insensitive():
    feats = _text_to_features("URGENT decision in front of an AUDIENCE")
    assert "time_pressure" in feats
    assert "social_visibility" in feats


def test_text_to_features_no_match():
    feats = _text_to_features("a quiet afternoon reading a book")
    assert feats == []


def test_text_to_features_prior_commitment():
    feats = _text_to_features("I already promised to do this and I'm committed")
    assert "prior_commitment" in feats


def test_text_to_features_power_holder():
    feats = _text_to_features("She's in charge and holds power over the team")
    assert "power_holder" in feats


def test_text_to_features_power_low():
    feats = _text_to_features("He's junior and answering to the director")
    assert "power_low" in feats
