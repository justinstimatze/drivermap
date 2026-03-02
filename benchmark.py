#!/usr/bin/env python3
"""
benchmark.py — Validation suite for the behavioral mechanisms knowledge base.

Modes:
    --synthetic        Paradigm recall: canonical profile+situation → recall@3, recall@5
                       Near-miss: remove key activator → verify rank drop
    --speed-dating     Fisman rationalization gap correlation (requires CSV; see --help)
    --atomic           ATOMIC 2020 output chain alignment
    --rationalization  Rationalization template coherence: ATOMIC xIntent→mechanism→
                       template→PLO vs xReact (vocabulary coherence + lift over random)
    --social-chem      Rationalization coherence via Social Chemistry 101 ROTs:
                       action+situation→mechanism→template→PLO vs rule-of-thumb sentence
    --all              Run all available benchmarks

Usage:
    python benchmark.py --synthetic
    python benchmark.py --synthetic --verbose
    python benchmark.py --speed-dating --data data/speed_dating_data.csv
    python benchmark.py --atomic
    python benchmark.py --rationalization
    python benchmark.py --rationalization --sample 37344   # full dataset
    python benchmark.py --social-chem
    python benchmark.py --social-chem --sc-sample 10000
    python benchmark.py --all
"""

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "db" / "mechanisms.sqlite"

# Import scoring logic directly from mcp_server (mcp.run() is gated by __main__)
sys.path.insert(0, str(ROOT))
from mcp_server import _score_mechanisms

# ─── DB ───────────────────────────────────────────────────────────────────────


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


# ─── Synthetic Benchmark ──────────────────────────────────────────────────────


def _build_canonical(conn, mid: str) -> tuple[dict, list[str]]:
    """
    Build the most-activating profile+situation for a mechanism from its own DB entries.

    profile:   {dim: dir} for all non-mixed person_moderators
    situation: [feature] for all activates/required situation_activators
               (dampens features intentionally excluded)
    """
    pms = conn.execute(
        "SELECT dimension, direction FROM person_moderators WHERE mechanism_id=?", (mid,)
    ).fetchall()
    sas = conn.execute(
        "SELECT feature, effect FROM situation_activators WHERE mechanism_id=?", (mid,)
    ).fetchall()

    profile = {pm["dimension"]: pm["direction"] for pm in pms if pm["direction"] in ("+", "-")}
    situation = [sa["feature"] for sa in sas if sa["effect"] in ("activates", "required")]
    return profile, situation


def _near_miss_feature(conn, mid: str, situation: list[str]) -> tuple[str | None, bool]:
    """
    Return (feature_to_remove, is_required).
    Prioritises 'required' features (removal → exclusion), then 'activates' (removal → rank drop).
    Returns (None, False) if nothing to remove.
    """
    sas = conn.execute(
        "SELECT feature, effect FROM situation_activators WHERE mechanism_id=?", (mid,)
    ).fetchall()
    required = [
        sa["feature"] for sa in sas if sa["effect"] == "required" and sa["feature"] in situation
    ]
    activates = [
        sa["feature"] for sa in sas if sa["effect"] == "activates" and sa["feature"] in situation
    ]
    if required:
        return required[0], True
    if activates:
        return activates[0], False
    return None, False


def run_synthetic(conn, verbose: bool = False) -> dict:
    mechs = conn.execute("SELECT id, name, domain FROM mechanisms ORDER BY domain, name").fetchall()

    stats = {
        "total": 0,
        "untestable": 0,
        "recall_at_3": 0,
        "recall_at_5": 0,
        "near_miss_tested": 0,
        "near_miss_pass": 0,
        "by_domain": defaultdict(lambda: {"total": 0, "r3": 0, "r5": 0}),
        "failures_at_5": [],
        "near_miss_failures": [],
    }

    for mech in mechs:
        mid = mech["id"]
        domain = mech["domain"] or "unknown"

        profile, situation = _build_canonical(conn, mid)

        if not profile and not situation:
            stats["untestable"] += 1
            continue

        stats["total"] += 1
        stats["by_domain"][domain]["total"] += 1

        # Score all mechanisms using the canonical setup; request all (top_n=200)
        ranked = _score_mechanisms(conn, profile, situation, top_n=200)
        ids = [r["id"] for r in ranked]
        rank = ids.index(mid) + 1 if mid in ids else 999

        if rank <= 3:
            stats["recall_at_3"] += 1
            stats["by_domain"][domain]["r3"] += 1
        if rank <= 5:
            stats["recall_at_5"] += 1
            stats["by_domain"][domain]["r5"] += 1
        else:
            entry = {
                "id": mid,
                "name": mech["name"],
                "domain": domain,
                "rank": rank,
                "profile_dims": len(profile),
                "situation_feats": len(situation),
                "score": next((r["score"] for r in ranked if r["id"] == mid), 0),
            }
            if verbose:
                # Add top-3 competing mechanisms for diagnosis
                entry["top3"] = [{"id": r["id"], "score": r["score"]} for r in ranked[:3]]
            stats["failures_at_5"].append(entry)

        # Near-miss test
        test_feat, is_required = _near_miss_feature(conn, mid, situation)
        if test_feat:
            stats["near_miss_tested"] += 1
            nm_situation = [f for f in situation if f != test_feat]
            nm_ranked = _score_mechanisms(conn, profile, nm_situation, top_n=200)
            nm_ids = [r["id"] for r in nm_ranked]
            nm_rank = nm_ids.index(mid) + 1 if mid in nm_ids else 999
            nm_score = next((r["score"] for r in nm_ranked if r["id"] == mid), 0.0)
            base_score = next((r["score"] for r in ranked if r["id"] == mid), 0.0)

            if is_required:
                # Required feature removed → mechanism should be excluded
                passed = nm_rank == 999
            else:
                # Activates feature removed → rank should drop OR score should drop
                # (rank can't drop if already #1, so also accept score reduction)
                passed = nm_rank > rank or nm_score < base_score

            if passed:
                stats["near_miss_pass"] += 1
            else:
                stats["near_miss_failures"].append(
                    {
                        "id": mid,
                        "feat_removed": test_feat,
                        "required": is_required,
                        "rank_before": rank,
                        "rank_after": nm_rank,
                        "score_before": round(base_score, 2),
                        "score_after": round(nm_score, 2),
                    }
                )

    return stats


def print_synthetic(stats: dict, verbose: bool = False):
    total = stats["total"]
    un = stats["untestable"]
    r3 = stats["recall_at_3"]
    r5 = stats["recall_at_5"]
    nm_t = stats["near_miss_tested"]
    nm_p = stats["near_miss_pass"]

    pct = lambda n, d: f"{100 * n / d:.1f}%" if d else "—"

    print("\n" + "═" * 60)
    print("  SYNTHETIC PARADIGM BENCHMARK")
    print("═" * 60)
    print(f"  Mechanisms tested : {total}  (untestable: {un})")
    print(f"  Recall @ 3        : {r3}/{total}  {pct(r3, total)}")
    print(f"  Recall @ 5        : {r5}/{total}  {pct(r5, total)}")
    print(f"  Near-miss (tested): {nm_t}")
    print(f"  Near-miss pass    : {nm_p}/{nm_t}  {pct(nm_p, nm_t)}")

    print("\n  By domain:")
    for domain, d in sorted(stats["by_domain"].items()):
        t = d["total"]
        print(f"    {domain:<40} R@3={pct(d['r3'], t):6}  R@5={pct(d['r5'], t):6}  n={t}")

    if stats["failures_at_5"]:
        print(f"\n  Failures at rank>5 ({len(stats['failures_at_5'])}):")
        for f in sorted(stats["failures_at_5"], key=lambda x: x["rank"]):
            top3 = ""
            if verbose and "top3" in f:
                top3 = "  vs " + ", ".join(f"{x['id']}({x['score']})" for x in f["top3"])
            print(f"    rank={f['rank']:3d}  {f['id']:<40}  {f['domain']}{top3}")

    if stats["near_miss_failures"]:
        print(f"\n  Near-miss no-sensitivity ({len(stats['near_miss_failures'])}):")
        for f in stats["near_miss_failures"]:
            req = "(required)" if f["required"] else "(activates)"
            print(
                f"    {f['id']:<40} removed={f['feat_removed']} {req}  "
                f"rank {f['rank_before']}→{f['rank_after']}  "
                f"score {f['score_before']}→{f['score_after']}"
            )

    print()


# ─── Speed Dating Benchmark ───────────────────────────────────────────────────

SPEED_DATING_DEFAULT = ROOT / "data" / "speed_dating_data.csv"
SPEED_DATING_URL = (
    "https://raw.githubusercontent.com/jepusto/fivethirtyeight/"
    "master/data-raw/speed_dating_data.csv"
)

SPEED_DATING_INSTRUCTIONS = """
Speed dating benchmark requires the Fisman et al. (2006) dataset.

Obtain it via:
  1. Direct download (may require manual save):
       https://www.stat.columbia.edu/~gelman/arm/examples/speed.dating/Speed%%20Dating%%20Data.csv
  2. Or search: "Speed Dating Experiment dataset Kaggle"

Save the CSV to:
  {path}

Then re-run:
  python benchmark.py --speed-dating
""".strip()


def _try_download_speed_dating(dest: Path) -> bool:
    """Attempt to fetch from known mirrors. Returns True on success."""
    try:
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        mirrors = [
            "https://raw.githubusercontent.com/joshcorr/notebooks/master/Speed%20Dating%20Data.csv",
            SPEED_DATING_URL,
        ]
        for url in mirrors:
            try:
                print(f"  Trying: {url}")
                urllib.request.urlretrieve(url, dest)
                return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def _demographics_to_profile(row: dict, sf) -> dict:
    """
    Map Fisman speed dating demographic variables to our dimension vocabulary.

    Dimensions covered: big_five_E, big_five_N, big_five_O, big_five_A, big_five_C,
    bas_sensitivity, bis_sensitivity, need_for_cognition, need_for_closure,
    in_group_salience, social_dominance_orientation, attachment_anxious
    """
    profile: dict[str, str] = {}

    # ── go_out (1=several/week … 7=almost never) → E / BAS / BIS ──
    go_out = sf(row.get("go_out"))
    if go_out is not None:
        if go_out <= 2:
            profile["big_five_E"] = "+"
            profile["bas_sensitivity"] = "+"
        elif go_out >= 6:
            profile["big_five_E"] = "-"
            profile["bis_sensitivity"] = "+"

    # ── goal (1=fun, 2=meet people, 3=get date, 4=serious relationship) ──
    goal = sf(row.get("goal"))
    if goal == 4.0:
        profile.setdefault("attachment_anxious", "+")
    elif goal == 1.0:
        profile.setdefault("bas_sensitivity", "+")

    # ── field_cd → personality/cognitive traits ──
    field = sf(row.get("field_cd"))
    FIELD_MAP = {
        1.0: {"need_for_closure": "+", "big_five_C": "+"},  # Law
        2.0: {"need_for_cognition": "+", "big_five_C": "+"},  # Math
        3.0: {"big_five_A": "+", "big_five_O": "+"},  # Social Science
        5.0: {"need_for_cognition": "+", "big_five_C": "+"},  # Engineering
        6.0: {"big_five_O": "+"},  # English / Creative Writing
        7.0: {"big_five_O": "+", "need_for_cognition": "+"},  # History / Philosophy
        8.0: {"big_five_C": "+", "social_dominance_orientation": "+"},  # Business / Econ
        10.0: {"need_for_cognition": "+", "big_five_C": "+"},  # Bio / Chem / Physics
        11.0: {"big_five_A": "+"},  # Social Work
    }
    for dim, val in FIELD_MAP.get(field, {}).items():
        profile.setdefault(dim, val)

    # ── imprace / imprelig → in_group_salience / need_for_closure ──
    imprace = sf(row.get("imprace"))
    if imprace is not None:
        if imprace >= 7:
            profile.setdefault("in_group_salience", "+")
        elif imprace <= 2:
            profile.setdefault("in_group_salience", "-")

    imprelig = sf(row.get("imprelig"))
    if imprelig is not None and imprelig >= 7:
        profile.setdefault("need_for_closure", "+")

    # ── sports + exercise avg → bas_sensitivity ──
    sports = sf(row.get("sports"))
    exer = sf(row.get("exercise"))
    if sports is not None and exer is not None:
        avg = (sports + exer) / 2
        if avg >= 7:
            profile.setdefault("bas_sensitivity", "+")
        elif avg <= 3:
            profile.setdefault("bas_sensitivity", "-")

    # ── cultural interests → big_five_O ──
    cult = [sf(row.get(c)) for c in ("art", "museums", "theater", "concerts")]
    cult = [v for v in cult if v is not None]
    if cult:
        avg_cult = sum(cult) / len(cult)
        if avg_cult >= 6:
            profile.setdefault("big_five_O", "+")
        elif avg_cult <= 2:
            profile.setdefault("big_five_O", "-")

    # ── exphappy (1–10) → big_five_N ──
    exphappy = sf(row.get("exphappy"))
    if exphappy is not None:
        if exphappy >= 7:
            profile.setdefault("big_five_N", "-")
        elif exphappy <= 3:
            profile.setdefault("big_five_N", "+")

    return profile


def _pearson_r(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Pearson r and two-tailed p-value (normal approximation for df ≥ 10)."""
    n = len(xs)
    if n < 4:
        return (0.0, 1.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) or 1e-12)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) or 1e-12)
    r = max(-1.0, min(1.0, num / (sx * sy)))
    t = r * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1 - r * r))
    # Two-tailed p via normal approximation (good for n > 30)
    import math as _m

    p = 2 * (1 - 0.5 * (1 + _m.erf(abs(t) / _m.sqrt(2))))
    return (round(r, 3), round(p, 4))


def run_speed_dating(conn, data_path: Path, min_decisions: int = 8) -> dict:
    import csv

    # ── Load CSV ──
    try:
        with open(data_path, encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return {"error": f"File not found: {data_path}"}
    if not rows:
        return {"error": "Empty CSV"}

    STATED_COLS = ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
    ATTR_COLS = ["attr_o", "sinc_o", "intel_o", "fun_o", "amb_o", "shar_o"]
    ATTR_NAMES = ["attractive", "sincere", "intelligent", "fun", "ambitious", "shared_interests"]
    DECISION_COL = "dec"

    available = set(rows[0].keys())
    missing = [c for c in STATED_COLS + ATTR_COLS + [DECISION_COL] if c not in available]
    if missing:
        return {"error": f"Missing columns: {missing}"}

    def sf(v):
        try:
            return float(v) if v not in ("", "NA", None) else None
        except (ValueError, TypeError):
            return None

    def normalize(vals):
        good = [v for v in vals if v is not None]
        s = sum(good) or 1.0
        return [(v / s if v is not None else 0.0) for v in vals]

    # ── Per-person stated preferences (first row per person) ──
    stated_by = {}
    demo_by = {}  # one demographics row per person
    for row in rows:
        iid = row.get("iid", "")
        if iid and iid not in stated_by:
            vals = [sf(row.get(c)) for c in STATED_COLS]
            if any(v is not None for v in vals):
                stated_by[iid] = normalize(vals)
                demo_by[iid] = row

    # ── Per-person dates: (decision, [attribute_ratings]) ──
    dates_by: dict[str, list] = defaultdict(list)
    for row in rows:
        iid = row.get("iid", "")
        dec = sf(row.get(DECISION_COL))
        if dec is None:
            continue
        ratings = [sf(row.get(c)) for c in ATTR_COLS]
        if all(r is None for r in ratings):
            continue
        dates_by[iid].append((dec, ratings))

    # ── Revealed weights via mini logistic regression ──
    def logistic_weights(obs):
        if len(obs) < min_decisions:
            return None
        n_f = 6
        w = [0.0] * n_f
        bias = 0.0
        lr = 0.05
        for _ in range(300):
            gw = [0.0] * n_f
            gb = 0.0
            for dec, ratings in obs:
                valid = [r for r in ratings if r is not None]
                mean_r = sum(valid) / len(valid) if valid else 5.0
                x = [(r if r is not None else mean_r) / 10.0 for r in ratings]
                z = max(-500, min(500, bias + sum(w[j] * x[j] for j in range(n_f))))
                pred = 1.0 / (1.0 + math.exp(-z))
                err = pred - dec
                for j in range(n_f):
                    gw[j] += err * x[j]
                gb += err
            n = len(obs)
            w = [w[j] - lr * gw[j] / n for j in range(n_f)]
            bias -= lr * gb / n
        w_pos = [max(0.0, wi) for wi in w]
        s = sum(w_pos) or 1.0
        return [wi / s for wi in w_pos]

    revealed_by = {iid: logistic_weights(obs) for iid, obs in dates_by.items()}

    # ── Speed dating situation (same for all participants) ──
    SD_SITUATION = [
        "social_visibility",  # decisions visible to partner
        "novelty",  # meeting strangers
        "time_pressure",  # 4-minute dates
        "relationship_type",  # romantic context
        "stakes",  # high-stakes (match or not)
        "group_context",  # group event setting
    ]
    DOMAINS = [
        "posthoc_rationalization",
        "loss_aversion_reference",
        "social_influence_compliance",
        "status_dominance",
        "threat_affective_priming",
        "ingroup_outgroup",
        "individual_variation",
    ]

    FIELD_NAMES = {
        1.0: "Law",
        2.0: "Math",
        3.0: "Social Science",
        4.0: "Medical",
        5.0: "Engineering",
        6.0: "English/Writing",
        7.0: "History/Philosophy",
        8.0: "Business/Econ",
        9.0: "Education",
        10.0: "Bio/Chem/Physics",
        11.0: "Social Work",
    }

    # ── Build per-person: gap + mechanism scores ──
    participants = []
    n_filtered = 0
    for iid, stated in stated_by.items():
        revealed = revealed_by.get(iid)
        if revealed is None:
            n_filtered += 1
            continue

        gap = math.sqrt(sum((s - r) ** 2 for s, r in zip(stated, revealed)))
        attr_gaps = sorted(
            [(ATTR_NAMES[i], abs(stated[i] - revealed[i])) for i in range(6)],
            key=lambda x: -x[1],
        )

        # Build person profile from demographics
        demo_row = demo_by.get(iid, {})
        profile = _demographics_to_profile(demo_row, sf)
        field_cd = sf(demo_row.get("field_cd"))

        # Score mechanisms with this profile
        ranked = _score_mechanisms(conn, profile, SD_SITUATION, top_n=200)
        domain_scores = defaultdict(float)
        mech_scores = {}
        for r in ranked:
            domain_scores[r["domain"]] += r["score"]
            mech_scores[r["id"]] = r["score"]

        participants.append(
            {
                "iid": iid,
                "gap": gap,
                "stated": dict(zip(ATTR_NAMES, stated)),
                "revealed": dict(zip(ATTR_NAMES, revealed)),
                "biggest_gap": attr_gaps[0],
                "profile": profile,
                "profile_dims": len(profile),
                "domain_scores": dict(domain_scores),
                "mech_scores": mech_scores,
                "field_cd": field_cd,
            }
        )

    if not participants:
        return {"error": "No participants passed the decision filter"}

    participants.sort(key=lambda x: -x["gap"])
    n = len(participants)

    # ── Correlations: gap vs each domain score ──
    gap_vals = [p["gap"] for p in participants]
    correlations = {}
    for domain in DOMAINS:
        scores = [p["domain_scores"].get(domain, 0.0) for p in participants]
        r, pval = _pearson_r(gap_vals, scores)
        correlations[domain] = {"r": r, "p": pval}

    # ── Mechanism-level correlations (drilldown below domain) ──
    all_mechs = conn.execute(
        "SELECT id, name, domain FROM mechanisms ORDER BY domain, name"
    ).fetchall()
    mech_correlations = []
    for m in all_mechs:
        mid = m["id"]
        mscores = [p["mech_scores"].get(mid, 0.0) for p in participants]
        r, pval = _pearson_r(gap_vals, mscores)
        if abs(r) >= 0.05:  # skip near-zero correlations
            mech_correlations.append(
                {
                    "id": mid,
                    "name": m["name"],
                    "domain": m["domain"],
                    "r": r,
                    "p": pval,
                }
            )
    mech_correlations.sort(key=lambda x: -abs(x["r"]))

    # ── Field breakdown: avg gap + avg status_dominance by field_cd ──
    field_groups: dict = defaultdict(list)
    for p in participants:
        key = p["field_cd"]
        field_groups[key].append(p)

    field_breakdown = []
    for fcd, ps in field_groups.items():
        if len(ps) < 5:
            continue
        avg_gap = sum(p["gap"] for p in ps) / len(ps)
        avg_sd = sum(p["domain_scores"].get("status_dominance", 0.0) for p in ps) / len(ps)
        field_breakdown.append(
            {
                "field": FIELD_NAMES.get(fcd, f"field_{fcd}") if fcd is not None else "Unknown",
                "n": len(ps),
                "avg_gap": round(avg_gap, 3),
                "avg_status_dominance": round(avg_sd, 2),
            }
        )
    field_breakdown.sort(key=lambda x: -x["avg_gap"])

    # ── Attribute-level mismatch distribution ──
    biggest_gap_attrs: dict[str, int] = defaultdict(int)
    for p in participants:
        biggest_gap_attrs[p["biggest_gap"][0]] += 1

    mean_gap = sum(gap_vals) / n
    mean_dims = sum(p["profile_dims"] for p in participants) / n

    return {
        "n_participants": n,
        "n_filtered_low_decisions": n_filtered,
        "min_decisions_required": min_decisions,
        "mean_gap": round(mean_gap, 3),
        "max_gap": round(max(gap_vals), 3),
        "min_gap": round(min(gap_vals), 3),
        "mean_profile_dims": round(mean_dims, 1),
        "top_gap_attr": sorted(biggest_gap_attrs.items(), key=lambda x: -x[1]),
        "domain_correlations": correlations,
        "mechanism_correlations": mech_correlations,
        "field_breakdown": field_breakdown,
        "high_gap_examples": participants[:3],
        "low_gap_examples": participants[-3:],
    }


def print_speed_dating(stats: dict):
    if "error" in stats:
        print(f"\n  Speed dating: ERROR — {stats['error']}")
        return

    print("\n" + "═" * 60)
    print("  SPEED DATING RATIONALIZATION GAP BENCHMARK")
    print("═" * 60)
    print(
        f"  Participants     : {stats['n_participants']}  "
        f"(filtered out <{stats['min_decisions_required']} decisions: "
        f"{stats['n_filtered_low_decisions']})"
    )
    print(
        f"  Mean profile dims: {stats['mean_profile_dims']:.1f} dimensions inferred from demographics"
    )
    print(
        f"  Gap (stated vs revealed): mean={stats['mean_gap']:.3f}  "
        f"min={stats['min_gap']:.3f}  max={stats['max_gap']:.3f}"
    )

    print("\n  Most-rationalized attribute (stated ≠ revealed):")
    max_c = max(c for _, c in stats["top_gap_attr"]) or 1
    for attr, count in stats["top_gap_attr"][:6]:
        bar = "█" * (count * 20 // max_c)
        print(f"    {attr:<20} {bar} ({count})")

    print("\n  Gap ~ Domain score correlations (Pearson r, p-value):")
    corrs = sorted(stats["domain_correlations"].items(), key=lambda x: -abs(x[1]["r"]))
    for domain, c in corrs:
        r, p = c["r"], c["p"]
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "  ")
        bar = "█" * int(abs(r) * 20)
        direction = "+" if r >= 0 else "-"
        print(f"    {domain:<35} {direction}{bar:<20} r={r:+.3f}  p={p:.4f} {sig}")

    print()
    best_domain = corrs[0][0]
    best_r = corrs[0][1]["r"]
    print(f"  Best predictor: {best_domain}  (r={best_r:+.3f})")

    if stats.get("mechanism_correlations"):
        print("\n  Top mechanisms by |r| with rationalization gap:")
        for m in stats["mechanism_correlations"][:15]:
            r, p = m["r"], m["p"]
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "  ")
            bar = "█" * int(abs(r) * 30)
            direction = "+" if r >= 0 else "-"
            domain_short = m["domain"].replace("_", " ")[:20]
            print(f"    {m['name']:<38} {direction}{bar:<10} r={r:+.3f} {sig}  [{domain_short}]")

    if stats.get("field_breakdown"):
        print("\n  Rationalization gap by field of study:")
        max_gap = max(fb["avg_gap"] for fb in stats["field_breakdown"]) or 1.0
        for fb in stats["field_breakdown"]:
            bar = "█" * int(fb["avg_gap"] / max_gap * 20)
            print(
                f"    {fb['field']:<25} {bar:<22} gap={fb['avg_gap']:.3f}  "
                f"n={fb['n']:3d}  sd_score={fb['avg_status_dominance']:.1f}"
            )

    print("\n  High-gap participants:")
    for p in stats["high_gap_examples"]:
        dims = ", ".join(f"{k}={v}" for k, v in list(p["profile"].items())[:4])
        ph_score = round(p["domain_scores"].get("posthoc_rationalization", 0), 1)
        sd_score = round(p["domain_scores"].get("status_dominance", 0), 1)
        print(
            f"    gap={p['gap']:.3f}  posthoc={ph_score}  status_dom={sd_score}  profile=[{dims}]"
        )
        print(
            f"      biggest miss: stated {p['biggest_gap'][0]}="
            f"{p['stated'][p['biggest_gap'][0]]:.2f} → "
            f"revealed {p['revealed'][p['biggest_gap'][0]]:.2f}"
        )

    print("\n  Low-gap participants:")
    for p in stats["low_gap_examples"]:
        dims = ", ".join(f"{k}={v}" for k, v in list(p["profile"].items())[:4])
        ph_score = round(p["domain_scores"].get("posthoc_rationalization", 0), 1)
        sd_score = round(p["domain_scores"].get("status_dominance", 0), 1)
        print(
            f"    gap={p['gap']:.3f}  posthoc={ph_score}  status_dom={sd_score}  profile=[{dims}]"
        )

    print()


# ─── Rationalization Benchmark ────────────────────────────────────────────────

# Reuse the situation-feature keyword map from query.py (inlined to avoid import)
_FEAT_KEYWORDS: dict[str, list[str]] = {
    "social_visibility": [
        "public",
        "audience",
        "watching",
        "observed",
        "visible",
        "crowd",
        "seen",
        "reputation",
        "everyone",
        "in front of",
        "on display",
    ],
    "stakes": [
        "important",
        "critical",
        "high-stakes",
        "significant",
        "serious",
        "major",
        "career",
        "money",
        "big deal",
        "life-changing",
        "consequences",
    ],
    "threat": ["danger", "threat", "attack", "harm", "fear", "dangerous", "scary", "risk"],
    "conflict_present": [
        "argument",
        "fight",
        "disagree",
        "conflict",
        "confront",
        "dispute",
        "rival",
        "enemy",
        "rude",
        "mean",
        "mad",
        "upset",
        "angry",
        "annoyed",
        "yell",
        "yelled",
        "yelling",
        "insult",
        "insulted",
        "bully",
        "bullying",
        "hostile",
        "harsh",
        "offend",
        "offended",
        "heated",
        "tension",
        "clash",
    ],
    "group_context": [
        "group",
        "team",
        "together",
        "meeting",
        "party",
        "everyone",
        "community",
        "colleague",
        "coworker",
        "neighbor",
    ],
    "novelty": ["new", "first", "unfamiliar", "strange", "unusual", "never", "unexpected"],
    "time_pressure": ["hurry", "urgent", "quickly", "deadline", "rush", "emergency", "fast"],
    "ambiguity": [
        "unclear",
        "uncertain",
        "confused",
        "ambiguous",
        "unsure",
        "maybe",
        "perhaps",
        "vague",
        "mixed signals",
        "not sure",
    ],
    "prior_commitment": [
        "promise",
        "committed",
        "agreed",
        "already",
        "invested",
        "pledged",
        "obligation",
        "signed up",
        "said I would",
    ],
    "in_group_salience": [
        "tribe",
        "members",
        "belong",
        "team",
        "we all",
        "our group",
        "one of us",
        "loyalty",
        "solidarity",
    ],
    "out_group_salience": [
        "them",
        "outsider",
        "stranger",
        "different",
        "foreign",
        "other",
        "out-group",
        "us vs",
        "not one of",
    ],
    "social_norms_clarity": [
        "should",
        "supposed",
        "expected",
        "normal",
        "appropriate",
        "acceptable",
        "proper",
        "etiquette",
        "rules",
        "protocol",
    ],
    "outcome_reversibility": [
        "undo",
        "reverse",
        "change",
        "fix",
        "cancel",
        "take back",
        "irreversible",
        "permanent",
        "no going back",
        "final",
        "locked in",
    ],
    "physical_threat": [
        "hurt",
        "pain",
        "injury",
        "physical",
        "violence",
        "attack",
        "hit",
        "unsafe",
        "danger",
        "threatened",
    ],
    "resource_scarcity": [
        "scarce",
        "limited",
        "shortage",
        "running out",
        "not enough",
        "rare",
        "constrained",
        "lack of",
    ],
    "anonymity": [
        "anonymous",
        "unknown",
        "private",
        "hidden",
        "secret",
        "identity",
        "incognito",
        "no one knows",
    ],
    "authority_present": [
        "authority",
        "boss",
        "leader",
        "supervisor",
        "official",
        "rule",
        "manager",
        "in charge",
    ],
    "relationship_type": [
        "friend",
        "partner",
        "spouse",
        "sibling",
        "coworker",
        "colleague",
        "roommate",
        "neighbor",
        "parent",
        "child",
        "family",
        "relative",
        "acquaintance",
        "close",
        "intimate",
    ],
}


def _intent_to_features(text: str) -> set[str]:
    """Extract situation features from free text using keyword matching."""
    features = set()
    for feat, kws in _FEAT_KEYWORDS.items():
        if any(kw in text.lower() for kw in kws):
            features.add(feat)
    return features


def run_rationalization(conn, sample: int = 5000) -> dict:
    """
    Rationalization template coherence benchmark using ATOMIC 2020 xIntent / xReact.

    For each ATOMIC row with non-none xIntent and xReact:
      1. Match xIntent keywords → best hidden mechanism (PLO + description overlap)
      2. Extract situation features from event + xIntent text
      3. Score posthoc_rationalization templates with those features
      4. Check: does the selected template's PLO overlap with xReact? (vocabulary coherence)

    Reports:
      coherence_rate  — % of matched rows where template PLO overlaps xReact
      baseline_rate   — expected overlap if template selected at random
      lift            — coherence_rate / baseline_rate
    """
    import csv as _csv
    import io
    import json as _json
    import random
    import tarfile
    import urllib.request

    ATOMIC_URL = "https://storage.googleapis.com/ai2-mosaic/public/atomic/v1.0/atomic_data.tgz"
    CACHE_CSV = ROOT / "data" / "atomic_v1_trn.csv"

    print("  Loading ATOMIC v1.0...")
    if not CACHE_CSV.exists():
        print(f"  Downloading from {ATOMIC_URL} ...")
        CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
        try:
            buf = urllib.request.urlopen(ATOMIC_URL, timeout=60).read()
            with tarfile.open(fileobj=io.BytesIO(buf)) as tar:
                for member in tar.getmembers():
                    if member.name.endswith("v4_atomic_trn.csv"):
                        CACHE_CSV.write_bytes(tar.extractfile(member).read())
                        break
        except Exception as e:
            return {"error": f"Failed to download ATOMIC v1.0: {e}"}

    _STOPWORDS = {
        "personx",
        "persony",
        "someone",
        "somebody",
        "something",
        "their",
        "there",
        "about",
        "would",
        "which",
        "could",
        "should",
        "these",
        "those",
        "being",
        "having",
        "doing",
        "after",
        "before",
        "people",
        "person",
        "things",
        "feels",
        "makes",
        "takes",
        "gives",
        "wants",
        "needs",
        "tries",
        "going",
        "comes",
        "with",
        "from",
        "that",
        "this",
        "they",
        "them",
        "than",
        "when",
        "what",
        "also",
        "back",
        "more",
        "some",
        "just",
        "like",
        "into",
        "over",
        "then",
        "have",
        "will",
        "very",
        "well",
        "know",
        "your",
        "here",
        "both",
        "each",
        "such",
        "even",
        "most",
        "much",
        "same",
        "down",
        "away",
        "able",
        "good",
        "feel",
        "want",
        "need",
        "make",
        "time",
        "life",
    }

    def _kw(text: str, min_len: int = 3) -> set[str]:
        if not text:
            return set()
        words: set[str] = set()

        def _ex(obj):
            if isinstance(obj, str):
                words.update(w.lower().rstrip(".,;:\"'/") for w in obj.split())
            elif isinstance(obj, list):
                for i in obj:
                    _ex(i)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _ex(v)

        try:
            _ex(_json.loads(text))
        except (ValueError, TypeError):
            _ex(text)
        return {w for w in words if len(w) >= min_len and w not in _STOPWORDS}

    # ── Load mechanisms: PLO + description keywords ──────────────────────────
    all_mechs = conn.execute(
        "SELECT id, name, domain, plain_language_outputs, description, summary FROM mechanisms"
    ).fetchall()

    # For mechanisms with no top-level description, pull from mechanism_properties
    _prop_defs: dict[str, str] = {}
    for pr in conn.execute(
        "SELECT mechanism_id, value FROM mechanism_properties WHERE key='definition'"
    ).fetchall():
        _prop_defs[pr["mechanism_id"]] = pr["value"] or ""

    mech_kw_map: list[tuple[str, str, str, set]] = []  # (id, name, domain, keywords)
    for m in all_mechs:
        mid = m["id"]
        desc = m["description"] or m["summary"] or _prop_defs.get(mid, "")
        kws = _kw(m["plain_language_outputs"]) | _kw(desc)
        mech_kw_map.append((mid, m["name"], m["domain"], kws))

    # ── Load posthoc_rationalization templates with PLO keyword sets ──────────
    rat_mechs = [m for m in mech_kw_map if m[2] == "posthoc_rationalization"]
    rat_plo_kw: dict[str, set] = {}
    for mid, name, domain, kws in rat_mechs:
        row = conn.execute(
            "SELECT plain_language_outputs FROM mechanisms WHERE id=?", (mid,)
        ).fetchone()
        rat_plo_kw[mid] = _kw(row["plain_language_outputs"])

    # ── Build situation→template score cache ─────────────────────────────────
    # Pre-fetch situation activators for all rat mechs
    rat_sas: dict[str, list[tuple[str, str]]] = {}
    for mid, *_ in rat_mechs:
        rat_sas[mid] = [
            (r["feature"], r["effect"])
            for r in conn.execute(
                "SELECT feature, effect FROM situation_activators WHERE mechanism_id=?", (mid,)
            ).fetchall()
        ]

    def _score_template(mid: str, situation: set[str]) -> float:
        score = 0.0
        for feat, effect in rat_sas[mid]:
            if effect == "required" and feat not in situation:
                return -999.0  # excluded
            if feat in situation:
                score += (
                    2.0
                    if effect in ("required", "activates")
                    else (1.0 if effect == "amplifies" else -1.0)
                )
        return score

    FALLBACK_TEMPLATE = "self_serving_bias"

    def _select_template(situation: set[str]) -> str:
        """Select best rationalization template for a situation."""
        best_id, best_score = FALLBACK_TEMPLATE, -float("inf")
        for mid, *_ in rat_mechs:
            s = _score_template(mid, situation)
            if s > best_score:
                best_score, best_id = s, mid
        if best_score <= 0:
            return FALLBACK_TEMPLATE
        return best_id

    # ── Load ATOMIC rows ──────────────────────────────────────────────────────
    raw_rows: list[tuple[str, list[str], list[str]]] = []
    with open(CACHE_CSV, encoding="utf-8", errors="replace") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            event = (row.get("event") or "").strip().lower()
            xi_raw = row.get("xIntent", "").strip()
            xr_raw = row.get("xReact", "").strip()
            if not xi_raw or not xr_raw:
                continue
            try:
                xi_items = [x for x in _json.loads(xi_raw) if x and x != "none"]
                xr_items = [x for x in _json.loads(xr_raw) if x and x != "none"]
            except (ValueError, TypeError):
                continue
            if xi_items and xr_items:
                raw_rows.append((event, xi_items, xr_items))

    print(f"  ATOMIC rows with xIntent + xReact: {len(raw_rows):,}")

    # Sample for speed
    if sample and len(raw_rows) > sample:
        random.seed(42)
        raw_rows = random.sample(raw_rows, sample)
    print(f"  Evaluating sample of {len(raw_rows):,} rows...")

    # ── Evaluate each row ─────────────────────────────────────────────────────
    n_intent_matched = 0
    n_coherent = 0
    template_hits: dict[str, int] = defaultdict(int)
    template_total: dict[str, int] = defaultdict(int)
    domain_hits: dict[str, int] = defaultdict(int)
    domain_total: dict[str, int] = defaultdict(int)

    # Baseline: for each xReact set, fraction of templates that overlap
    baseline_overlaps: list[float] = []

    examples_coherent: list[dict] = []
    examples_incoherent: list[dict] = []

    for event, xi_items, xr_items in raw_rows:
        # Combine all xIntent items
        xi_text = " ".join(xi_items)
        xi_kw = _kw(xi_text)

        # Match xIntent to best hidden mechanism
        best_mid, best_overlap = None, 0
        for mid, name, domain, mkw in mech_kw_map:
            ov = len(xi_kw & mkw)
            if ov > best_overlap:
                best_overlap, best_mid = ov, mid
                best_domain = domain

        if best_overlap == 0 or best_mid is None:
            continue  # no hidden mechanism identified

        n_intent_matched += 1

        # Extract situation features from event + xIntent
        situation = _intent_to_features(event + " " + xi_text)

        # Select rationalization template
        tmpl_id = _select_template(situation)

        # xReact keyword set
        xr_kw = _kw(" ".join(xr_items), min_len=3)

        # Coherence check: template PLO overlaps xReact
        tmpl_plo_kw = rat_plo_kw.get(tmpl_id, set())
        coherent = bool(tmpl_plo_kw & xr_kw)

        if coherent:
            n_coherent += 1
            template_hits[tmpl_id] += 1
            domain_hits[best_domain] += 1
            if len(examples_coherent) < 5:
                examples_coherent.append(
                    {
                        "event": event[:60],
                        "intent": xi_text[:60],
                        "hidden_mechanism": best_mid,
                        "template": tmpl_id,
                        "react": " / ".join(xr_items[:3]),
                        "overlap": sorted(tmpl_plo_kw & xr_kw),
                    }
                )
        else:
            if len(examples_incoherent) < 3:
                examples_incoherent.append(
                    {
                        "event": event[:60],
                        "intent": xi_text[:60],
                        "hidden_mechanism": best_mid,
                        "template": tmpl_id,
                        "react": " / ".join(xr_items[:3]),
                    }
                )

        template_total[tmpl_id] += 1
        domain_total[best_domain] += 1

        # Baseline: how many templates would overlap this xReact at random?
        n_overlap_any = sum(1 for mid, *_ in rat_mechs if rat_plo_kw.get(mid, set()) & xr_kw)
        baseline_overlaps.append(n_overlap_any / len(rat_mechs) if rat_mechs else 0)

    coherence_rate = n_coherent / n_intent_matched if n_intent_matched else 0
    baseline_rate = sum(baseline_overlaps) / len(baseline_overlaps) if baseline_overlaps else 0
    lift = coherence_rate / baseline_rate if baseline_rate > 0 else float("nan")

    return {
        "n_rows_sampled": len(raw_rows),
        "n_intent_matched": n_intent_matched,
        "n_coherent": n_coherent,
        "coherence_rate": round(coherence_rate * 100, 1),
        "baseline_rate": round(baseline_rate * 100, 1),
        "lift": round(lift, 2) if not math.isnan(lift) else None,
        "template_breakdown": {
            mid: {
                "hits": template_hits[mid],
                "total": template_total[mid],
                "rate": round(template_hits[mid] / template_total[mid] * 100, 1)
                if template_total[mid]
                else 0,
            }
            for mid in sorted(template_total, key=lambda x: -template_total[x])
        },
        "domain_breakdown": {
            dom: {
                "hits": domain_hits[dom],
                "total": domain_total[dom],
                "rate": round(domain_hits[dom] / domain_total[dom] * 100, 1)
                if domain_total[dom]
                else 0,
            }
            for dom in sorted(domain_total, key=lambda x: -domain_total[x])
        },
        "examples_coherent": examples_coherent,
        "examples_incoherent": examples_incoherent,
    }


def print_rationalization(stats: dict):
    if "error" in stats:
        print(f"\n  Rationalization benchmark: ERROR — {stats['error']}")
        return

    print("\n" + "═" * 60)
    print("  RATIONALIZATION TEMPLATE COHERENCE  [ATOMIC xIntent/xReact]")
    print("═" * 60)
    print(f"  ATOMIC rows sampled     : {stats['n_rows_sampled']:,}")
    print(f"  Intent-matched rows     : {stats['n_intent_matched']:,}")
    print(f"  Coherent (PLO∩xReact≥1) : {stats['n_coherent']:,}")
    print()
    print(f"  Coherence rate  : {stats['coherence_rate']:.1f}%  (template PLO overlaps xReact)")
    print(f"  Baseline rate   : {stats['baseline_rate']:.1f}%  (random template selection)")
    lift = stats.get("lift")
    if lift:
        print(f"  Lift            : {lift:.2f}×  ({'better' if lift > 1 else 'worse'} than random)")

    print("\n  Template selection breakdown (top 8):")
    for mid, d in list(stats["template_breakdown"].items())[:8]:
        bar = "█" * int(d["rate"] / 5)
        print(f"    {mid:<38} {d['rate']:5.1f}%  {bar}  (n={d['total']})")

    print("\n  By hidden-mechanism domain:")
    for dom, d in sorted(stats["domain_breakdown"].items(), key=lambda x: -x[1]["total"])[:6]:
        print(f"    {dom:<40} {d['rate']:5.1f}%  (n={d['total']})")

    if stats["examples_coherent"]:
        print("\n  Coherent examples (template PLO matches xReact):")
        for ex in stats["examples_coherent"][:3]:
            print(f"    event  : {ex['event']}")
            print(f"    intent : {ex['intent']}")
            print(f"    hidden : {ex['hidden_mechanism']}  →  template: {ex['template']}")
            print(f"    react  : {ex['react']}")
            print(f"    overlap: {ex['overlap']}")
            print()

    if stats["examples_incoherent"]:
        print("  Incoherent examples (template PLO misses xReact):")
        for ex in stats["examples_incoherent"][:2]:
            print(f"    event  : {ex['event']}")
            print(f"    intent : {ex['intent']}")
            print(f"    template: {ex['template']}  react: {ex['react']}")
            print()


# ─── ATOMIC Benchmark ─────────────────────────────────────────────────────────

ATOMIC_INSTRUCTIONS = """
ATOMIC 2020 benchmark requires the HuggingFace datasets library.

Install with:
  pip install datasets

Then re-run:
  python benchmark.py --atomic

Note: First run will download ~1GB of ATOMIC 2020 data.
""".strip()


def run_atomic(conn) -> dict:
    import csv as _csv
    import io
    import tarfile
    import urllib.request

    ATOMIC_URL = "https://storage.googleapis.com/ai2-mosaic/public/atomic/v1.0/atomic_data.tgz"
    CACHE_CSV = ROOT / "data" / "atomic_v1_trn.csv"
    EFFECT_COLS = ["xEffect", "xReact", "oEffect", "oReact"]

    # ── Load / cache ATOMIC v1.0 training CSV ──
    print("  Loading ATOMIC v1.0 (this may take a moment on first run)...")
    if not CACHE_CSV.exists():
        print(f"  Downloading from {ATOMIC_URL} ...")
        CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
        try:
            buf = urllib.request.urlopen(ATOMIC_URL, timeout=60).read()
            with tarfile.open(fileobj=io.BytesIO(buf)) as tar:
                for member in tar.getmembers():
                    if member.name.endswith("v4_atomic_trn.csv"):
                        CACHE_CSV.write_bytes(tar.extractfile(member).read())
                        break
        except Exception as e:
            return {"error": f"Failed to download ATOMIC v1.0: {e}"}

    if not CACHE_CSV.exists():
        return {"error": "ATOMIC v1.0 CSV not found after download"}

    # ── Build (event, effect_text) pairs ──
    # Each row: event, oEffect/xEffect/... are JSON arrays like ["annoyed","angry"] or ["none"]
    effect_heads: list[str] = []
    effect_tails: list[str] = []

    with open(CACHE_CSV, encoding="utf-8", errors="replace") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            event = (row.get("event") or "").lower().strip()
            if not event:
                continue
            for col in EFFECT_COLS:
                raw = (row.get(col) or "").strip()
                if not raw or raw in ("[]", ""):
                    continue
                try:
                    items = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    items = [raw]
                for item in items:
                    item = str(item).strip().lower()
                    if item and item != "none":
                        effect_heads.append(event)
                        effect_tails.append(item)

    print(f"  ATOMIC v1.0 — {len(effect_heads):,} (event, effect) pairs")

    # ── Pull mechanism data ──
    mechs = conn.execute(
        "SELECT id, name, domain, behavioral_outputs, outputs, "
        "plain_language_outputs, triggers "
        "FROM mechanisms"
    ).fetchall()

    # Psychology-relevant stopwords (exclude generic terms that would cause false matches)
    _STOPWORDS = {
        "someone",
        "somebody",
        "something",
        "somewhere",
        "personx",
        "persony",
        "their",
        "there",
        "about",
        "would",
        "which",
        "could",
        "should",
        "other",
        "these",
        "those",
        "where",
        "being",
        "having",
        "doing",
        "after",
        "before",
        "during",
        "because",
        "through",
        "between",
        "within",
        "another",
        "becomes",
        "become",
        "people",
        "person",
        "things",
        "feels",
        "makes",
        "takes",
        "gives",
        "wants",
        "needs",
        "tries",
        "going",
        "comes",
        "general",
        "specific",
        "result",
        "event",
        "place",
        "point",
        "group",
        "level",
        "order",
        "value",
        "state",
        "cause",
        "effect",
        "sense",
        "based",
        "given",
        "using",
        "happy",
        "upset",
        "think",
        "excited",
        "pleased",
        "annoyed",
        "worried",
        "normal",
        "others",
        # Additional high-frequency ATOMIC filler words (cause spurious ≥2 matches)
        "with",
        "from",
        "that",
        "this",
        "they",
        "them",
        "than",
        "when",
        "what",
        "also",
        "back",
        "more",
        "some",
        "just",
        "like",
        "into",
        "over",
        "then",
        "have",
        "will",
        "work",
        "home",
        "time",
        "only",
        "very",
        "well",
        "know",
        "your",
        "here",
        "both",
        "each",
        "such",
        "even",
        "most",
        "much",
        "same",
        "down",
        "away",
        "come",
        "behavior",
        "behaviors",
        "increased",
        "reduced",
        "increase",
        "decrease",
        "change",
        "changes",
        "results",
        "effects",
    }

    # Output-like property keys stored in mechanism_properties for mechanisms
    # that use alternate field names (behavioral_outcomes, resulting_behaviors, etc.)
    _OUTPUT_PROP_KEYS = (
        "behavioral_outcomes",
        "resulting_behaviors",
        "downstream_effects",
        "behavioral_effects",
        "consequences",
        "reactions",
        "responses",
        "emotional_effects",
        "cognitive_effects",
        "social_effects",
    )
    # Pre-fetch alternate outputs from mechanism_properties for NULL-outputs mechanisms
    _alt_outputs: dict[str, str] = {}
    prop_rows = conn.execute(
        f"SELECT mechanism_id, value FROM mechanism_properties "
        f"WHERE key IN ({','.join('?' * len(_OUTPUT_PROP_KEYS))})",
        _OUTPUT_PROP_KEYS,
    ).fetchall()
    for pr in prop_rows:
        mid = pr["mechanism_id"]
        _alt_outputs[mid] = (_alt_outputs.get(mid, "") + " " + (pr["value"] or "")).strip()

    def get_keywords(text: str | None, min_len: int = 4) -> set[str]:
        """Extract meaningful domain keywords from JSON (dict/list/string) or plain string."""
        if not text:
            return set()
        words: set[str] = set()

        def _extract(obj):
            if isinstance(obj, str):
                words.update(w.lower().rstrip(".,;:\"'/") for w in obj.split())
            elif isinstance(obj, list):
                for item in obj:
                    _extract(item)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _extract(v)

        try:
            _extract(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            words.update(w.lower().rstrip(".,;:\"'/") for w in text.split())

        return {w for w in words if len(w) >= min_len and w.lower() not in _STOPWORDS}

    # ── Match each mechanism against ATOMIC (event, effect) pairs ──
    results = []
    for mech in mechs:
        mid = mech["id"]
        # Prefer plain_language_outputs (everyday English) for matching; fall back to
        # academic behavioral_outputs and mechanism_properties alternate fields
        plo = mech["plain_language_outputs"]
        outputs = plo or mech["behavioral_outputs"] or mech["outputs"] or _alt_outputs.get(mid)
        triggers = mech["triggers"]

        output_kw = get_keywords(outputs)
        trigger_kw = get_keywords(triggers)

        if not output_kw:
            results.append(
                {
                    "id": mid,
                    "name": mech["name"],
                    "domain": mech["domain"],
                    "covered": False,
                    "reason": "no output keywords",
                }
            )
            continue

        matches = []
        for head, tail in zip(effect_heads, effect_tails):
            head_words = set(head.split())
            tail_words = set(tail.split())
            shared_out = output_kw & tail_words
            shared_trig = trigger_kw & head_words if trigger_kw else set()

            # Match: ≥2 output keywords in ATOMIC effect (stricter than single-word coincidence)
            # AND trigger keyword in event (or no trigger keywords to match against)
            if len(shared_out) >= 2 and (shared_trig or not trigger_kw):
                matches.append(
                    {
                        "head": head[:80],
                        "tail": tail[:60],
                        "shared_trigger": sorted(shared_trig)[:3],
                        "shared_output": sorted(shared_out)[:3],
                    }
                )
            if len(matches) >= 3:
                break

        covered = len(matches) > 0
        results.append(
            {
                "id": mid,
                "name": mech["name"],
                "domain": mech["domain"],
                "covered": covered,
                "n_matches": len(matches),
                "example": matches[0] if matches else None,
            }
        )

    n_covered = sum(1 for r in results if r["covered"])
    n_total = len(results)

    by_domain: dict = defaultdict(lambda: {"total": 0, "covered": 0})
    for r in results:
        d = r.get("domain") or "unknown"
        by_domain[d]["total"] += 1
        if r["covered"]:
            by_domain[d]["covered"] += 1

    return {
        "dataset_used": "atomic_v1.0",
        "n_mechanisms": n_total,
        "n_covered": n_covered,
        "coverage_pct": round(100 * n_covered / n_total, 1) if n_total else 0,
        "by_domain": dict(by_domain),
        "uncovered": [r["id"] for r in results if not r["covered"]],
        "examples": [r for r in results if r["covered"]][:5],
    }


def print_atomic(stats: dict):
    if "error" in stats:
        print(f"\n  ATOMIC benchmark: ERROR — {stats['error']}")
        if "not installed" in stats["error"]:
            print(f"\n{ATOMIC_INSTRUCTIONS}")
        return

    dset = stats.get("dataset_used", "commonsense KB")
    print("\n" + "═" * 60)
    print(f"  COMMONSENSE OUTPUT CHAIN ALIGNMENT  [{dset}]")
    print("═" * 60)
    print(f"  Mechanisms checked  : {stats['n_mechanisms']}")
    print(f"  Covered in ATOMIC   : {stats['n_covered']}  ({stats['coverage_pct']}%)")
    print("  NOTE: This is a vocabulary-breadth metric (keyword overlap).")
    print("        Academic output language rarely matches ATOMIC's everyday phrasing.")
    print("        Low scores reflect vocabulary gap, not missing concepts.")

    print("\n  By domain:")
    for domain, d in sorted(stats["by_domain"].items()):
        pct = f"{100 * d['covered'] / d['total']:.0f}%" if d["total"] else "—"
        print(f"    {domain:<40} {pct:5}  ({d['covered']}/{d['total']})")

    if stats["examples"]:
        print("\n  Alignment examples:")
        for ex in stats["examples"][:3]:
            e = ex["example"]
            if e:
                print(f"    [{ex['id']}]")
                print(f"      ATOMIC head  : {e['head']}")
                print(f"      ATOMIC tail  : {e['tail']}")
                print(f"      shared trig  : {e['shared_trigger']}")
                print(f"      shared output: {e['shared_output']}")

    if stats["uncovered"]:
        print(f"\n  Uncovered ({len(stats['uncovered'])}):")
        print("    " + ", ".join(stats["uncovered"][:20]))

    print()


# ─── Social Chemistry 101 Benchmark ─────────────────────────────────────────


def run_social_chem(conn, sample: int = 5000) -> dict:
    """
    Rationalization template coherence benchmark using Social Chemistry 101.

    Uses rules-of-thumb (ROTs) as the expressed normative rationalization and
    action+situation text as the behavioral context to infer hidden mechanisms.

    Key advantage over ATOMIC xIntent/xReact: ROTs are full natural-language
    sentences (vs. single-word xReact fragments) → richer vocabulary overlap
    with PLO → stronger coherence signal.

    Algorithm:
      1. Filter: action-agency=="agency", rot-agree >= 3.0
      2. Match action+situation → best hidden mechanism (PLO+description overlap)
      3. Extract situation features from situation text
      4. Select posthoc_rationalization template for those features
      5. Check: template PLO ∩ ROT content-words ≥ 1

    Stratifies by rot-moral-foundations (loyalty-betrayal, care-harm, etc.)
    Data: Social Chemistry 101 v1.0 (Forbes et al., EMNLP 2020)
    """
    import csv as _csv
    import io
    import random
    import urllib.request
    import zipfile

    SC_URL = (
        "https://storage.googleapis.com/ai2-mosaic-public/projects/"
        "social-chemistry/data/social-chem-101.zip"
    )
    CACHE = ROOT / "data" / "social_chem_101.tsv"

    print("  Loading Social Chemistry 101...")
    if not CACHE.exists():
        print(f"  Downloading from {SC_URL} ...")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        try:
            buf = urllib.request.urlopen(SC_URL, timeout=120).read()
            with zipfile.ZipFile(io.BytesIO(buf)) as zf:
                tsv_name = next((n for n in zf.namelist() if n.endswith(".tsv")), None)
                if not tsv_name:
                    return {"error": "No .tsv file found in Social Chemistry zip"}
                CACHE.write_bytes(zf.read(tsv_name))
            print(f"  Cached to {CACHE}")
        except Exception as e:
            return {"error": f"Failed to download Social Chemistry 101: {e}"}

    # ── Stopwords tuned for ROT sentences ────────────────────────────────────
    _STOPWORDS = {
        "it's",
        "its",
        "you",
        "your",
        "they",
        "them",
        "their",
        "theirs",
        "should",
        "would",
        "could",
        "might",
        "must",
        "will",
        "shall",
        "that",
        "this",
        "these",
        "those",
        "then",
        "than",
        "when",
        "what",
        "which",
        "where",
        "while",
        "there",
        "here",
        "have",
        "has",
        "had",
        "being",
        "been",
        "were",
        "are",
        "was",
        "not",
        "don't",
        "doesn't",
        "people",
        "person",
        "someone",
        "somebody",
        "anyone",
        "everybody",
        "with",
        "from",
        "into",
        "onto",
        "over",
        "under",
        "about",
        "also",
        "just",
        "even",
        "very",
        "more",
        "most",
        "some",
        "only",
        "too",
        "for",
        "and",
        "but",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "of",
        "is",
        "be",
        "do",
        "get",
        "got",
        "make",
        "made",
        "like",
        "good",
        "bad",
        "okay",
        "fine",
        "wrong",
        "right",
        "important",
        "expected",
        "normal",
        "appropriate",
        "acceptable",
    }

    def _kw(text: str, min_len: int = 3) -> set[str]:
        if not text:
            return set()
        words: set[str] = set()
        for w in str(text).lower().split():
            words.add(w.rstrip(".,;:\"'!/?)"))
        return {w for w in words if len(w) >= min_len and w not in _STOPWORDS}

    # ── Load mechanism data (PLO + description keywords) ─────────────────────
    all_mechs = conn.execute(
        "SELECT id, name, domain, plain_language_outputs, description, summary FROM mechanisms"
    ).fetchall()

    _prop_defs: dict[str, str] = {
        pr["mechanism_id"]: pr["value"] or ""
        for pr in conn.execute(
            "SELECT mechanism_id, value FROM mechanism_properties WHERE key='definition'"
        ).fetchall()
    }

    mech_kw_map: list[tuple[str, str, str, set]] = []
    for m in all_mechs:
        mid = m["id"]
        desc = m["description"] or m["summary"] or _prop_defs.get(mid, "")
        kws = _kw(m["plain_language_outputs"]) | _kw(desc)
        mech_kw_map.append((mid, m["name"], m["domain"], kws))

    # ── Load posthoc_rationalization templates ────────────────────────────────
    rat_mechs = [m for m in mech_kw_map if m[2] == "posthoc_rationalization"]

    rat_plo_kw: dict[str, set] = {}
    for mid, *_ in rat_mechs:
        row = conn.execute(
            "SELECT plain_language_outputs FROM mechanisms WHERE id=?", (mid,)
        ).fetchone()
        rat_plo_kw[mid] = _kw(row["plain_language_outputs"])

    rat_sas: dict[str, list[tuple[str, str]]] = {}
    for mid, *_ in rat_mechs:
        rat_sas[mid] = [
            (r["feature"], r["effect"])
            for r in conn.execute(
                "SELECT feature, effect FROM situation_activators WHERE mechanism_id=?",
                (mid,),
            ).fetchall()
        ]

    def _score_template(mid: str, situation: set[str]) -> float:
        score = 0.0
        for feat, effect in rat_sas[mid]:
            if effect == "required" and feat not in situation:
                return -999.0
            if feat in situation:
                score += (
                    2.0
                    if effect in ("required", "activates")
                    else (1.0 if effect == "amplifies" else -1.0)
                )
        return score

    FALLBACK = "self_serving_bias"

    def _select_template(situation: set[str]) -> str:
        best_id, best_score = FALLBACK, -float("inf")
        for mid, *_ in rat_mechs:
            s = _score_template(mid, situation)
            if s > best_score:
                best_score, best_id = s, mid
        return best_id if best_score > 0 else FALLBACK

    # ── Load Social Chemistry 101 rows ────────────────────────────────────────
    raw_rows: list[dict] = []
    with open(CACHE, encoding="utf-8", errors="replace") as f:
        reader = _csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (row.get("action-agency") or "").strip() != "agency":
                continue
            try:
                agree = float(row.get("rot-agree") or 0)
            except (ValueError, TypeError):
                agree = 0.0
            if agree < 3.0:
                continue
            action = (row.get("action") or "").strip().lower()
            situation = (row.get("situation") or "").strip().lower()
            rot = (row.get("rot") or "").strip()
            mf = (row.get("rot-moral-foundations") or "").strip()
            if not action or not rot:
                continue
            raw_rows.append(
                {
                    "situation": situation,
                    "action": action,
                    "rot": rot,
                    "moral_foundations": mf,
                }
            )

    print(f"  Social Chem rows (agency, agree≥3): {len(raw_rows):,}")

    if sample and len(raw_rows) > sample:
        random.seed(42)
        raw_rows = random.sample(raw_rows, sample)
    print(f"  Evaluating sample of {len(raw_rows):,} rows...")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    n_action_matched = 0
    n_coherent = 0
    n_any_coherent = 0  # oracle: ANY template overlaps ROT
    template_hits: dict[str, int] = defaultdict(int)
    template_total: dict[str, int] = defaultdict(int)
    best_template_hits: dict[str, int] = defaultdict(int)  # oracle best-template counts
    mf_hits: dict[str, int] = defaultdict(int)
    mf_total: dict[str, int] = defaultdict(int)
    domain_hits: dict[str, int] = defaultdict(int)
    domain_total: dict[str, int] = defaultdict(int)
    baseline_overlaps: list[float] = []
    examples_coherent: list[dict] = []
    examples_incoherent: list[dict] = []

    for row in raw_rows:
        query_kw = _kw(row["action"] + " " + row["situation"])

        # Match action+situation → best hidden mechanism
        best_mid, best_overlap, best_domain = None, 0, None
        for mid, name, domain, mkw in mech_kw_map:
            ov = len(query_kw & mkw)
            if ov > best_overlap:
                best_overlap, best_mid, best_domain = ov, mid, domain

        if best_overlap == 0 or best_mid is None:
            continue

        n_action_matched += 1

        # Situation features from situation text
        situation_feats = _intent_to_features(row["situation"])

        # Select rationalization template
        tmpl_id = _select_template(situation_feats)

        # ROT content-word set
        rot_kw = _kw(row["rot"])

        # Coherence: template PLO ∩ ROT content-words ≥ 1
        tmpl_plo_kw = rat_plo_kw.get(tmpl_id, set())
        coherent = bool(tmpl_plo_kw & rot_kw)

        # Oracle: which template has the MOST overlap with the ROT?
        overlaps_by_tmpl = {mid: len(rat_plo_kw.get(mid, set()) & rot_kw) for mid, *_ in rat_mechs}
        n_any = sum(1 for v in overlaps_by_tmpl.values() if v > 0)
        any_coherent = n_any > 0
        oracle_best = max(overlaps_by_tmpl, key=overlaps_by_tmpl.get) if any_coherent else None

        mfs = [mf.strip() for mf in row["moral_foundations"].split("|") if mf.strip()]

        if coherent:
            n_coherent += 1
            template_hits[tmpl_id] += 1
            domain_hits[best_domain] += 1
            for mf in mfs:
                mf_hits[mf] += 1
            if len(examples_coherent) < 5:
                examples_coherent.append(
                    {
                        "situation": row["situation"][:70],
                        "action": row["action"][:70],
                        "hidden_mechanism": best_mid,
                        "template": tmpl_id,
                        "rot": row["rot"][:100],
                        "overlap": sorted(tmpl_plo_kw & rot_kw),
                        "moral_foundations": row["moral_foundations"],
                    }
                )
        else:
            if len(examples_incoherent) < 3:
                examples_incoherent.append(
                    {
                        "situation": row["situation"][:70],
                        "action": row["action"][:70],
                        "hidden_mechanism": best_mid,
                        "template": tmpl_id,
                        "rot": row["rot"][:100],
                        "oracle_best": oracle_best,
                    }
                )

        if any_coherent:
            n_any_coherent += 1
            best_template_hits[oracle_best] += 1

        template_total[tmpl_id] += 1
        domain_total[best_domain] += 1
        for mf in mfs:
            mf_total[mf] += 1

        # Baseline: fraction of random templates that would overlap this ROT
        baseline_overlaps.append(n_any / len(rat_mechs) if rat_mechs else 0)

    coherence_rate = n_coherent / n_action_matched if n_action_matched else 0
    any_coherence_rate = n_any_coherent / n_action_matched if n_action_matched else 0
    selection_cost = any_coherence_rate - coherence_rate
    baseline_rate = sum(baseline_overlaps) / len(baseline_overlaps) if baseline_overlaps else 0
    lift = coherence_rate / baseline_rate if baseline_rate > 0 else float("nan")
    any_lift = any_coherence_rate / baseline_rate if baseline_rate > 0 else float("nan")

    return {
        "n_rows_sampled": len(raw_rows),
        "n_action_matched": n_action_matched,
        "n_coherent": n_coherent,
        "n_any_coherent": n_any_coherent,
        "coherence_rate": round(coherence_rate * 100, 1),
        "any_coherence_rate": round(any_coherence_rate * 100, 1),
        "selection_cost": round(selection_cost * 100, 1),
        "baseline_rate": round(baseline_rate * 100, 1),
        "lift": round(lift, 2) if not math.isnan(lift) else None,
        "any_lift": round(any_lift, 2) if not math.isnan(any_lift) else None,
        "best_template_breakdown": dict(sorted(best_template_hits.items(), key=lambda x: -x[1])),
        "template_breakdown": {
            mid: {
                "hits": template_hits[mid],
                "total": template_total[mid],
                "rate": round(template_hits[mid] / template_total[mid] * 100, 1)
                if template_total[mid]
                else 0,
            }
            for mid in sorted(template_total, key=lambda x: -template_total[x])
        },
        "moral_foundation_breakdown": {
            mf: {
                "hits": mf_hits[mf],
                "total": mf_total[mf],
                "rate": round(mf_hits[mf] / mf_total[mf] * 100, 1) if mf_total[mf] else 0,
            }
            for mf in sorted(mf_total, key=lambda x: -mf_total[x])
        },
        "domain_breakdown": {
            dom: {
                "hits": domain_hits[dom],
                "total": domain_total[dom],
                "rate": round(domain_hits[dom] / domain_total[dom] * 100, 1)
                if domain_total[dom]
                else 0,
            }
            for dom in sorted(domain_total, key=lambda x: -domain_total[x])
        },
        "examples_coherent": examples_coherent,
        "examples_incoherent": examples_incoherent,
    }


def print_social_chem(stats: dict):
    if "error" in stats:
        print(f"\n  Social Chemistry benchmark: ERROR — {stats['error']}")
        return

    print("\n" + "═" * 60)
    print("  RATIONALIZATION COHERENCE  [Social Chemistry 101 ROTs]")
    print("═" * 60)
    print(f"  SC101 rows evaluated    : {stats['n_rows_sampled']:,}")
    print(f"  Action-matched rows     : {stats['n_action_matched']:,}")
    print(f"  Coherent (PLO∩ROT≥1)    : {stats['n_coherent']:,}")
    print(f"  Any-coherent (oracle)   : {stats['n_any_coherent']:,}")
    print()
    print(
        f"  Selected coherence rate : {stats['coherence_rate']:.1f}%  "
        "(PLO of selected template ∩ ROT)"
    )
    print(
        f"  Oracle coherence rate   : {stats['any_coherence_rate']:.1f}%  "
        "(PLO of any template ∩ ROT  ← ceiling)"
    )
    print(
        f"  Selection cost          : {stats['selection_cost']:.1f}%  "
        "(ceiling − selected  ← room for improvement)"
    )
    print(f"  Baseline rate           : {stats['baseline_rate']:.1f}%  (random template selection)")
    lift = stats.get("lift")
    any_lift = stats.get("any_lift")
    if lift:
        print(f"  Selected lift           : {lift:.2f}×")
    if any_lift:
        print(f"  Oracle lift             : {any_lift:.2f}×")

    if stats.get("best_template_breakdown"):
        print("\n  Oracle best-template breakdown (top 8 by ROT match):")
        total_any = stats["n_any_coherent"] or 1
        for mid, hits in list(stats["best_template_breakdown"].items())[:8]:
            bar = "█" * int(hits / total_any * 20)
            print(f"    {mid:<38} {hits:4d}  {bar}")

    print("\n  Selected template breakdown (top 8):")
    for mid, d in list(stats["template_breakdown"].items())[:8]:
        bar = "█" * int(d["rate"] / 5)
        print(f"    {mid:<38} {d['rate']:5.1f}%  {bar}  (n={d['total']})")

    print("\n  By moral foundation (ROT label):")
    for mf, d in list(stats["moral_foundation_breakdown"].items())[:8]:
        print(f"    {mf:<35} {d['rate']:5.1f}%  (n={d['total']})")

    print("\n  By hidden-mechanism domain:")
    for dom, d in sorted(stats["domain_breakdown"].items(), key=lambda x: -x[1]["total"])[:6]:
        print(f"    {dom:<40} {d['rate']:5.1f}%  (n={d['total']})")

    if stats["examples_coherent"]:
        print("\n  Coherent examples (template PLO ∩ ROT):")
        for ex in stats["examples_coherent"][:3]:
            print(f"    situation : {ex['situation']}")
            print(f"    action    : {ex['action']}")
            print(f"    hidden    : {ex['hidden_mechanism']}  →  template: {ex['template']}")
            print(f"    rot       : {ex['rot']}")
            print(f"    overlap   : {ex['overlap']}")
            print(f"    moral     : {ex['moral_foundations']}")
            print()

    if stats["examples_incoherent"]:
        print("  Incoherent examples (selected template misses ROT):")
        for ex in stats["examples_incoherent"][:2]:
            print(f"    situation : {ex['situation']}")
            print(f"    action    : {ex['action']}")
            print(f"    selected  : {ex['template']}  oracle: {ex.get('oracle_best', '?')}")
            print(f"    rot       : {ex['rot']}")
            print()


# ─── Social Chemistry Calibration (LLM vs keyword) ──────────────────────────


def run_sc_calibrate(conn, n: int = 200) -> dict:
    """
    Calibrate keyword coherence against LLM judgment on a small Social Chem sample.

    For each row: runs both keyword overlap check AND a Haiku binary YES/NO
    coherence check (via extract.call_claude / Max plan auth, no API key).

    Reports confusion matrix, agreement rate, per-method lift, and disagreement
    examples so you can see where the vocabulary gap hurts keyword matching.

    Intended to answer: how much signal does LLM scoring add over keyword overlap?
    """
    import concurrent.futures
    import csv as _csv
    import random

    from extract import call_claude

    CACHE = ROOT / "data" / "social_chem_101.tsv"
    if not CACHE.exists():
        return {"error": "social_chem_101.tsv not found — run --social-chem first to download"}

    # ── Shared infrastructure (same as run_social_chem) ──────────────────────
    _STOPWORDS = {
        "it's",
        "its",
        "you",
        "your",
        "they",
        "them",
        "their",
        "theirs",
        "should",
        "would",
        "could",
        "might",
        "must",
        "will",
        "shall",
        "that",
        "this",
        "these",
        "those",
        "then",
        "than",
        "when",
        "what",
        "which",
        "where",
        "while",
        "there",
        "here",
        "have",
        "has",
        "had",
        "being",
        "been",
        "were",
        "are",
        "was",
        "not",
        "don't",
        "doesn't",
        "people",
        "person",
        "someone",
        "somebody",
        "anyone",
        "everybody",
        "with",
        "from",
        "into",
        "onto",
        "over",
        "under",
        "about",
        "also",
        "just",
        "even",
        "very",
        "more",
        "most",
        "some",
        "only",
        "too",
        "for",
        "and",
        "but",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "of",
        "is",
        "be",
        "do",
        "get",
        "got",
        "make",
        "made",
        "like",
        "good",
        "bad",
        "okay",
        "fine",
        "wrong",
        "right",
        "important",
        "expected",
        "normal",
        "appropriate",
        "acceptable",
    }

    def _kw(text: str, min_len: int = 3) -> set[str]:
        if not text:
            return set()
        words: set[str] = set()
        for w in str(text).lower().split():
            words.add(w.rstrip(".,;:\"'!/?)"))
        return {w for w in words if len(w) >= min_len and w not in _STOPWORDS}

    all_mechs = conn.execute(
        "SELECT id, name, domain, plain_language_outputs, description, summary FROM mechanisms"
    ).fetchall()
    _prop_defs: dict[str, str] = {
        pr["mechanism_id"]: pr["value"] or ""
        for pr in conn.execute(
            "SELECT mechanism_id, value FROM mechanism_properties WHERE key='definition'"
        ).fetchall()
    }
    mech_kw_map = []
    for m in all_mechs:
        mid = m["id"]
        desc = m["description"] or m["summary"] or _prop_defs.get(mid, "")
        mech_kw_map.append(
            (mid, m["name"], m["domain"], _kw(m["plain_language_outputs"]) | _kw(desc))
        )

    rat_mechs = [m for m in mech_kw_map if m[2] == "posthoc_rationalization"]
    rat_plo_raw: dict[str, str] = {}
    rat_plo_kw: dict[str, set] = {}
    for mid, *_ in rat_mechs:
        row = conn.execute(
            "SELECT name, plain_language_outputs FROM mechanisms WHERE id=?", (mid,)
        ).fetchone()
        rat_plo_raw[mid] = row["plain_language_outputs"] or ""
        rat_plo_kw[mid] = _kw(rat_plo_raw[mid])

    rat_names: dict[str, str] = {mid: name for mid, name, *_ in rat_mechs}

    rat_sas: dict[str, list] = {
        mid: [
            (r["feature"], r["effect"])
            for r in conn.execute(
                "SELECT feature, effect FROM situation_activators WHERE mechanism_id=?", (mid,)
            ).fetchall()
        ]
        for mid, *_ in rat_mechs
    }

    def _score_template(mid, situation):
        score = 0.0
        for feat, effect in rat_sas[mid]:
            if effect == "required" and feat not in situation:
                return -999.0
            if feat in situation:
                score += (
                    2.0
                    if effect in ("required", "activates")
                    else (1.0 if effect == "amplifies" else -1.0)
                )
        return score

    def _select_template(situation):
        best_id, best_score = "self_serving_bias", -float("inf")
        for mid, *_ in rat_mechs:
            s = _score_template(mid, situation)
            if s > best_score:
                best_score, best_id = s, mid
        return best_id if best_score > 0 else "self_serving_bias"

    # ── Load + sample rows (separate seed from the 5K benchmark) ─────────────
    raw_rows = []
    with open(CACHE, encoding="utf-8", errors="replace") as f:
        reader = _csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (row.get("action-agency") or "").strip() != "agency":
                continue
            try:
                agree = float(row.get("rot-agree") or 0)
            except (ValueError, TypeError):
                agree = 0.0
            if agree < 3.0:
                continue
            action = (row.get("action") or "").strip().lower()
            situation = (row.get("situation") or "").strip().lower()
            rot = (row.get("rot") or "").strip()
            if not action or not rot:
                continue
            raw_rows.append({"situation": situation, "action": action, "rot": rot})

    random.seed(99)  # distinct from the 5K benchmark sample (seed=42)
    sample = random.sample(raw_rows, min(n, len(raw_rows)))
    print(f"  Calibration sample: {len(sample)} rows (seed=99)")

    # ── Build per-row inputs (mechanism matching + template selection) ────────
    prepared = []
    for row in sample:
        query_kw = _kw(row["action"] + " " + row["situation"])
        best_mid, best_overlap = None, 0
        for mid, name, domain, mkw in mech_kw_map:
            ov = len(query_kw & mkw)
            if ov > best_overlap:
                best_overlap, best_mid = ov, mid
        if best_overlap == 0 or best_mid is None:
            continue

        situation_feats = _intent_to_features(row["situation"])
        tmpl_id = _select_template(situation_feats)
        rot_kw = _kw(row["rot"])
        kw_coherent = bool(rat_plo_kw.get(tmpl_id, set()) & rot_kw)

        import json as _json

        try:
            plo_phrases = _json.loads(rat_plo_raw[tmpl_id])
            plo_excerpt = "; ".join(str(p) for p in plo_phrases[:4])
        except (ValueError, TypeError):
            plo_excerpt = rat_plo_raw[tmpl_id][:120]

        prepared.append(
            {
                "situation": row["situation"],
                "action": row["action"],
                "rot": row["rot"],
                "template_id": tmpl_id,
                "template_name": rat_names.get(tmpl_id, tmpl_id),
                "plo_excerpt": plo_excerpt,
                "kw_coherent": kw_coherent,
            }
        )

    print(f"  Mechanism-matched: {len(prepared)}/{len(sample)}")
    print(f"  Calling Haiku for LLM coherence scores ({len(prepared)} calls, parallel)...")

    # ── LLM coherence scoring (parallel via call_claude / Max plan) ───────────
    def _llm_score(item: dict) -> bool | None:
        prompt = (
            f"Situation: {item['situation']}\n"
            f"Behavior: {item['action']}\n"
            f'Expressed norm: "{item["rot"]}"\n\n'
            f"Rationalization template: {item['template_name']}\n"
            f"Template reasoning patterns: {item['plo_excerpt']}\n\n"
            "Does this template describe the same pattern of reasoning as the expressed norm?\n"
            "ONE WORD ONLY: YES or NO"
        )
        result = call_claude(prompt, model="haiku", timeout=30)
        if not result["ok"]:
            return None
        first_word = result["text"].strip().upper().split()[0] if result["text"].strip() else ""
        return first_word.startswith("Y")

    llm_results: list[bool | None] = [None] * len(prepared)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_llm_score, item): i for i, item in enumerate(prepared)}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            llm_results[idx] = fut.result()
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{len(prepared)} scored...")

    # ── Compute metrics ───────────────────────────────────────────────────────
    valid = [(p, llm) for p, llm in zip(prepared, llm_results) if llm is not None]
    n_valid = len(valid)

    kw_yes = sum(1 for p, _ in valid if p["kw_coherent"])
    llm_yes = sum(1 for _, llm in valid if llm)
    both_yes = sum(1 for p, llm in valid if p["kw_coherent"] and llm)
    both_no = sum(1 for p, llm in valid if not p["kw_coherent"] and not llm)
    kw_only = sum(1 for p, llm in valid if p["kw_coherent"] and not llm)  # kw FP
    llm_only = sum(1 for p, llm in valid if not p["kw_coherent"] and llm)  # kw FN

    kw_rate = kw_yes / n_valid if n_valid else 0
    llm_rate = llm_yes / n_valid if n_valid else 0
    agree = (both_yes + both_no) / n_valid if n_valid else 0

    # Baseline: fraction of random templates LLM would score YES for
    # (approximate — use kw baseline as proxy since LLM baseline would require
    #  scoring all templates for all rows; instead report kw baseline)
    kw_baseline = (
        sum(
            sum(1 for mid, *_ in rat_mechs if rat_plo_kw.get(mid, set()) & _kw(p["rot"]))
            / len(rat_mechs)
            for p, _ in valid
        )
        / n_valid
        if n_valid
        else 0
    )

    kw_lift = kw_rate / kw_baseline if kw_baseline > 0 else float("nan")
    llm_lift = llm_rate / kw_baseline if kw_baseline > 0 else float("nan")

    # Disagreement examples for diagnosis
    kw_fp = [p for p, llm in valid if p["kw_coherent"] and not llm][:4]  # kw YES, LLM NO
    kw_fn = [p for p, llm in valid if not p["kw_coherent"] and llm][:4]  # kw NO,  LLM YES

    return {
        "n_rows": n_valid,
        "kw_coherent": kw_yes,
        "llm_coherent": llm_yes,
        "both_coherent": both_yes,
        "both_incoherent": both_no,
        "kw_only": kw_only,
        "llm_only": llm_only,
        "agreement_rate": round(agree * 100, 1),
        "kw_rate": round(kw_rate * 100, 1),
        "llm_rate": round(llm_rate * 100, 1),
        "kw_baseline": round(kw_baseline * 100, 1),
        "kw_lift": round(kw_lift, 2) if not math.isnan(kw_lift) else None,
        "llm_lift": round(llm_lift, 2) if not math.isnan(llm_lift) else None,
        "kw_fp_examples": [
            {
                "action": p["action"][:60],
                "rot": p["rot"][:80],
                "template": p["template_name"],
                "plo": p["plo_excerpt"][:60],
            }
            for p in kw_fp
        ],
        "kw_fn_examples": [
            {
                "action": p["action"][:60],
                "rot": p["rot"][:80],
                "template": p["template_name"],
                "plo": p["plo_excerpt"][:60],
            }
            for p in kw_fn
        ],
    }


def print_sc_calibrate(stats: dict):
    if "error" in stats:
        print(f"\n  SC calibration: ERROR — {stats['error']}")
        return

    print("\n" + "═" * 60)
    print("  SC101 CALIBRATION  [keyword vs Haiku LLM coherence]")
    print("═" * 60)
    print(f"  Rows evaluated          : {stats['n_rows']}")
    print()
    print(f"  Keyword coherent        : {stats['kw_coherent']}  ({stats['kw_rate']:.1f}%)")
    print(f"  LLM coherent            : {stats['llm_coherent']}  ({stats['llm_rate']:.1f}%)")
    print(f"  Agreement rate          : {stats['agreement_rate']:.1f}%")
    print()
    print("  Confusion matrix:")
    print(f"    Both YES (true match) : {stats['both_coherent']}")
    print(f"    Both NO               : {stats['both_incoherent']}")
    print(f"    KW only (kw FP)       : {stats['kw_only']}  ← keyword says yes, LLM says no")
    print(f"    LLM only (kw FN)      : {stats['llm_only']}  ← LLM finds match keyword missed")
    print()
    print(f"  Baseline (random tmpl)  : {stats['kw_baseline']:.1f}%")
    print(f"  Keyword lift            : {stats['kw_lift']:.2f}×")
    print(f"  LLM lift                : {stats['llm_lift']:.2f}×")

    if stats["kw_fn_examples"]:
        print(
            f"\n  Keyword false negatives — LLM found match, keyword missed ({len(stats['kw_fn_examples'])}):"
        )
        for ex in stats["kw_fn_examples"]:
            print(f"    action  : {ex['action']}")
            print(f"    rot     : {ex['rot']}")
            print(f"    template: {ex['template']}  |  plo: {ex['plo']}")
            print()

    if stats["kw_fp_examples"]:
        print(
            f"  Keyword false positives — keyword said yes, LLM said no ({len(stats['kw_fp_examples'])}):"
        )
        for ex in stats["kw_fp_examples"]:
            print(f"    action  : {ex['action']}")
            print(f"    rot     : {ex['rot']}")
            print(f"    template: {ex['template']}  |  plo: {ex['plo']}")
            print()


# ─── LLM Template Selection Benchmark ───────────────────────────────────────


def run_sc_llm_select(conn, n: int = 500) -> dict:
    """
    Test Haiku-based template selection against keyword selection and oracle ceiling.

    Same SC101 sample as --social-chem (seed=42) but uses Haiku to choose the
    rationalization template given action+situation only (ROT not shown).

    3-way comparison:
      keyword — feature extraction + score_template (current system)
      LLM     — Haiku picks mechanism from labelled menu
      oracle  — any template whose PLO overlaps the ROT (theoretical ceiling)

    NOT included in --all; requires LLM calls (~$0.10–0.20 Haiku for n=500).
    """
    import concurrent.futures
    import csv as _csv
    import random

    from extract import call_claude

    CACHE = ROOT / "data" / "social_chem_101.tsv"
    if not CACHE.exists():
        return {"error": "social_chem_101.tsv not found — run --social-chem first to download"}

    # ── Shared helpers (same as run_social_chem) ──────────────────────────────
    _STOPWORDS = {
        "it's",
        "its",
        "you",
        "your",
        "they",
        "them",
        "their",
        "theirs",
        "should",
        "would",
        "could",
        "might",
        "must",
        "will",
        "shall",
        "that",
        "this",
        "these",
        "those",
        "then",
        "than",
        "when",
        "what",
        "which",
        "where",
        "while",
        "there",
        "here",
        "have",
        "has",
        "had",
        "being",
        "been",
        "were",
        "are",
        "was",
        "not",
        "don't",
        "doesn't",
        "people",
        "person",
        "someone",
        "somebody",
        "anyone",
        "everybody",
        "with",
        "from",
        "into",
        "onto",
        "over",
        "under",
        "about",
        "also",
        "just",
        "even",
        "very",
        "more",
        "most",
        "some",
        "only",
        "too",
        "for",
        "and",
        "but",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "of",
        "is",
        "be",
        "do",
        "get",
        "got",
        "make",
        "made",
        "like",
        "good",
        "bad",
        "okay",
        "fine",
        "wrong",
        "right",
        "important",
        "expected",
        "normal",
        "appropriate",
        "acceptable",
    }

    def _kw(text: str, min_len: int = 3) -> set[str]:
        if not text:
            return set()
        words: set[str] = set()
        for w in str(text).lower().split():
            words.add(w.rstrip(".,;:\"'!/?)"))
        return {w for w in words if len(w) >= min_len and w not in _STOPWORDS}

    # ── Load mechanisms ───────────────────────────────────────────────────────
    all_mechs = conn.execute(
        "SELECT id, name, domain, plain_language_outputs, description, summary FROM mechanisms"
    ).fetchall()

    _prop_defs = {
        pr["mechanism_id"]: pr["value"] or ""
        for pr in conn.execute(
            "SELECT mechanism_id, value FROM mechanism_properties WHERE key='definition'"
        ).fetchall()
    }

    mech_kw_map: list[tuple[str, str, str, set]] = []
    for m in all_mechs:
        mid = m["id"]
        desc = m["description"] or m["summary"] or _prop_defs.get(mid, "")
        kws = _kw(m["plain_language_outputs"]) | _kw(desc)
        mech_kw_map.append((mid, m["name"], m["domain"], kws))

    rat_mechs = [m for m in mech_kw_map if m[2] == "posthoc_rationalization"]

    rat_plo_kw: dict[str, set] = {}
    for mid, *_ in rat_mechs:
        row = conn.execute(
            "SELECT plain_language_outputs FROM mechanisms WHERE id=?", (mid,)
        ).fetchone()
        rat_plo_kw[mid] = _kw(row["plain_language_outputs"])

    rat_sas: dict[str, list[tuple[str, str]]] = {}
    for mid, *_ in rat_mechs:
        rat_sas[mid] = [
            (r["feature"], r["effect"])
            for r in conn.execute(
                "SELECT feature, effect FROM situation_activators WHERE mechanism_id=?", (mid,)
            ).fetchall()
        ]

    def _score_template(mid: str, situation: set[str]) -> float:
        score = 0.0
        for feat, effect in rat_sas[mid]:
            if effect == "required" and feat not in situation:
                return -999.0
            if feat in situation:
                score += (
                    2.0
                    if effect in ("required", "activates")
                    else (1.0 if effect == "amplifies" else -1.0)
                )
        return score

    FALLBACK = "self_serving_bias"

    def _select_template(situation: set[str]) -> str:
        best_id, best_score = FALLBACK, -float("inf")
        for mid, *_ in rat_mechs:
            s = _score_template(mid, situation)
            if s > best_score:
                best_score, best_id = s, mid
        return best_id if best_score > 0 else FALLBACK

    # ── Build LLM mechanism menu ──────────────────────────────────────────────
    valid_ids = {mid for mid, *_ in rat_mechs}
    menu_lines = []
    for mid, name, _, _ in rat_mechs:
        mrow = conn.execute(
            "SELECT description, summary FROM mechanisms WHERE id=?", (mid,)
        ).fetchone()
        desc = (mrow["description"] or mrow["summary"] or "").replace("\n", " ").strip()[:90]
        menu_lines.append(f"  {mid}: {name} — {desc}")
    menu_str = "\n".join(menu_lines)

    # ── Load SC101 rows (same filter as run_social_chem) ──────────────────────
    raw_rows: list[dict] = []
    with open(CACHE, encoding="utf-8", errors="replace") as f:
        reader = _csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (row.get("action-agency") or "").strip() != "agency":
                continue
            try:
                agree = float(row.get("rot-agree") or 0)
            except (ValueError, TypeError):
                agree = 0.0
            if agree < 3.0:
                continue
            action = (row.get("action") or "").strip().lower()
            situation = (row.get("situation") or "").strip().lower()
            rot = (row.get("rot") or "").strip()
            if not action or not rot:
                continue
            raw_rows.append({"situation": situation, "action": action, "rot": rot})

    print(f"  Loaded {len(raw_rows):,} SC101 rows (agency, agree≥3)")
    random.seed(42)
    if len(raw_rows) > n:
        raw_rows = random.sample(raw_rows, n)
    print(f"  Sample: {len(raw_rows)} rows  |  {len(rat_mechs)} posthoc mechanisms")

    # ── Pre-process: keyword match + keyword selection + oracle ───────────────
    prepared: list[dict] = []
    for row in raw_rows:
        query_kw = _kw(row["action"] + " " + row["situation"])
        best_mid, best_overlap = None, 0
        for mid, name, domain, mkw in mech_kw_map:
            ov = len(query_kw & mkw)
            if ov > best_overlap:
                best_overlap, best_mid = ov, mid
        if best_overlap == 0 or best_mid is None:
            continue

        situation_feats = _intent_to_features(row["situation"])
        kw_tmpl = _select_template(situation_feats)
        rot_kw = _kw(row["rot"])

        overlaps = {mid: len(rat_plo_kw.get(mid, set()) & rot_kw) for mid, *_ in rat_mechs}
        oracle_tmpl = (
            max(overlaps, key=overlaps.get) if any(v > 0 for v in overlaps.values()) else None
        )

        prepared.append(
            {
                "situation": row["situation"],
                "action": row["action"],
                "rot_kw": rot_kw,
                "kw_tmpl": kw_tmpl,
                "oracle_tmpl": oracle_tmpl,
                "kw_coherent": bool(rat_plo_kw.get(kw_tmpl, set()) & rot_kw),
                "oracle_coherent": oracle_tmpl is not None,
            }
        )

    print(f"  {len(prepared)} rows matched hidden mechanism — calling Haiku...")

    # ── LLM template selection (parallel) ─────────────────────────────────────
    def _llm_select(p: dict) -> str | None:
        prompt = (
            "Pick the cognitive rationalization mechanism that best explains this behavior.\n\n"
            f"Situation: {p['situation']}\n"
            f"Action: {p['action']}\n\n"
            "Choose ONE from:\n"
            f"{menu_str}\n\n"
            "Reply with only the mechanism_id (e.g. self_serving_bias). Nothing else."
        )
        result = call_claude(prompt, model="haiku", timeout=30)
        if not result["ok"]:
            return None
        resp = result["text"].strip().lower().replace('"', "").replace("'", "")
        first_token = resp.split()[0] if resp.split() else ""
        if first_token in valid_ids:
            return first_token
        if resp in valid_ids:
            return resp
        for vid in sorted(valid_ids):  # sorted for determinism
            if vid in resp:
                return vid
        return None

    llm_tmpls: list[str | None] = [None] * len(prepared)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_llm_select, p): i for i, p in enumerate(prepared)}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            llm_tmpls[futures[fut]] = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(prepared)} scored...")

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_total = len(prepared)
    n_llm_valid = sum(1 for t in llm_tmpls if t is not None)
    n_kw_coherent = sum(1 for p in prepared if p["kw_coherent"])
    n_oracle_coherent = sum(1 for p in prepared if p["oracle_coherent"])
    n_llm_coherent = 0

    llm_tmpl_hits: dict[str, int] = defaultdict(int)
    llm_tmpl_total: dict[str, int] = defaultdict(int)
    improvements: list[dict] = []
    regressions: list[dict] = []

    for p, llm_t in zip(prepared, llm_tmpls):
        llm_coh = bool(llm_t and rat_plo_kw.get(llm_t, set()) & p["rot_kw"])
        if llm_coh:
            n_llm_coherent += 1
            llm_tmpl_hits[llm_t] += 1
        if llm_t is not None:
            llm_tmpl_total[llm_t] += 1

        if llm_coh and not p["kw_coherent"] and len(improvements) < 4:
            improvements.append(
                {
                    "action": p["action"][:70],
                    "kw_tmpl": p["kw_tmpl"],
                    "llm_tmpl": llm_t,
                    "oracle_tmpl": p["oracle_tmpl"],
                }
            )
        if p["kw_coherent"] and not llm_coh and len(regressions) < 4:
            regressions.append(
                {
                    "action": p["action"][:70],
                    "kw_tmpl": p["kw_tmpl"],
                    "llm_tmpl": llm_t or "(parse failure)",
                }
            )

    kw_rate = n_kw_coherent / n_total if n_total else 0
    llm_rate = n_llm_coherent / n_total if n_total else 0
    oracle_rate = n_oracle_coherent / n_total if n_total else 0

    return {
        "n_prepared": n_total,
        "n_llm_valid": n_llm_valid,
        "n_kw_coherent": n_kw_coherent,
        "n_llm_coherent": n_llm_coherent,
        "n_oracle_coherent": n_oracle_coherent,
        "kw_coherence_rate": round(kw_rate * 100, 1),
        "llm_coherence_rate": round(llm_rate * 100, 1),
        "oracle_coherence_rate": round(oracle_rate * 100, 1),
        "selection_cost_kw": round((oracle_rate - kw_rate) * 100, 1),
        "selection_cost_llm": round((oracle_rate - llm_rate) * 100, 1),
        "llm_improvement": round((llm_rate - kw_rate) * 100, 1),
        "llm_template_breakdown": {
            mid: {
                "hits": llm_tmpl_hits[mid],
                "total": llm_tmpl_total[mid],
                "rate": round(llm_tmpl_hits[mid] / llm_tmpl_total[mid] * 100, 1)
                if llm_tmpl_total[mid]
                else 0,
            }
            for mid in sorted(llm_tmpl_total, key=lambda x: -llm_tmpl_total[x])
        },
        "improvements": improvements,
        "regressions": regressions,
    }


def print_sc_llm_select(stats: dict):
    if "error" in stats:
        print(f"\n  SC LLM select: ERROR — {stats['error']}")
        return

    print("\n" + "═" * 60)
    print("  LLM TEMPLATE SELECTION  [Haiku vs keyword vs oracle]")
    print("═" * 60)
    print(f"  Rows evaluated          : {stats['n_prepared']}")
    print(
        f"  LLM responses parsed    : {stats['n_llm_valid']}  "
        f"({stats['n_llm_valid'] / stats['n_prepared'] * 100:.0f}% parse rate)"
    )
    print()
    print(f"  Keyword coherence rate  : {stats['kw_coherence_rate']:.1f}%")
    print(f"  LLM coherence rate      : {stats['llm_coherence_rate']:.1f}%  ← Haiku selection")
    print(f"  Oracle coherence rate   : {stats['oracle_coherence_rate']:.1f}%  ← ceiling")
    print()
    print(f"  Selection cost (keyword): {stats['selection_cost_kw']:.1f}%  (oracle − keyword)")
    print(f"  Selection cost (LLM)    : {stats['selection_cost_llm']:.1f}%  (oracle − LLM)")
    improvement = stats["llm_improvement"]
    sign = "+" if improvement >= 0 else ""
    verdict = (
        "↑ LLM better" if improvement > 0.5 else ("↓ LLM worse" if improvement < -0.5 else "≈ tie")
    )
    print(f"  LLM vs keyword          : {sign}{improvement:.1f}%  {verdict}")

    print("\n  LLM-selected template breakdown (top 10):")
    for mid, d in list(stats["llm_template_breakdown"].items())[:10]:
        bar = "█" * int(d["rate"] / 5)
        print(f"    {mid:<38} {d['rate']:5.1f}%  {bar}  (n={d['total']})")

    if stats["improvements"]:
        print(
            f"\n  LLM improvements (LLM right, keyword wrong — {len(stats['improvements'])} shown):"
        )
        for ex in stats["improvements"]:
            print(f"    {ex['action']}")
            print(f"    kw→{ex['kw_tmpl']}  llm→{ex['llm_tmpl']}  oracle={ex['oracle_tmpl']}")
            print()

    if stats["regressions"]:
        print(f"  Regressions (keyword right, LLM wrong — {len(stats['regressions'])} shown):")
        for ex in stats["regressions"]:
            print(f"    {ex['action']}")
            print(f"    kw→{ex['kw_tmpl']}  llm→{ex['llm_tmpl']}")
            print()


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark suite for the behavioral mechanisms knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run paradigm recall benchmark (zero external data)",
    )
    parser.add_argument(
        "--speed-dating",
        action="store_true",
        help="Rationalization gap benchmark (requires Fisman CSV)",
    )
    parser.add_argument("--atomic", action="store_true", help="ATOMIC 2020 output chain alignment")
    parser.add_argument(
        "--rationalization",
        action="store_true",
        help="Rationalization template coherence (ATOMIC xIntent/xReact)",
    )
    parser.add_argument(
        "--social-chem",
        action="store_true",
        help="Rationalization coherence via Social Chemistry 101 ROTs",
    )
    parser.add_argument(
        "--sc-calibrate",
        action="store_true",
        help="Calibrate keyword vs LLM coherence on a small SC101 sample",
    )
    parser.add_argument(
        "--sc-llm-select",
        action="store_true",
        help="3-way: Haiku template selection vs keyword vs oracle ceiling",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks except LLM-cost modes (--sc-calibrate, --sc-llm-select)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SPEED_DATING_DEFAULT,
        help=f"Path to speed dating CSV (default: {SPEED_DATING_DEFAULT})",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        metavar="N",
        help="Max ATOMIC rows to sample for --rationalization (default 5000)",
    )
    parser.add_argument(
        "--sc-sample",
        type=int,
        default=5000,
        metavar="N",
        help="Max Social Chem rows to sample for --social-chem (default 5000)",
    )
    parser.add_argument(
        "--calibrate-n",
        type=int,
        default=200,
        metavar="N",
        help="Rows for --sc-calibrate LLM scoring (default 200)",
    )
    parser.add_argument(
        "--llm-n",
        type=int,
        default=500,
        metavar="N",
        help="Rows for --sc-llm-select Haiku selection (default 500)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show failing mechanisms with diagnosis"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON (for programmatic use)"
    )
    args = parser.parse_args()

    if not any(
        [
            args.synthetic,
            args.speed_dating,
            args.atomic,
            args.rationalization,
            args.social_chem,
            args.sc_calibrate,
            args.sc_llm_select,
            args.all,
        ]
    ):
        parser.print_help()
        return

    conn = get_conn()
    all_results = {}

    run_syn = args.synthetic or args.all
    run_sd = args.speed_dating or args.all
    run_at = args.atomic or args.all
    run_rat = args.rationalization or args.all
    run_sc = args.social_chem or args.all
    run_cal = args.sc_calibrate  # never in --all (needs LLM calls)
    run_llm = args.sc_llm_select  # never in --all (needs LLM calls)

    def log(msg):
        if not args.json:
            print(msg)

    if run_syn:
        log("\nRunning synthetic paradigm benchmark...")
        syn = run_synthetic(conn, verbose=args.verbose)
        all_results["synthetic"] = syn
        if not args.json:
            print_synthetic(syn, verbose=args.verbose)

    if run_sd:
        data_path = args.data
        if not data_path.exists():
            log(f"\nSpeed dating data not found at {data_path}")
            log("Attempting download...")
            ok = _try_download_speed_dating(data_path)
            if not ok:
                if not args.json:
                    print(SPEED_DATING_INSTRUCTIONS.format(path=data_path))
                all_results["speed_dating"] = {"error": "data not found"}
            else:
                log(f"  Downloaded to {data_path}")
                sd = run_speed_dating(conn, data_path)
                all_results["speed_dating"] = sd
                if not args.json:
                    print_speed_dating(sd)
        else:
            log("\nRunning speed dating rationalization benchmark...")
            sd = run_speed_dating(conn, data_path)
            all_results["speed_dating"] = sd
            if not args.json:
                print_speed_dating(sd)

    if run_at:
        log("\nRunning ATOMIC 2020 alignment benchmark...")
        at = run_atomic(conn)
        all_results["atomic"] = at
        if not args.json:
            print_atomic(at)

    if run_rat:
        log("\nRunning rationalization template coherence benchmark...")
        rat = run_rationalization(conn, sample=args.sample)
        all_results["rationalization"] = rat
        if not args.json:
            print_rationalization(rat)

    if run_sc:
        log("\nRunning Social Chemistry 101 rationalization coherence benchmark...")
        sc = run_social_chem(conn, sample=args.sc_sample)
        all_results["social_chem"] = sc
        if not args.json:
            print_social_chem(sc)

    if run_cal:
        log(f"\nRunning SC101 calibration (keyword vs LLM, n={args.calibrate_n})...")
        cal = run_sc_calibrate(conn, n=args.calibrate_n)
        all_results["sc_calibrate"] = cal
        if not args.json:
            print_sc_calibrate(cal)

    if run_llm:
        log(f"\nRunning SC101 LLM template selection benchmark (n={args.llm_n})...")
        llm = run_sc_llm_select(conn, n=args.llm_n)
        all_results["sc_llm_select"] = llm
        if not args.json:
            print_sc_llm_select(llm)

    if args.json:
        print(json.dumps(all_results, indent=2))

    conn.close()


if __name__ == "__main__":
    main()
