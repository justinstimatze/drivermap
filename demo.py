#!/usr/bin/env python3
"""
demo.py — Five scenarios showing drivermap end to end.

  profile + situation → ranked active mechanisms → verbalized rationalization

No server required. Calls Claude CLI for the final generation step.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extract import call_claude
from mcp_server import _score_mechanisms, get_conn, verbalize_motivation

# ─── Scenarios ────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "title": "The Passed-Over Colleague",
        "setup": (
            "Someone was just passed over for a promotion. They are now explaining\n"
            "  to a teammate why the newly promoted colleague's approach will probably fail."
        ),
        "profile": {
            "big_five_N": "+",  # threat-sensitive
            "social_dominance_orientation": "+",  # hierarchy matters to them
            "power_state": "-",  # just lost relative standing
            "threat_appraisal": "+",  # currently feeling threatened
        },
        "situation": ["power_differential", "social_visibility", "conflict_present"],
        "action": (
            "explained to a teammate why the newly promoted colleague's "
            "approach to the project would probably fail"
        ),
    },
    {
        "title": "The Founder in Denial",
        "setup": (
            "A startup founder with six months of declining metrics just sent an\n"
            "  investor update calling it 'healthy consolidation' and 'part of the journey.'"
        ),
        "profile": {
            "big_five_C": "+",  # committed, goal-persistent
            "need_for_closure": "+",  # resists uncertainty
            "threat_appraisal": "+",  # company identity under threat
        },
        "situation": ["stakes", "prior_commitment", "social_visibility"],
        "action": (
            "reframed six months of declining metrics as 'healthy consolidation' "
            "in an investor update"
        ),
    },
    {
        "title": "One More Chance",
        "setup": (
            "After the third major conflict this month, someone decides to give\n"
            "  a four-year relationship 'one more chance.'"
        ),
        "profile": {
            "attachment_anxious": "+",  # fear of abandonment
            "big_five_N": "+",  # emotionally reactive
            "fatigue_depletion": "+",  # depleted, reduced deliberate reasoning
        },
        "situation": ["prior_commitment", "ambiguity", "outcome_reversibility"],
        "action": (
            "decided to give the relationship another chance "
            "after the third major conflict this month"
        ),
    },
    {
        "title": "Logan Roy — the morning after",
        "setup": (
            "Logan Roy. Media patriarch. The night before, he was hospitalized.\n"
            "  This morning he cancelled the succession process and told the board\n"
            "  his health was 'absolutely fine.'\n"
            "\n"
            "  Trait profile (stable):\n"
            "    social_dominance_orientation:+  hierarchy is the only real structure\n"
            "    dark_triad_narcissism:+          Waystar is not a company he built, it is what he is\n"
            "    dark_triad_machiavellianism:+    everyone in the room is a piece to be moved\n"
            "    attachment_avoidant:+            closeness = leverage = threat\n"
            "    hexaco_H:-                       rules are for people he controls\n"
            "    bis_sensitivity:-               threat → attack, not freeze\n"
            "    need_for_closure:+               ambiguity about who leads is intolerable\n"
            "    big_five_A:-                     does not accommodate, ever\n"
            "\n"
            "  State (this morning, post-hospitalization):\n"
            "    power_state:-                   first time in decades he has looked fallible\n"
            "    threat_appraisal:+              his children's concern is indistinguishable from ambition\n"
            "    affective_valence:-             angry, humiliated\n"
            "    affective_arousal:+             high activation — dominant response amplified"
        ),
        "profile": {
            # Traits
            "social_dominance_orientation": "+",  # hierarchy is the only real structure
            "dark_triad_narcissism": "+",  # Waystar is not built by him, it is him
            "dark_triad_machiavellianism": "+",  # everyone is a piece to be moved
            "attachment_avoidant": "+",  # closeness = leverage = threat
            "hexaco_H": "-",  # rules are for people he controls
            "bis_sensitivity": "-",  # threat → attack, not withdraw
            "need_for_closure": "+",  # ambiguity about succession is intolerable
            "big_five_A": "-",  # does not accommodate, ever
            # States — this specific morning
            "power_state": "-",  # first time he has looked fallible
            "threat_appraisal": "+",  # concern and ambition are the same signal
            "affective_valence": "-",  # angry, humiliated
            "affective_arousal": "+",  # high activation amplifies dominant response
        },
        "situation": [
            "stakes",  # company, legacy, identity — all the same thing to him
            "power_differential",  # temporarily inverted; everyone saw him weak
            "surveillance",  # board, family, press all watching for signs
            "prior_commitment",  # he built this; it cannot exist without him
            "social_visibility",
        ],
        "action": (
            "cancelled the board-mandated succession process and told his family "
            "the company was not going anywhere without him"
        ),
        "framing": "dialogue",
    },
    {
        "title": "The Thread",
        "setup": (
            "A new async job queue library is posted in a technical community.\n"
            "  Within four minutes, an established regular replies: 'Isn't this\n"
            "  just Celery with a thinner API?' The author explains the\n"
            "  architectural differences and asks which specific part they're\n"
            "  referring to. The regular doesn't answer. Instead they edit their\n"
            "  comment to add: 'The README doesn't even mention prior art.\n"
            "  That alone tells you something.'"
        ),
        "profile": {
            "social_dominance_orientation": "+",  # status through technical authority
            "dark_triad_narcissism": "+",  # needs to be seen as the most informed person in the room
            "need_for_closure": "+",  # verdict was in before the tab finished loading
            "in_group_salience": "+",  # community gatekeeper identity active
            "power_state": "+",  # high karma = legitimate standing to speak
            "big_five_A": "-",  # not here to be liked
        },
        "situation": ["social_visibility", "out_group_salience", "conflict_present"],
        "action": (
            "dismissed a new open-source library as 'just Celery with a thinner API' "
            "four minutes after it was posted, without opening the repository, "
            "then ignored the author's direct question and moved the goalposts to "
            "a README complaint when pressed"
        ),
    },
]

# ─── Situationist comparison ──────────────────────────────────────────────────
# Same person, three situations → different mechanisms → different behavior.
# This is the core thesis: B = f(P, E).

SITUATIONIST_PROFILE = {
    "title": "Same Person, Three Rooms",
    "setup": (
        "A department head. Politically shrewd, low agreeableness, high\n"
        "  dominance orientation. Same person in three settings — watch the\n"
        "  mechanisms shift."
    ),
    "profile": {
        "dark_triad_machiavellianism": "+",
        "social_dominance_orientation": "+",
        "big_five_A": "-",
        "threat_appraisal": "+",
    },
    "situations": [
        {
            "label": "At the team lunch (power holder, group context)",
            "features": [
                "group_context",
                "social_visibility",
                "power_differential",
                "power_holder",
            ],
        },
        {
            "label": "At the board review (under authority)",
            "features": ["power_differential", "power_low", "social_visibility", "stakes"],
        },
        {
            "label": "Alone in the office (no audience)",
            "features": ["ambiguity", "prior_commitment"],
        },
    ],
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

W = 70


def rule(char="─"):
    print("  " + char * (W - 2))


def fmt_profile(profile: dict) -> str:
    parts = []
    for dim, val in profile.items():
        sign = "+" if val == "+" else "−"
        parts.append(f"{sign}{dim}")
    return "  ".join(parts)


def best_output(r: dict) -> str | None:
    """Return best mechanism description: prefer narrative_outputs, fall back to PLO."""
    for key in ("narrative_outputs", "plain_language_outputs"):
        items = r.get(key)
        if not items or not isinstance(items, list):
            continue
        candidates = [p for p in items if isinstance(p, str) and len(p) > 15]
        if candidates:
            return max(candidates, key=len)
    return None


def parse_lines(text: str) -> list[str]:
    """Parse Claude output — handles JSON arrays and markdown-wrapped JSON."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(s) for s in parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: split on newlines, strip bullets
    lines = []
    for line in text.splitlines():
        line = line.strip().lstrip("•-*").strip().strip('"').strip()
        if line:
            lines.append(line)
    return lines


# ─── Main renderer ────────────────────────────────────────────────────────────


def run_scenario(scenario: dict, conn):
    print()
    print("  " + "═" * (W - 2))
    print(f"  {scenario['title']}")
    print("  " + "═" * (W - 2))
    print(f"\n  {scenario['setup']}\n")

    results = _score_mechanisms(conn, scenario["profile"], scenario["situation"], top_n=8)

    if not results:
        print("  (no mechanisms scored above zero)\n")
        return

    print(f"  PROFILE:   {fmt_profile(scenario['profile'])}")
    print(f"  SITUATION: {', '.join(scenario['situation'])}\n")
    rule()
    print("  PREDICTED MECHANISMS\n")

    for i, r in enumerate(results[:5]):
        amp = [m["dimension"] for m in r["person_matches"] if m.get("effect") == "amplifies"]
        feats = [
            m["feature"] for m in r["situation_matches"] if "dampens" not in m.get("effect", "")
        ]
        tags = "  ".join([f"+{d}" for d in amp[:3]] + feats[:2])

        print(f"  {i + 1}.  {r['name']:<40} {r['score']:>4.1f}  [{r['domain']}]")
        if tags:
            print(f"       {tags}")
        desc = best_output(r)
        if desc:
            print(f'       → "{desc}"')
        print()

    rule()
    top = results[0]
    print("\n  VERBALIZATION — what they say out loud\n")
    print(f"  Hidden driver : {top['name']}")
    print(f"  Action        : {scenario['action']}\n")

    v = verbalize_motivation(
        hidden_mechanism_id=top["id"],
        action_description=scenario["action"],
        profile=scenario["profile"],
        situation=scenario["situation"],
        framing=scenario.get("framing", "first_person"),
    )

    if "error" in v:
        print(f"  Error: {v['error']}\n")
        return

    rat = v.get("rationalization_template", {})
    if rat:
        print(f"  Rationalization template: {rat.get('name', '?')}\n")

    prompt = v.get("verbalization_prompt", "")
    result = call_claude(prompt)

    if result["ok"]:
        lines = parse_lines(result["text"])
        for line in lines:
            print(f'  • "{line}"')
    else:
        print("  [Claude unavailable — verbalization_prompt ready for external use]")

    print()


def run_situationist(spec: dict, conn):
    """Same profile, multiple situations — show mechanism rankings shift."""
    print()
    print("  " + "═" * (W - 2))
    print(f"  {spec['title']}")
    print("  " + "═" * (W - 2))
    print(f"\n  {spec['setup']}\n")
    print(f"  PROFILE: {fmt_profile(spec['profile'])}\n")
    rule()
    print("  B = f(P, E) — same person, different environments:\n")

    for sit in spec["situations"]:
        results = _score_mechanisms(conn, spec["profile"], sit["features"], top_n=5)
        print(f"  ▸ {sit['label']}")
        for i, r in enumerate(results[:3]):
            desc = best_output(r)
            tag = f'  "{desc}"' if desc else ""
            print(f"    {i + 1}. {r['name']:<38} {r['score']:>5.2f}{tag}")
        print()

    rule()
    print()


def main():
    conn = get_conn()
    try:
        run_situationist(SITUATIONIST_PROFILE, conn)
        for scenario in SCENARIOS:
            run_scenario(scenario, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
