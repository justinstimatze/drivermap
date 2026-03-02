#!/usr/bin/env python3
"""
mcp_server.py — MCP server for the behavioral mechanisms knowledge base.

Tools:
  predict_mechanisms       — profile + situation → ranked active mechanisms
  get_profile_questions    — situation-weighted interview questions (iterative)
  analyze_description_bias — detect likely distortions in a self-report description
  save_profile / load_profile / list_profiles — named profile persistence
  get_mechanism            — full mechanism detail with moderators + activators
  search_mechanisms        — full-text search
  list_dimensions          — vocabulary reference (traits, states, situation features)

Usage (register in ~/.claude/settings.json):
  "drivermap": {
    "command": "python3",
    "args": ["/path/to/drivermap/mcp_server.py"]
  }
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).parent
DB_PATH = ROOT / "db" / "mechanisms.sqlite"
PROFILES_PATH = ROOT / "profiles.json"

mcp = FastMCP("drivermap")


# ─── DB ───────────────────────────────────────────────────────────────────────


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


@contextmanager
def db():
    conn = get_conn()
    try:
        yield conn
    finally:
        conn.close()


# ─── Scoring ──────────────────────────────────────────────────────────────────

STRENGTH_WEIGHT = {"strong": 1.5, "moderate": 1.0, "weak": 0.5}
SITUATION_MULTIPLIER = 0.5


def _score_mechanisms(conn, profile: dict, situation: list[str], top_n: int = 15) -> list[dict]:
    """
    Score all mechanisms against a profile and situation.

    profile:   {dimension: '+' | '-'}  — '+' = high/present, '-' = low/absent
    situation: [feature, ...]          — list of active situation features
    """
    mechs = conn.execute(
        "SELECT id, name, domain, description, summary, "
        "behavioral_outputs, outputs, triggers, effect_size, "
        "replication, replication_status, accuracy_score, "
        "plain_language_outputs, narrative_outputs "
        "FROM mechanisms"
    ).fetchall()

    situation_set = set(situation)
    results = []

    for mech in mechs:
        mid = mech["id"]
        person_score = 0.0
        situation_score = 0.0
        person_matches = []
        situation_matches = []
        excluded = False
        # ── Person moderators ──
        pms = conn.execute(
            "SELECT dimension, direction, strength, note "
            "FROM person_moderators WHERE mechanism_id=?",
            (mid,),
        ).fetchall()

        for pm in pms:
            dim = pm["dimension"]
            if dim not in profile:
                continue
            w = STRENGTH_WEIGHT.get(pm["strength"] or "moderate", 1.0)
            user_dir = profile[dim]  # '+' or '-'
            mech_dir = pm["direction"]

            if mech_dir == "mixed":
                person_score += 0.25
                person_matches.append(
                    {"dimension": dim, "direction": "mixed", "weight": 0.25, "note": pm["note"]}
                )
            elif user_dir == mech_dir:
                person_score += w
                person_matches.append(
                    {
                        "dimension": dim,
                        "direction": mech_dir,
                        "weight": w,
                        "effect": "amplifies",
                        "note": pm["note"],
                    }
                )
            else:
                person_score -= w * 0.5
                person_matches.append(
                    {
                        "dimension": dim,
                        "direction": mech_dir,
                        "weight": -w * 0.5,
                        "effect": "dampens",
                        "note": pm["note"],
                    }
                )

        # ── Situation activators ──
        sas = conn.execute(
            "SELECT feature, effect, note FROM situation_activators WHERE mechanism_id=?", (mid,)
        ).fetchall()

        for sa in sas:
            feat = sa["feature"]
            effect = sa["effect"]
            w = 1.0  # situation_activators has no strength column; use moderate default

            if effect == "required":
                if feat in situation_set:
                    situation_score += w * 2.0
                    situation_matches.append(
                        {
                            "feature": feat,
                            "effect": "required+present",
                            "weight": w * 2.0,
                            "note": sa["note"],
                        }
                    )
                else:
                    excluded = True
                    break
            elif feat in situation_set:
                if effect == "activates":
                    situation_score += w * 2.0
                    situation_matches.append(
                        {
                            "feature": feat,
                            "effect": "activates",
                            "weight": w * 2.0,
                            "note": sa["note"],
                        }
                    )
                elif effect == "amplifies":
                    situation_score += w
                    situation_matches.append(
                        {"feature": feat, "effect": "amplifies", "weight": w, "note": sa["note"]}
                    )
                elif effect == "dampens":
                    situation_score -= w
                    situation_matches.append(
                        {"feature": feat, "effect": "dampens", "weight": -w, "note": sa["note"]}
                    )

        if excluded:
            continue

        if person_score <= 0:
            continue

        total = person_score * (1 + situation_score * SITUATION_MULTIPLIER)
        if total <= 0:
            continue

        outputs = mech["behavioral_outputs"] or mech["outputs"]
        try:
            outputs_parsed = json.loads(outputs) if outputs else None
        except (json.JSONDecodeError, TypeError):
            outputs_parsed = outputs

        plo = mech["plain_language_outputs"]
        try:
            plo_parsed = json.loads(plo) if plo else None
        except (json.JSONDecodeError, TypeError):
            plo_parsed = plo

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
                "behavioral_outputs": outputs_parsed,
                "plain_language_outputs": plo_parsed,
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

    # ── Tension detection ──
    # Flag pairs in top results where one dampens what another activates
    top = results[:top_n]
    _annotate_tensions(top)

    return top


def _annotate_tensions(results: list[dict]):
    """
    Find mechanism pairs where the same situation feature activates one
    and dampens the other, or where both fire strongly in opposite person directions.
    Annotates results in place with a 'tensions' list.
    """
    for r in results:
        r["tensions"] = []

    for i, a in enumerate(results):
        for b in results[i + 1 :]:
            # Same domain, opposite person_score signs
            if (
                a["domain"] == b["domain"]
                and a["person_score"] * b["person_score"] < 0
                and abs(a["person_score"]) > 0.5
                and abs(b["person_score"]) > 0.5
            ):
                tension = {
                    "with": b["id"],
                    "type": "person_opposition",
                    "note": f"Profile amplifies {a['id']} but dampens {b['id']} (same domain)",
                }
                a["tensions"].append(tension)

            # One activates, other dampens for same feature
            a_feats = {m["feature"]: m["effect"] for m in a["situation_matches"]}
            b_feats = {m["feature"]: m["effect"] for m in b["situation_matches"]}
            for feat in set(a_feats) & set(b_feats):
                ae = a_feats[feat]
                be = b_feats[feat]
                if ("activates" in ae or "amplifies" in ae) and "dampens" in be:
                    a["tensions"].append(
                        {
                            "with": b["id"],
                            "type": "situation_opposition",
                            "feature": feat,
                            "note": f"{feat} activates {a['id']} but dampens {b['id']}",
                        }
                    )


# ─── Dimension leverage ───────────────────────────────────────────────────────


def _dimension_leverage(conn, situation_features: list[str]) -> dict[str, float]:
    """
    For a given set of active situation features, compute how much predictive
    leverage each person dimension has (how many mechanisms it moderates that
    are also activated by this situation).
    """
    if not situation_features:
        # No situation — use overall coverage
        rows = conn.execute("""
            SELECT dimension,
                   COUNT(*) as n,
                   AVG(CASE strength WHEN 'strong' THEN 1.5
                                     WHEN 'moderate' THEN 1.0
                                     ELSE 0.5 END) as avg_w
            FROM person_moderators
            GROUP BY dimension
        """).fetchall()
    else:
        placeholders = ",".join("?" for _ in situation_features)
        rows = conn.execute(
            f"""
            SELECT pm.dimension,
                   COUNT(DISTINCT pm.mechanism_id) as n,
                   AVG(CASE pm.strength WHEN 'strong' THEN 1.5
                                        WHEN 'moderate' THEN 1.0
                                        ELSE 0.5 END) as avg_w
            FROM person_moderators pm
            JOIN situation_activators sa ON pm.mechanism_id = sa.mechanism_id
            WHERE sa.feature IN ({placeholders})
              AND sa.effect IN ('activates', 'amplifies', 'required')
            GROUP BY pm.dimension
        """,
            situation_features,
        ).fetchall()

    return {r["dimension"]: round(r["n"] * r["avg_w"], 2) for r in rows}


def _coverage(leverage: dict, known: dict) -> dict:
    total = sum(leverage.values())
    remaining = sum(v for k, v in leverage.items() if k not in known)
    pct = (1 - remaining / total) if total > 0 else 1.0
    return {
        "dimensions_established": len(known),
        "dimensions_remaining": len([k for k in leverage if k not in known]),
        "remaining_leverage": round(remaining, 2),
        "total_leverage": round(total, 2),
        "coverage_pct": round(pct * 100, 1),
        "ready_to_predict": remaining < total * 0.30,
    }


# ─── Question bank ────────────────────────────────────────────────────────────

QUESTION_BANK = {
    # ── Trait dimensions ──
    "big_five_N": {
        "third_person": "When things go badly for this person, how long does it tend to affect them? Do they recover quickly or carry it for days?",
        "first_person": "When something goes wrong for you, how long does it typically affect your mood and focus?",
        "follow_ups": [
            "Do they catastrophize, or tend to assume things will work out?",
            "How do they handle being criticized in front of others?",
        ],
        "markers": "Rumination, irritability, worry, emotional reactivity, slow recovery from setbacks",
    },
    "big_five_C": {
        "third_person": "Is this person someone who prepares, follows through, and keeps track of obligations — or more spontaneous and flexible about commitments?",
        "first_person": "Would people who know you well describe you as reliable and organized, or more spontaneous?",
        "follow_ups": [
            "Do they tend to finish what they start?",
            "How do they respond when plans change unexpectedly?",
        ],
        "markers": "Planning, follow-through, punctuality, self-discipline, orderliness",
    },
    "big_five_E": {
        "third_person": "Does this person gain energy from social situations, or does being around people drain them?",
        "first_person": "After a long social event, do you feel energized or like you need to decompress alone?",
        "follow_ups": [
            "Do they seek out attention and conversation, or tend to hang back?",
            "Are they comfortable being the center of attention?",
        ],
        "markers": "Talkativeness, social initiation, expressiveness, excitement-seeking, positive affect in groups",
    },
    "big_five_A": {
        "third_person": "When there's a conflict, does this person prioritize harmony and others' needs, or are they comfortable asserting themselves even if it causes friction?",
        "first_person": "When your interests conflict with someone else's, what's your first instinct?",
        "follow_ups": [
            "Do they have trouble saying no?",
            "How do they respond to someone they perceive as acting unfairly?",
        ],
        "markers": "Cooperativeness, empathy, conflict avoidance, willingness to compromise, trust in others",
    },
    "big_five_O": {
        "third_person": "Does this person actively seek out new ideas, experiences, or ways of thinking — or do they prefer what's familiar and proven?",
        "first_person": "Do you find yourself drawn to unfamiliar ideas and experiences, or do you prefer staying in your domain?",
        "follow_ups": [
            "How do they respond to having their assumptions challenged?",
            "Are they interested in abstract or theoretical questions?",
        ],
        "markers": "Curiosity, aesthetic sensitivity, intellectual engagement, novelty-seeking, openness to alternative views",
    },
    "dark_triad_narcissism": {
        "third_person": "Does this person have a strong sense of being special or exceptional — and a need for others to recognize it?",
        "first_person": "Do you tend to feel you see things more clearly than most people around you?",
        "follow_ups": [
            "How do they react to criticism or being overlooked?",
            "Do they talk about themselves and their achievements a lot?",
        ],
        "markers": "Entitlement, need for admiration, self-aggrandizement, sensitivity to status slights, grandiose self-narrative",
    },
    "dark_triad_machiavellianism": {
        "third_person": "Does this person think several moves ahead in social situations — who's useful, who's a threat, how to position themselves?",
        "first_person": "Do you find yourself naturally analyzing what others want and how you might use that?",
        "follow_ups": [
            "Are they willing to be strategically dishonest if it serves their goals?",
            "Do they compartmentalize relationships based on usefulness?",
        ],
        "markers": "Strategic thinking, instrumental view of relationships, willingness to deceive, long-term manipulation, charm as a tool",
    },
    "dark_triad_psychopathy": {
        "third_person": "Does this person seem genuinely affected when others are hurt or distressed, or does it not seem to register?",
        "first_person": "When someone around you is in pain, do you feel it viscerally, or do you process it more cognitively?",
        "follow_ups": [
            "Do they take risks without apparent anxiety?",
            "Do they seem to feel remorse after harmful actions?",
        ],
        "markers": "Absence of guilt/remorse, low empathy, fearlessness, thrill-seeking, callousness",
    },
    "attachment_anxious": {
        "third_person": "In close relationships, does this person worry about whether they're loved enough or whether the other person will leave?",
        "first_person": "In close relationships, do you find yourself anxious about whether the other person really cares, or reading into their behavior for signs?",
        "follow_ups": [
            "Do they become clingy or demanding when they feel the relationship is threatened?",
            "How do they handle periods of low contact with important people?",
        ],
        "markers": "Hypervigilance to relationship cues, fear of abandonment, clinginess, reassurance-seeking",
    },
    "attachment_avoidant": {
        "third_person": "Does this person pull back from intimacy or vulnerability — preferring to stay self-reliant even in close relationships?",
        "first_person": "Do you tend to feel uncomfortable depending on others, or prefer to handle things yourself?",
        "follow_ups": [
            "Do they shut down or withdraw when conflicts get emotional?",
            "Are they comfortable expressing need or vulnerability?",
        ],
        "markers": "Emotional distance, discomfort with dependence, dismissiveness, self-reliance as identity",
    },
    "bis_sensitivity": {
        "third_person": "Is this person particularly sensitive to potential threats, punishments, or things that could go wrong — does anxiety motivate their behavior a lot?",
        "first_person": "Do you often find yourself thinking about what could go wrong, or feeling inhibited by potential negative consequences?",
        "follow_ups": [
            "Do they tend to freeze or avoid in novel or uncertain situations?",
            "Are they prone to guilt and worry?",
        ],
        "markers": "Anxiety-driven inhibition, avoidance of punishment, freeze response, vigilance to threat cues",
    },
    "bas_sensitivity": {
        "third_person": "Is this person strongly motivated by rewards, opportunities, and the excitement of potential gains — do they move toward things energetically?",
        "first_person": "Do you find yourself strongly drawn toward exciting opportunities, or easily activated by the prospect of reward?",
        "follow_ups": [
            "Do they take risks when there's something to gain?",
            "Do they have trouble stopping a rewarding behavior once started?",
        ],
        "markers": "Reward-seeking, impulsivity, excitement about opportunities, difficulty delaying gratification",
    },
    "need_for_cognition": {
        "third_person": "Does this person enjoy thinking through complex problems, or do they prefer quick, practical answers?",
        "first_person": "Do you enjoy puzzles, arguments, or sitting with a complex question — or do you prefer to reach a conclusion quickly?",
        "follow_ups": [
            "Do they engage with ideas for their own sake?",
            "How do they respond to ambiguity in information?",
        ],
        "markers": "Enjoys analysis, engages with complex problems voluntarily, resists oversimplification",
    },
    "need_for_closure": {
        "third_person": "Does this person get uncomfortable with uncertainty and ambiguity — do they push to reach a clear answer even under incomplete information?",
        "first_person": "How comfortable are you sitting with an unresolved question for a long time?",
        "follow_ups": [
            "Do they make decisions quickly and stick to them?",
            "Do they get frustrated by discussions that don't reach a conclusion?",
        ],
        "markers": "Decisiveness, discomfort with ambiguity, premature closure, rule-following, intolerance of uncertainty",
    },
    "disgust_sensitivity": {
        "third_person": "Is this person particularly sensitive to things that feel impure, contaminating, or morally repulsive — beyond just physical disgust?",
        "first_person": "Do you find certain behaviors or ideas viscerally repulsive in a way that goes beyond just thinking they're wrong?",
        "follow_ups": [
            "Does physical or moral messiness bother them more than most people?",
            "Do they have strong reactions to perceived violations of purity or tradition?",
        ],
        "markers": "Heightened moral disgust, sensitivity to contamination/purity violations, reactive to out-group norm violations",
    },
    "sensory_processing_sensitivity": {
        "third_person": "Is this person particularly affected by sensory or environmental stimulation — noise, crowds, others' moods — in a way that becomes overwhelming?",
        "first_person": "Do you find yourself deeply affected by environmental inputs — noise, others' emotions, aesthetic details — more than most people seem to be?",
        "follow_ups": [
            "Do they need more recovery time after stimulating environments?",
            "Are they especially reactive to others' emotional states?",
        ],
        "markers": "Overwhelm in stimulating environments, emotional contagion susceptibility, deep processing, needs recovery time",
    },
    "alexithymia": {
        "third_person": "Does this person have difficulty identifying or describing what they're feeling, or seem to process emotions primarily intellectually?",
        "first_person": "Do you find it hard to identify what you're feeling in a given moment, or to put emotions into words?",
        "follow_ups": [
            "Do they tend to describe events factually rather than emotionally?",
            "Do they seem puzzled by others' emotional reactions?",
        ],
        "markers": "Difficulty labeling emotions, intellectualization, pragmatic communication style, reduced emotional expressivity",
    },
    "social_dominance_orientation": {
        "third_person": "Does this person believe social hierarchies are natural and appropriate — that some groups or people are just better suited to lead or have more?",
        "first_person": "Do you tend to see social hierarchies as natural and legitimate, or does inequality bother you?",
        "follow_ups": [
            "Do they strongly prefer being in higher-status positions?",
            "Are they comfortable with systems that produce unequal outcomes?",
        ],
        "markers": "Hierarchy endorsement, outgroup derogation, dominance motivation, opposition to equality",
    },
    "hexaco_H": {
        "third_person": "Is this person genuinely motivated by fairness and honesty even when dishonesty would go undetected — or do they take advantage of opportunities when no one's watching?",
        "first_person": "Would you act the same way if you knew no one would ever find out?",
        "follow_ups": [
            "Do they avoid boasting or claiming credit they don't deserve?",
            "Are they willing to lose something to preserve fairness?",
        ],
        "markers": "Honesty under no surveillance, non-exploitativeness, modesty, genuine fairness motivation",
    },
    # ── State dimensions ──
    "cognitive_load": {
        "third_person": "Right now, is this person dealing with a lot of competing demands on their attention — mentally stretched thin?",
        "first_person": "Are you currently mentally overloaded — juggling a lot of things, feeling like your bandwidth is limited?",
        "follow_ups": [
            "Have they been sleeping poorly or under sustained stress lately?",
            "Are there multiple active problems they're managing simultaneously?",
        ],
        "markers": "Reduced deliberate reasoning, increased reliance on heuristics, irritability, decision fatigue",
    },
    "affective_valence": {
        "third_person": "Is this person in a positive or negative emotional state right now — generally up or down?",
        "first_person": "How would you describe your baseline mood going into this situation — positive, neutral, or negative?",
        "follow_ups": [
            "Has something recently happened that significantly shifted their mood?",
            "Is the negative/positive state specific to this situation or general?",
        ],
        "markers": "Colors interpretation of ambiguous cues; negative valence amplifies threat-detection, positive increases risk-tolerance",
    },
    "affective_arousal": {
        "third_person": "Is this person in a heightened, activated state — energized, agitated, or emotionally intense — or calm and low-key?",
        "first_person": "How activated or aroused are you right now — calm and measured, or keyed up?",
        "follow_ups": [
            "Did something just happen that elevated their arousal?",
            "Are they someone who tends to run hot emotionally in general?",
        ],
        "markers": "High arousal narrows cognition, amplifies dominant responses, reduces nuanced processing",
    },
    "resource_scarcity_state": {
        "third_person": "Is this person currently experiencing resource scarcity — financial pressure, time pressure, or feeling like they don't have enough of something important?",
        "first_person": "Do you currently feel like you're operating under scarcity — not enough money, time, or something else that feels essential?",
        "follow_ups": [
            "Is this scarcity chronic or recent?",
            "Does it feel like it's dominating their mental bandwidth?",
        ],
        "markers": "Cognitive tunneling, reduced future orientation, increased present bias, heightened loss sensitivity",
    },
    "power_state": {
        "third_person": "Does this person currently feel powerful — in control, with options and resources — or powerless, dependent on others' decisions?",
        "first_person": "In this situation, do you feel like you have power and options, or like you're dependent on others?",
        "follow_ups": [
            "Have they recently gained or lost power or status?",
            "Do they feel they could walk away from this situation if they wanted to?",
        ],
        "markers": "High power → approach behavior, reduced inhibition; low power → inhibition, hypervigilance to powerful others",
    },
    "threat_appraisal": {
        "third_person": "Does this person perceive the current situation as genuinely threatening to something they value?",
        "first_person": "Does this situation feel like a genuine threat to something important to you?",
        "follow_ups": [
            "What specifically feels at risk — status, safety, relationship, identity?",
            "Is the threat concrete or more diffuse/anticipated?",
        ],
        "markers": "Activates fight/flight/freeze, status threat response, hypervigilance, motivated reasoning",
    },
    "in_group_salience": {
        "third_person": "Is this person's group identity particularly activated right now — does it feel like an us-versus-them situation?",
        "first_person": "Does this situation feel like it involves your group identity — like your tribe is somehow at stake?",
        "follow_ups": [
            "Is there an out-group present or implied?",
            "Are there markers that make group membership especially visible?",
        ],
        "markers": "Amplifies in-group favoritism, out-group hostility, conformity to group norms, coalition behavior",
    },
    "relationship_history": {
        "third_person": "What's the history between this person and the others involved — is there accumulated debt, trust, betrayal, or obligation?",
        "first_person": "What's your history with the people involved — is there prior debt, trust, or unresolved tension?",
        "follow_ups": [
            "Has there been a significant prior exchange — favor given, harm done, promise made?",
            "Is this a repeated interaction or relatively new relationship?",
        ],
        "markers": "Activates reciprocity, forgiveness/revenge dynamics, trust or distrust heuristics",
    },
    "fatigue_depletion": {
        "third_person": "Is this person currently fatigued or depleted — physically tired, or emotionally drained from prior demands?",
        "first_person": "Are you running on empty right now — physically or emotionally drained?",
        "follow_ups": [
            "Have they been under sustained pressure without recovery?",
            "When did they last have meaningful rest?",
        ],
        "markers": "Impairs self-regulation, increases default/automatic responding, reduces inhibitory control",
    },
}


# ─── Posthoc rationalization bias probes ─────────────────────────────────────
# Used by analyze_description_bias to structure the self-audit interview

BIAS_PROBES = [
    {
        "mechanism": "fundamental_attribution_error",
        "signal_patterns": [
            "always",
            "never",
            "he's just",
            "she's the type",
            "they always",
            "that's just how",
            "typical",
            "he never",
            "she never",
            "they never",
            "he's the kind",
            "she's the kind",
            "that's who they are",
            "he always does",
            "she always does",
            "predictably",
        ],
        "description": "Other party's behavior attributed to character rather than situation",
        "probe": "What might the other person have been under pressure about at the time?",
        "what_to_listen_for": "Whether the narrator can generate a situational explanation for the other's behavior",
    },
    {
        "mechanism": "self_serving_bias",
        "signal_patterns": [
            "I worked hard",
            "my effort",
            "wasn't my fault",
            "they didn't listen",
            "wasn't appreciated",
            "more than anyone",
            "more than everyone",
            "put in more",
            "way more",
            "no one recognized",
            "ignored my",
            "dismissed my",
            "my idea",
            "my contribution",
            "never get credit",
            "don't get credit",
            "should have been",
            "deserved",
            "i've been",
            "i had done",
            "i was the one",
            "nobody else",
        ],
        "description": "Narrator's contribution minimized or absent; focus on effort and others' failures to recognize it",
        "probe": "What was your goal going into this, and what did you do to pursue it?",
        "what_to_listen_for": "Whether actions taken are acknowledged and how they're framed",
    },
    {
        "mechanism": "naive_realism",
        "signal_patterns": [
            "clearly",
            "obviously",
            "anyone could see",
            "it was obvious",
            "they refused to see",
            "just doesn't",
            "just won't",
            "can't see",
            "refuses to acknowledge",
            "it's clear",
            "plain to see",
            "objectively",
            "anyone would",
            "any reasonable",
            "it's not complicated",
            "simple fact",
            "undeniable",
        ],
        "description": "High confidence in own perception; others' disagreement attributed to bias or bad faith",
        "probe": "What would someone who sided with the other party say happened?",
        "what_to_listen_for": "Whether narrator can genuinely inhabit the opposing frame",
    },
    {
        "mechanism": "motivated_reasoning",
        "signal_patterns": [
            "I knew it",
            "I was right",
            "it confirmed",
            "just as I thought",
            "proves my point",
            "exactly what i expected",
            "knew this would",
            "told you so",
            "saw it coming",
            "knew all along",
        ],
        "description": "Narrative assembled to support a conclusion already held",
        "probe": "What information made you uncomfortable, or didn't fit the story?",
        "what_to_listen_for": "Whether contradicting evidence is acknowledged or explained away",
    },
    {
        "mechanism": "positive_illusions",
        "signal_patterns": [
            "I did my best",
            "I handled it well",
            "I stayed calm",
            "I was professional",
            "i tried to",
            "i made every effort",
            "i was fair",
            "i was reasonable",
            "i was just trying to",
            "i only wanted",
            "my intentions were",
            "i was being",
            "i kept my cool",
            "i was the bigger person",
        ],
        "description": "Narrator's role described charitably; others' roles described critically",
        "probe": "What would you have done differently if you could replay it?",
        "what_to_listen_for": "Whether genuine self-criticism is available",
    },
    {
        "mechanism": "choice_blindness",
        "signal_patterns": [
            "I decided to",
            "I chose",
            "I said that because",
            "my reason was",
            "so i",
            "that's why i",
            "which is why",
            "because of that i",
            "i responded by",
            "i reacted",
            "naturally i",
        ],
        "description": "Reasons given for choices may be post-hoc constructions",
        "probe": "What were you feeling in the moment just before you said/did that?",
        "what_to_listen_for": "Whether the stated reason holds up or a more visceral driver emerges",
    },
    {
        "mechanism": "moral_licensing",
        "signal_patterns": [
            "after everything i've done",
            "i've always",
            "i've given so much",
            "i deserved",
            "after all i",
            "considering how much",
            "given everything i",
            "i've been nothing but",
            "i've always been",
            "i gave",
            "i sacrificed",
        ],
        "description": "Prior good behavior invoked to justify current action",
        "probe": "Does the prior good actually connect to this situation, or is it being used as credit?",
        "what_to_listen_for": "Whether the justification is proportionate or a rationalization",
    },
]


# ─── Profile storage ──────────────────────────────────────────────────────────


def _load_profiles() -> dict:
    if PROFILES_PATH.exists():
        return json.loads(PROFILES_PATH.read_text())
    return {}


def _save_profiles(profiles: dict):
    tmp = PROFILES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(profiles, indent=2))
    tmp.replace(PROFILES_PATH)


# ─── MCP Tools ────────────────────────────────────────────────────────────────


@mcp.tool()
def predict_mechanisms(
    profile: dict[str, str],
    situation: list[str] = [],
    top_n: int = 12,
) -> dict:
    """
    Given a person profile and active situation features, return the most likely
    active behavioral mechanisms ranked by predictive score.

    profile: dict mapping dimension keys to '+' (high/present) or '-' (low/absent).
      Trait keys: big_five_O, big_five_C, big_five_E, big_five_A, big_five_N,
        dark_triad_narcissism, dark_triad_machiavellianism, dark_triad_psychopathy,
        attachment_anxious, attachment_avoidant, bis_sensitivity, bas_sensitivity,
        need_for_cognition, need_for_closure, disgust_sensitivity,
        sensory_processing_sensitivity, alexithymia, social_dominance_orientation, hexaco_H
      State keys: cognitive_load, affective_valence, affective_arousal,
        resource_scarcity_state, power_state, threat_appraisal, in_group_salience,
        relationship_history, fatigue_depletion

    situation: list of active situation feature keys.
      Valid features: stakes, social_visibility, time_pressure, ambiguity,
        out_group_salience, power_differential, resource_availability, novelty,
        relationship_type, conflict_present, anonymity, social_norms_clarity,
        physical_threat, group_context, outcome_reversibility, prior_commitment, surveillance

    Returns ranked mechanisms with scores, matches, behavioral outputs, and detected tensions.
    """
    with db() as conn:
        results = _score_mechanisms(conn, profile, situation, top_n)
    return {
        "profile_dimensions": len(profile),
        "situation_features": situation,
        "mechanisms": results,
        "total_active": len(results),
    }


@mcp.tool()
def get_profile_questions(
    situation: list[str] = [],
    known_dimensions: dict[str, str] = {},
    framing: str = "third_person",
    n_questions: int = 3,
) -> dict:
    """
    Return the highest-leverage interview questions for building a profile,
    given the current situation and what's already been established.

    situation: list of active situation feature keys (use [] if unknown)
    known_dimensions: dict of already-established dimensions {dim: '+'/'-'}
    framing: 'third_person' (narrator describing a character) or
             'first_person' (self-audit — asking about yourself)
    n_questions: number of questions to return per round (keep small for conversational feel)

    Returns prioritized questions and a coverage signal.
    Call repeatedly until coverage.ready_to_predict is True.
    """
    with db() as conn:
        leverage = _dimension_leverage(conn, situation)

    coverage = _coverage(leverage, known_dimensions)

    # Sort dimensions by leverage, exclude known ones
    remaining = [
        (dim, lev)
        for dim, lev in sorted(leverage.items(), key=lambda x: -x[1])
        if dim not in known_dimensions and dim in QUESTION_BANK
    ]

    questions = []
    for dim, lev in remaining[:n_questions]:
        qdata = QUESTION_BANK[dim]
        questions.append(
            {
                "dimension": dim,
                "leverage": round(lev, 2),
                "question": qdata.get(framing, qdata["third_person"]),
                "follow_ups": qdata.get("follow_ups", []),
                "what_it_reveals": qdata.get("markers", ""),
            }
        )

    return {
        "questions": questions,
        "coverage": coverage,
        "framing": framing,
        "hint": (
            "Ready to predict — call predict_mechanisms with current profile."
            if coverage["ready_to_predict"]
            else f"Continue interview — {coverage['dimensions_remaining']} high-leverage "
            f"dimensions remain ({coverage['remaining_leverage']} leverage units uncovered)."
        ),
    }


@mcp.tool()
def analyze_description_bias(
    description: str,
    known_profile: dict[str, str] | None = None,
) -> dict:
    """
    Analyze a self-report description for systematic distortions predicted by
    posthoc rationalization mechanisms. Returns likely bias patterns and
    the probe questions most likely to surface what's missing.

    For use in self-audit interviews: call this with the person's initial
    description before asking follow-up questions. Do NOT reveal the bias
    analysis to the person being interviewed until the full picture is assembled.

    description: the person's natural-language account of the situation
    known_profile: optional dict of established profile dimensions
    """
    desc_lower = description.lower()

    triggered = []
    for probe in BIAS_PROBES:
        # Check signal patterns
        pattern_hits = [p for p in probe["signal_patterns"] if p in desc_lower]

        # Always include the core biases (they're structurally present in self-reports)
        # but mark pattern-matched ones as higher confidence
        triggered.append(
            {
                "mechanism": probe["mechanism"],
                "confidence": "high" if pattern_hits else "moderate",
                "signal": (
                    f"Pattern match: {pattern_hits}"
                    if pattern_hits
                    else "Structurally likely in self-report context"
                ),
                "description": probe["description"],
                "probe_question": probe["probe"],
                "what_to_listen_for": probe["what_to_listen_for"],
            }
        )

    # Sort: high confidence first
    triggered.sort(key=lambda x: 0 if x["confidence"] == "high" else 1)

    # Identify missing situation context
    missing_context = []
    for feat, label in [
        ("power_differential", "power dynamics between parties"),
        ("relationship_type", "relationship history and type"),
        ("stakes", "what's concretely at stake for each party"),
        ("prior_commitment", "prior obligations or exchanges"),
    ]:
        if feat not in desc_lower and label not in desc_lower:
            missing_context.append(
                {
                    "feature": feat,
                    "prompt": f"Ask about: {label}",
                }
            )

    return {
        "likely_distortions": triggered,
        "missing_context": missing_context,
        "interview_note": (
            "Conduct the probe questions before revealing this analysis. "
            "Premature disclosure allows impression_management to contaminate answers."
        ),
        "argyris_note": (
            "The gap between their espoused account and their theory-in-use will "
            "emerge through the probes — particularly around their own goals and actions."
        ),
    }


@mcp.tool()
def save_profile(name: str, profile: dict[str, str], notes: str = "") -> dict:
    """
    Save a named profile for reuse across calls.

    name: identifier for this profile (e.g. 'marcus', 'self', 'npc_merchant')
    profile: dict of {dimension: '+'/'-'}
    notes: optional free-text description of who this profile represents
    """
    profiles = _load_profiles()
    profiles[name] = {"dimensions": profile, "notes": notes}
    _save_profiles(profiles)
    return {"saved": name, "dimensions": len(profile), "notes": notes}


@mcp.tool()
def load_profile(name: str) -> dict:
    """
    Load a previously saved named profile.

    name: profile identifier used in save_profile
    """
    profiles = _load_profiles()
    if name not in profiles:
        available = list(profiles.keys())
        return {"error": f"Profile '{name}' not found", "available": available}
    return {"name": name, **profiles[name]}


@mcp.tool()
def list_profiles() -> dict:
    """List all saved profiles."""
    profiles = _load_profiles()
    return {
        "profiles": [
            {"name": k, "dimensions": len(v["dimensions"]), "notes": v.get("notes", "")}
            for k, v in profiles.items()
        ]
    }


@mcp.tool()
def get_mechanism(mechanism_id: str) -> dict:
    """
    Return full detail for a mechanism including description, evidence quality,
    behavioral outputs, triggers, person moderators, and situation activators.

    mechanism_id: the snake_case id (e.g. 'loss_aversion', 'reciprocity')
    Use search_mechanisms to find IDs by keyword.
    """
    with db() as conn:
        row = conn.execute("SELECT * FROM mechanisms WHERE id=?", (mechanism_id,)).fetchone()
        if row is None:
            # Fuzzy fallback
            rows = conn.execute(
                "SELECT id, name FROM mechanisms WHERE id LIKE ? OR name LIKE ?",
                (f"%{mechanism_id}%", f"%{mechanism_id}%"),
            ).fetchall()
            if not rows:
                return {"error": f"Mechanism '{mechanism_id}' not found"}
            return {"matches": [{"id": r["id"], "name": r["name"]} for r in rows]}

        pms = conn.execute(
            "SELECT dimension, direction, strength, note FROM person_moderators WHERE mechanism_id=?",
            (mechanism_id,),
        ).fetchall()
        sas = conn.execute(
            "SELECT feature, effect, note FROM situation_activators WHERE mechanism_id=?",
            (mechanism_id,),
        ).fetchall()
        props = conn.execute(
            "SELECT key, value FROM mechanism_properties WHERE mechanism_id=? ORDER BY key",
            (mechanism_id,),
        ).fetchall()

    def _parse(v):
        if v is None:
            return None
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return v

    return {
        "id": row["id"],
        "name": row["name"],
        "domain": row["domain"],
        "description": row["description"] or row["summary"],
        "mechanism_type": row["mechanism_type"],
        "triggers": _parse(row["triggers"]),
        "behavioral_outputs": _parse(row["behavioral_outputs"] or row["outputs"]),
        "plain_language_outputs": _parse(row["plain_language_outputs"]),
        "narrative_outputs": _parse(row["narrative_outputs"]),
        "evidence": {
            "effect_size": row["effect_size"],
            "replication": row["replication"] or row["replication_status"],
            "cross_cultural": row["cross_cultural"] or row["cross_cultural_status"],
            "accuracy_score": row["accuracy_score"],
        },
        "individual_variation": _parse(row["individual_variation"] or row["variation"]),
        "cross_cultural": row["cross_cultural"] or row["cross_cultural_status"],
        "notes": row["notes"],
        "person_moderators": [dict(r) for r in pms],
        "situation_activators": [dict(r) for r in sas],
        "properties": {r["key"]: _parse(r["value"]) for r in props},
    }


def _fetch_mechanism_data(conn, mechanism_id: str) -> dict | None:
    """Return a plain dict with id, name, domain, description, plain_language_outputs for a mechanism."""
    row = conn.execute(
        "SELECT id, name, domain, description, summary, plain_language_outputs "
        "FROM mechanisms WHERE id=?",
        (mechanism_id,),
    ).fetchone()
    if row is None:
        return None
    # description may live in mechanism_properties if not in top-level columns
    desc = row["description"] or row["summary"] or ""
    if not desc:
        prop = conn.execute(
            "SELECT value FROM mechanism_properties WHERE mechanism_id=? AND key='definition'",
            (mechanism_id,),
        ).fetchone()
        if prop:
            desc = prop["value"] or ""
    plo = row["plain_language_outputs"]
    try:
        plo_parsed = json.loads(plo) if plo else []
    except (json.JSONDecodeError, TypeError):
        plo_parsed = []
    return {
        "id": row["id"],
        "name": row["name"],
        "domain": row["domain"],
        "description": desc,
        "plain_language_outputs": plo_parsed,
    }


@mcp.tool()
def verbalize_motivation(
    hidden_mechanism_id: str,
    action_description: str,
    profile: dict[str, str] | None = None,
    situation: list[str] | None = None,
    framing: str = "first_person",
    rationalization_template_id: str | None = None,
) -> dict:
    """
    Given a hidden behavioral mechanism driving someone's action, generate plausible
    verbalizations — the surface-level rationalizations they would actually voice.

    Useful for character dialogue, NPC behavior, deception detection, and self-audit.

    hidden_mechanism_id: snake_case mechanism ID (e.g. 'status_dominance_assertion',
        'loss_aversion', 'ingroup_favoritism'). Use search_mechanisms to find IDs.
    action_description: what the person just did (e.g. "took credit for a teammate's idea
        in a meeting", "refused to admit the strategy was wrong")
    profile: optional {dimension: '+'/'-'} — used to select the most fitting
        rationalization template for this person type
    situation: optional list of situation features — used to select rationalization template
    framing: 'first_person' (default) | 'third_person' | 'dialogue'
        first_person  → "I just needed to…"
        third_person  → "She told herself that…"
        dialogue      → tagged dialogue lines: Character: "..."
    rationalization_template_id: optional — directly specify which posthoc_rationalization
        mechanism to use as the verbalization template (e.g. 'belief_perseverance',
        'moral_licensing', 'self_serving_bias'). Bypasses automatic template selection.
        Use get_mechanism or search_mechanisms to browse options.
    """
    with db() as conn:
        # ── 1. Look up the hidden mechanism ───────────────────────────────────
        hidden = _fetch_mechanism_data(conn, hidden_mechanism_id)
        if hidden is None:
            return {"error": f"Mechanism '{hidden_mechanism_id}' not found. Use search_mechanisms."}

        # ── 2. Score posthoc_rationalization mechanisms for this profile/situation
        profile = profile or {}
        situation = situation or []

        # Score only the rationalization domain
        rat_mechs = conn.execute(
            "SELECT id, name, domain, description, summary, "
            "behavioral_outputs, outputs, triggers, effect_size, "
            "replication, replication_status, accuracy_score, plain_language_outputs "
            "FROM mechanisms WHERE domain='posthoc_rationalization'"
        ).fetchall()

        situation_set = set(situation)
        rat_results = []
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

            plo = mech["plain_language_outputs"]
            try:
                plo_parsed = json.loads(plo) if plo else []
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

            rat_results.append(
                {
                    "id": mid,
                    "name": mech["name"],
                    "score": person_score * (1 + situation_score * SITUATION_MULTIPLIER),
                    "description": desc_r,
                    "plain_language_outputs": plo_parsed,
                }
            )

    # If caller specified a template explicitly, use it directly
    if rationalization_template_id:
        rat_template = next(
            (r for r in rat_results if r["id"] == rationalization_template_id), None
        )
        if rat_template is None:
            return {
                "error": (
                    f"rationalization_template_id '{rationalization_template_id}' not found "
                    "in posthoc_rationalization domain. Use search_mechanisms to browse options."
                )
            }
    else:
        # Auto-select: sort by score descending; fall back to self_serving_bias
        # (motivated_reasoning has a required activator so it may be excluded with no situation)
        rat_results.sort(key=lambda x: x["score"], reverse=True)
        if rat_results and rat_results[0]["score"] > 0:
            rat_template = rat_results[0]
        else:
            rat_template = next(
                (r for r in rat_results if r["id"] == "self_serving_bias"),
                rat_results[0] if rat_results else None,
            )

    if rat_template is None:
        return {"error": "No rationalization templates available."}

    # ── 3. Build verbalization prompt ─────────────────────────────────────────
    framing_instructions = {
        "first_person": (
            "Generate 4-5 first-person statements (using I/me/my) this person would "
            "actually say out loud. They sound sincere and self-justifying."
        ),
        "third_person": (
            "Generate 4-5 statements in third person describing what the person tells "
            "themselves internally. Use 'She told herself…', 'He felt that…', etc."
        ),
        "dialogue": (
            "Generate 4-5 dialogue lines this character would speak aloud. "
            'Format each as: Character: "..."'
        ),
    }.get(framing, "Generate 4-5 first-person statements.")

    hidden_plo = ", ".join(f'"{p}"' for p in hidden["plain_language_outputs"][:6])
    rat_plo = ", ".join(f'"{p}"' for p in rat_template["plain_language_outputs"][:6])

    prompt = f"""You are generating dialogue for a character study in behavioral psychology.

ACTUAL HIDDEN DRIVER
Mechanism: {hidden["name"]}
What it produces: {hidden_plo}
Core dynamic: {hidden["description"][:280]}

ACTION TAKEN: {action_description}

RATIONALIZATION TEMPLATE
The character processes this via: {rat_template["name"]}
Verbal patterns for this rationalization: {rat_plo}
Core pattern: {rat_template["description"][:200]}

TASK
{framing_instructions}

Rules:
- Do NOT name or reference the actual psychological mechanism
- The character is NOT self-aware about the rationalization
- Statements should sound genuine, not rehearsed
- Draw on the verbal pattern vocabulary above
- Each statement 1-2 sentences max

Output: a JSON array of strings only. No commentary."""

    # Return structured data + verbalization_prompt.
    # The calling Claude session generates the actual dialogue from the prompt —
    # drivermap's value is the lookup/selection, not the text generation.
    return {
        "hidden_mechanism": {
            "id": hidden["id"],
            "name": hidden["name"],
            "domain": hidden["domain"],
            "description": hidden["description"][:300],
            "plain_language_outputs": hidden["plain_language_outputs"],
        },
        "rationalization_template": {
            "id": rat_template["id"],
            "name": rat_template["name"],
            "score": round(rat_template["score"], 2),
            "description": rat_template["description"][:300],
            "plain_language_outputs": rat_template["plain_language_outputs"],
        },
        "action": action_description,
        "framing": framing,
        "verbalization_prompt": prompt,
    }


@mcp.tool()
def search_mechanisms(query: str, domain: str = None) -> dict:
    """
    Full-text search across mechanism names, descriptions, and notes.

    query: search string
    domain: optional domain filter (e.g. 'posthoc_rationalization', 'status_dominance')
    """
    with db() as conn:
        q = f"%{query}%"
        if domain:
            rows = conn.execute(
                "SELECT id, name, domain, description, summary FROM mechanisms "
                "WHERE (name LIKE ? OR description LIKE ? OR summary LIKE ? OR notes LIKE ?) "
                "AND domain LIKE ? ORDER BY name",
                (q, q, q, q, f"%{domain}%"),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, domain, description, summary FROM mechanisms "
                "WHERE name LIKE ? OR description LIKE ? OR summary LIKE ? OR notes LIKE ? "
                "ORDER BY name",
                (q, q, q, q),
            ).fetchall()

    return {
        "query": query,
        "results": [
            {
                "id": r["id"],
                "name": r["name"],
                "domain": r["domain"],
                "description": (r["description"] or r["summary"] or "")[:200],
            }
            for r in rows
        ],
        "count": len(rows),
    }


@mcp.tool()
def list_dimensions() -> dict:
    """
    Return the complete vocabulary of valid dimension and feature keys,
    for use when constructing profiles and situation descriptions.
    """
    return {
        "person_trait_dimensions": {
            "big_five_O": "Openness to experience — curiosity, creativity, novelty-seeking",
            "big_five_C": "Conscientiousness — self-control, diligence, follow-through",
            "big_five_E": "Extraversion — social energy, assertiveness, positive affect",
            "big_five_A": "Agreeableness — cooperativeness, empathy, conflict avoidance",
            "big_five_N": "Neuroticism — emotional reactivity, anxiety, moodiness",
            "dark_triad_narcissism": "Grandiosity, entitlement, need for admiration",
            "dark_triad_machiavellianism": "Strategic manipulation, instrumental view of others",
            "dark_triad_psychopathy": "Low empathy, fearlessness, callousness, impulsivity",
            "attachment_anxious": "Fear of abandonment, hypervigilance to relationship cues",
            "attachment_avoidant": "Discomfort with intimacy, self-reliance as defense",
            "bis_sensitivity": "Behavioral inhibition — anxiety-driven avoidance of punishment",
            "bas_sensitivity": "Behavioral activation — reward-driven approach motivation",
            "need_for_cognition": "Intrinsic motivation to think; enjoyment of complex problems",
            "need_for_closure": "Discomfort with ambiguity; push for definite answers",
            "disgust_sensitivity": "Heightened moral and physical disgust reactivity",
            "sensory_processing_sensitivity": "Deep processing, emotional contagion susceptibility",
            "alexithymia": "Difficulty identifying and describing emotional states",
            "social_dominance_orientation": "Endorsement of group hierarchies and inequality",
            "hexaco_H": "Honesty-humility — fairness motivation, non-exploitation",
        },
        "person_state_dimensions": {
            "cognitive_load": "Currently mentally stretched — limited deliberate processing bandwidth",
            "affective_valence": "Current emotional tone: '+' = positive, '-' = negative",
            "affective_arousal": "Current activation level: '+' = high/agitated, '-' = calm",
            "resource_scarcity_state": "Currently experiencing scarcity (money, time, status)",
            "power_state": "Current power/control: '+' = high power, '-' = low/dependent",
            "threat_appraisal": "Perceiving current situation as genuinely threatening",
            "in_group_salience": "Group identity actively activated; us-vs-them salient",
            "relationship_history": "Prior exchanges with others in situation: '+' = positive history, '-' = negative/adversarial",
            "fatigue_depletion": "Physically or emotionally depleted; self-regulation impaired",
        },
        "situation_features": {
            "stakes": "High-stakes outcome — significant consequences riding on this",
            "social_visibility": "Actions observable by others; reputational consequences present",
            "time_pressure": "Urgency; limited time to deliberate",
            "ambiguity": "Situation or others' intentions unclear",
            "out_group_salience": "Out-group presence or contrast is salient",
            "power_differential": "Clear asymmetry of power between parties",
            "resource_availability": "Resources (money, status, information) are scarce or contested",
            "novelty": "Situation is unfamiliar or unprecedented",
            "relationship_type": "Interaction occurs within an established relationship",
            "conflict_present": "Active conflict or disagreement in the situation",
            "anonymity": "Actions not linked to identity; accountability reduced",
            "social_norms_clarity": "Strong, clear norms governing appropriate behavior",
            "physical_threat": "Physical safety or bodily integrity at risk",
            "group_context": "Decision or behavior occurs in a group setting",
            "outcome_reversibility": "Outcomes can ('+') or cannot ('-') be undone",
            "prior_commitment": "Prior commitment, promise, or exchange already made",
            "surveillance": "Being watched or monitored",
        },
        "usage_notes": {
            "profile_values": "'+' = high/present/elevated, '-' = low/absent/diminished",
            "situation_values": "List feature keys that are actively present in the situation",
            "partial_profiles": "Unspecified dimensions are treated as neutral (no moderating effect)",
        },
    }


if __name__ == "__main__":
    mcp.run()
