# Drivermap Integration Guide

Drivermap is a behavioral mechanisms knowledge base with 137 mechanisms across 7 domains.
It exposes a FastMCP server (`mcp_server.py`) for use in Claude Code sessions.

**Primary use cases:**
- Predicting which behavioral mechanisms are active for a given person in a given situation
- Generating plausible rationalizations a person would voice for a behavior (character writing, deception detection, dialog)
- Conducting a structured behavioral interview to build a profile
- Self-auditing decisions for posthoc rationalization distortions

**Intended use:** Profiles are local and ephemeral by default (`profiles.json` in the project
root, gitignored). This is designed for self-audit, research, and character/dialog work —
not covert profiling. The `analyze_description_bias` tool operates on text the user provides
about themselves or a situation, not on third-party data collected without consent.

**`accuracy_score`** on each mechanism (visible in `get_mechanism` output and the heatmap)
is a 0–1 score from a verifier pass during extraction: the LLM checks whether the extracted
record accurately represents its Wikipedia/Kagi source. Average score across all 137
mechanisms is ~0.90. It measures extraction fidelity, not empirical validity of the
underlying psychology.

---

## Quick Start

The MCP server is registered as `"drivermap"` in `~/.claude/settings.json`. All tools are
available in any Claude Code session without additional setup.

```
# Start the server manually if needed:
python3 /path/to/drivermap/mcp_server.py
```

---

## Data Model

### Domains (7)

| Domain | Description | Mechanisms |
|--------|-------------|------------|
| `individual_variation` | Personality and trait-based behavioral differences | 28 |
| `posthoc_rationalization` | How people explain their own behavior after the fact | 26 |
| `threat_affective_priming` | Fear, threat appraisal, and emotional priming effects | 22 |
| `social_influence_compliance` | Conformity, social proof, authority, persuasion | 18 |
| `status_dominance` | Status signaling, dominance, hierarchy navigation | 16 |
| `loss_aversion_reference` | Loss aversion, reference points, sunk costs | 14 |
| `ingroup_outgroup` | In-group favoritism, tribalism, out-group derogation | 13 |

### Profile Dimensions (29)

A profile is a dict mapping dimension names to `"+"` (elevated) or `"-"` (diminished).
You only need to specify dimensions you know — omitted dimensions are treated as neutral.

**Trait dimensions** (stable personality characteristics):
```
big_five_O        # Openness to experience
big_five_C        # Conscientiousness
big_five_E        # Extraversion
big_five_A        # Agreeableness
big_five_N        # Neuroticism
hexaco_H          # Honesty-Humility
attachment_anxious
attachment_avoidant
bas_sensitivity   # Behavioral activation system (reward-seeking)
bis_sensitivity   # Behavioral inhibition system (threat-avoidance)
need_for_cognition
need_for_closure
sensory_processing_sensitivity
dark_triad_narcissism
dark_triad_machiavellianism
dark_triad_psychopathy
social_dominance_orientation
disgust_sensitivity
alexithymia       # Difficulty identifying/describing emotions
```

**State dimensions** (situational/transient):
```
affective_arousal      # Current emotional activation level
affective_valence      # Current emotional tone (positive/negative)
cognitive_load         # Current mental load
fatigue_depletion      # Current energy depletion
power_state            # Current perceived power/status
threat_appraisal       # Current threat perception
in_group_salience      # How salient group identity is right now
resource_scarcity_state
prior_commitment       # Degree of prior investment/commitment
relationship_history   # Quality of recent relationship history
```

### Situation Features

Situation features describe the objective properties of the current context.
Pass as a list of strings from this vocabulary:

```
ambiguity              conflict_present       group_context
novelty                out_group_salience     outcome_reversibility
physical_threat        power_differential     power_holder
power_low              prior_commitment       relationship_type
resource_availability  social_norms_clarity   social_visibility
stakes                 surveillance           time_pressure
anonymity
```

**Power directionality:** `power_differential` indicates a power asymmetry exists.
`power_holder` / `power_low` specify which side the profiled person is on. A manager
with `power_holder` activates dominance/power mechanisms; the same situation
with `power_low` activates obedience/sycophancy. The undirected `power_differential`
remains valid and backward-compatible for integrations that don't track direction.

### Scoring Formula

Mechanisms are scored using a multiplicative person × situation model:

```
total = person_score * (1 + situation_score * SITUATION_MULTIPLIER)
```

where `SITUATION_MULTIPLIER = 0.5`.

**Person score** sums dimension matches (strong=1.5, moderate=1.0, weak=0.5; mismatch
penalizes at 0.5× weight; mixed directions contribute 0.25). If person_score ≤ 0, the
mechanism is excluded — situation cannot rescue a personality mismatch.

**Situation score** sums activated features (required=2.0/excluded, activates=2.0,
amplifies=1.0, dampens=−1.0).

This follows the interactionist principle (Lewin, Mischel): person traits are necessary
but not sufficient — situation enables/triggers what the person is disposed toward.

---

## MCP Tools

### `predict_mechanisms(profile, situation, top_n)`

**The core tool.** Given a profile and situation, returns ranked behavioral mechanisms.

```python
predict_mechanisms(
    profile={
        "big_five_N": "+",
        "attachment_anxious": "+",
        "need_for_closure": "+"
    },
    situation=["social_visibility", "stakes", "ambiguity"],
    top_n=10   # default 10, max ~30
)
```

Returns: ranked list with mechanism IDs, names, domains, scores, plain-language outputs,
and any active tensions (mechanisms pulling in opposing directions).

Each result includes two output registers:
- **`narrative_outputs`** — clinical-register mechanism descriptions for LLM system prompts
  (e.g. "status-maintenance drive active — monitors for challenges to authority"). **Prefer
  these for LLM integrations.**
- **`plain_language_outputs`** — casual-register observable behaviors for dashboards and
  diagnostics (e.g. "wants approval", "follows the crowd").

---

### `get_profile_questions(known_dimensions, situation, framing)`

Conducts an iterative behavioral interview. Call repeatedly, updating `known_dimensions`
with each answer, until `ready_to_predict` is `true`.

```python
# First call — no profile yet
get_profile_questions(
    known_dimensions={},
    situation=["stakes", "social_visibility"],
    framing="third_person"  # or "first_person"
)

# Returns: {
#   "questions": [...],
#   "coverage": {"known": 3, "total": 28, "pct": 11},
#   "ready_to_predict": false
# }

# Subsequent calls — pass accumulated profile
get_profile_questions(
    known_dimensions={"big_five_N": "+", "bas_sensitivity": "+"},
    situation=["stakes"],
    framing="third_person"
)
```

`ready_to_predict` becomes `true` when enough high-signal dimensions are covered.
Questions are situation-weighted — the most diagnostic dimensions for the given
situation are asked first.

---

### `verbalize_motivation(hidden_mechanism_id, action_description, profile, situation, framing, rationalization_template_id)`

**The dialog/rationalization tool.** This is a prompt builder, not a text generator.
It returns a structured `verbalization_prompt` that you pass to Claude (or any LLM) to
produce the actual rationalization text. Given a hidden behavioral mechanism and the
action it produced, it assembles the context needed to generate what the person would
actually say out loud.

```python
verbalize_motivation(
    hidden_mechanism_id="status_threat_response",
    action_description="refused to acknowledge the participant's point in front of others",
    profile={"dark_triad_narcissism": "+", "social_dominance_orientation": "+"},
    situation=["social_visibility", "power_differential"],
    framing="dialogue",   # "first_person" | "third_person" | "dialogue"
    # rationalization_template_id="belief_perseverance"  # optional override
)
```

**Framing options:**
- `first_person` → `"I just needed to make sure the direction was clear…"`
- `third_person` → `"He told himself the other person simply hadn't understood…"`
- `dialogue` → `Character: "Look, I just wanted to make sure we stayed on track."`

**Returns** a dict with `verbalization_prompt` (pass this to Claude to generate the
actual text), `rationalization_template` (the selected posthoc mechanism and its outputs),
and `hidden_mechanism` metadata. **You must pass `verbalization_prompt` to an LLM to get
the rationalization — this tool provides data and context, not finished prose.**

**`rationalization_template_id` override** (important for character/dialog work):
Auto-selection can be unreliable for short action descriptions. When you know which
rationalization pattern applies, specify it directly:

```python
rationalization_template_id="belief_perseverance"  # ignores my evidence
rationalization_template_id="moral_licensing"       # I earned this
rationalization_template_id="fundamental_attribution_error"  # they're just that way
rationalization_template_id="self_serving_bias"     # success=me, failure=circumstances
rationalization_template_id="identity_protective_cognition"  # threatens my worldview
rationalization_template_id="moral_dumbfounding"    # I know it's wrong but...
rationalization_template_id="sacred_values"         # some things aren't negotiable
rationalization_template_id="motivated_reasoning"   # working backward from conclusion
```

Full list of templates: `search_mechanisms(query="", domain="posthoc_rationalization")`.

---

### `analyze_description_bias(description, profile)`

Detects posthoc rationalization patterns in a person's self-report. Use **before**
revealing your analysis — asking people to elaborate first surfaces more signal.

```python
analyze_description_bias(
    description="I didn't want to embarrass him in front of everyone, so I waited "
                "until after the meeting to give feedback. It was the kind thing to do.",
    profile={"dark_triad_narcissism": "+"}  # optional context
)
```

Returns: detected distortion patterns, likely hidden mechanisms, and follow-up questions
designed to probe without telegraphing the analysis.

---

### `save_profile(name, profile, notes)` / `load_profile(name)` / `list_profiles()`

Persist participant profiles across sessions.

```python
save_profile(
    name="participant_42",
    profile={"big_five_N": "+", "attachment_anxious": "+"},
    notes="Completed intake 2025-03-01. High threat sensitivity."
)

profile = load_profile("participant_42")
```

---

### `get_mechanism(mechanism_id)` / `search_mechanisms(query, domain)`

Browse the knowledge base.

```python
get_mechanism("loss_aversion")
# Returns full mechanism data: description, triggers, outputs, plain_language_outputs,
# person_moderators, situation_activators, interactions, effect_size, replication status

search_mechanisms(query="status threat face saving")
search_mechanisms(query="", domain="posthoc_rationalization")  # browse a full domain
```

---

### `list_dimensions()`

Returns the full dimension vocabulary with descriptions. Useful when building
a profile from behavioral observations rather than direct questioning.

---

## Worked Example: Intake → Prediction → Verbalization

```python
# 1. INTAKE — build profile via iterative interview
profile = {}
while True:
    result = get_profile_questions(
        known_dimensions=profile,
        situation=["stakes", "social_visibility", "novelty"],
        framing="third_person"
    )
    # ... present questions, collect answers, update profile ...
    if result["ready_to_predict"]:
        break

save_profile("session_7", profile)

# 2. PREDICT — rank active mechanisms for this person + situation
mechanisms = predict_mechanisms(
    profile=profile,
    situation=["social_visibility", "stakes", "ambiguity"],
    top_n=15
)
# mechanisms[0]["id"], mechanisms[0]["plain_language_outputs"], etc.

# 3. VERBALIZE — surface rationalization for a specific behavior
dialog_lines = verbalize_motivation(
    hidden_mechanism_id="ingroup_favoritism",
    action_description="confided a personal failure to make the other person feel less alone",
    framing="first_person",
    rationalization_template_id="self_serving_bias"
)
# → pass dialog_lines["verbalization_prompt"] to Claude to generate the actual text

# 4. BIAS CHECK — did the person's own account distort what happened?
bias_check = analyze_description_bias(
    description=debrief_text,
    profile=profile
)
# → detected mechanisms reveal which rationalizations are active
```

---

## Tips for Effective Use

**Profile building:**
- State dimensions (arousal, threat_appraisal, cognitive_load) are often more predictive
  than trait dimensions for a specific moment. Don't neglect them.
- A profile with 4-6 well-chosen dimensions outperforms one with 15 weakly-inferred ones.
- `analyze_description_bias` on the participant's own words often surfaces trait dimensions
  more reliably than direct questioning.

**Situation features:**
- `stakes` + `social_visibility` + `ambiguity` is the high-activation triad — activates
  the most mechanisms across domains.
- Situation features alone produce no results — multiplicative scoring requires
  person_score > 0. Always provide at least one profile dimension.
- `prior_commitment` (situation feature) and `prior_commitment` (person dimension) are
  different: the situation feature means "they've already invested in this"; the person
  dimension means "they're dispositionally prone to escalating commitments."

**Verbalization:**
- `verbalize_motivation` returns a **prompt**, not text — pass the returned
  `verbalization_prompt` to `claude --print` or your LLM API to generate the actual
  rationalization.
- Always provide `profile` and `situation` to `verbalize_motivation` even if incomplete —
  it substantially improves template selection.
- For character work, prefer explicit `rationalization_template_id` over auto-selection.
- `dialogue` framing produces lines suitable for direct use; `third_person` is better for
  narrator/stage direction style writing.

**Mechanism IDs:**
- Use `search_mechanisms` to find IDs — don't guess. The IDs are exact snake_case strings.
- `get_mechanism(id)` returns `plain_language_outputs` — these are the observable surface
  behaviors and statements. They're the most useful field for matching to real-world events.
