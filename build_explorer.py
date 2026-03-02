#!/usr/bin/env python3
"""
build_explorer.py — Generate a self-contained static HTML explorer.

Reads from:
  - db/mechanisms.sqlite (mechanisms, person_moderators, situation_activators, mechanism_properties)
  - dimensions.json (human-readable labels)

Writes:
  - docs/index.html (all data embedded as JSON, scoring in vanilla JS)

Usage:
    python build_explorer.py           # → docs/index.html
    python build_explorer.py --output docs/index.html
"""

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "db" / "mechanisms.sqlite"
DIMS_PATH = ROOT / "dimensions.json"

DOMAIN_ORDER = [
    "threat_affective_priming",
    "status_dominance",
    "posthoc_rationalization",
    "ingroup_outgroup",
    "social_influence_compliance",
    "individual_variation",
    "loss_aversion_reference",
]

DOMAIN_RGB = {
    "threat_affective_priming": (0.82, 0.22, 0.24),
    "status_dominance": (0.52, 0.25, 0.72),
    "posthoc_rationalization": (0.88, 0.53, 0.10),
    "ingroup_outgroup": (0.22, 0.56, 0.28),
    "social_influence_compliance": (0.12, 0.58, 0.58),
    "individual_variation": (0.16, 0.40, 0.70),
    "loss_aversion_reference": (0.68, 0.33, 0.14),
}

DOMAIN_LABEL = {
    "threat_affective_priming": "Threat / Affective Priming",
    "status_dominance": "Status & Dominance",
    "posthoc_rationalization": "Posthoc Rationalization",
    "ingroup_outgroup": "Ingroup / Outgroup",
    "social_influence_compliance": "Social Influence",
    "individual_variation": "Individual Variation",
    "loss_aversion_reference": "Loss Aversion / Reference",
}


def rgb_to_css(rgb):
    return f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Mechanisms
    mechs = []
    for row in conn.execute(
        "SELECT id, name, domain, description, summary, plain_language_outputs, "
        "narrative_outputs, accuracy_score "
        "FROM mechanisms ORDER BY domain, name"
    ):
        desc = row["description"] or row["summary"] or ""
        if not desc:
            prop = conn.execute(
                "SELECT value FROM mechanism_properties WHERE mechanism_id=? AND key='definition'",
                (row["id"],),
            ).fetchone()
            if prop:
                desc = prop["value"] or ""
        plo = row["plain_language_outputs"]
        try:
            plo_parsed = json.loads(plo) if plo else []
        except (json.JSONDecodeError, TypeError):
            plo_parsed = []
        narr = row["narrative_outputs"]
        try:
            narr_parsed = json.loads(narr) if narr else []
        except (json.JSONDecodeError, TypeError):
            narr_parsed = []
        mechs.append(
            {
                "id": row["id"],
                "name": row["name"],
                "domain": row["domain"],
                "description": desc,
                "plo": plo_parsed,
                "narrative": narr_parsed,
                "accuracy": row["accuracy_score"],
            }
        )

    # Person moderators
    person_mods = {}
    for row in conn.execute(
        "SELECT mechanism_id, dimension, direction, strength FROM person_moderators"
    ):
        person_mods.setdefault(row["mechanism_id"], []).append(
            {
                "dim": row["dimension"],
                "dir": row["direction"],
                "str": row["strength"] or "moderate",
            }
        )

    # Situation activators
    sit_acts = {}
    for row in conn.execute("SELECT mechanism_id, feature, effect FROM situation_activators"):
        sit_acts.setdefault(row["mechanism_id"], []).append(
            {
                "feat": row["feature"],
                "eff": row["effect"],
            }
        )

    conn.close()

    # Dimensions
    dims = json.loads(DIMS_PATH.read_text())

    return mechs, person_mods, sit_acts, dims


def build_html(mechs, person_mods, sit_acts, dims):
    """Build the complete self-contained HTML string."""

    # Prepare dimension lists
    trait_dims = {k: v for k, v in dims["person_trait_dimensions"].items() if k != "_note"}
    state_dims = {k: v for k, v in dims["person_state_dimensions"].items() if k != "_note"}
    sit_feats = {k: v for k, v in dims["situation_dimensions"].items() if k != "_note"}

    # Domain info for JS
    domain_info = {}
    for d in DOMAIN_ORDER:
        domain_info[d] = {
            "label": DOMAIN_LABEL[d],
            "color": rgb_to_css(DOMAIN_RGB[d]),
            "rgb": list(DOMAIN_RGB[d]),
        }

    # Count by domain
    domain_counts = {}
    for m in mechs:
        domain_counts[m["domain"]] = domain_counts.get(m["domain"], 0) + 1

    n_mechs = len(mechs)
    n_dims = len(trait_dims) + len(state_dims)
    n_feats = len(sit_feats)

    # JSON data blocks
    mechs_json = json.dumps(mechs, separators=(",", ":"))
    pmods_json = json.dumps(person_mods, separators=(",", ":"))
    sacts_json = json.dumps(sit_acts, separators=(",", ":"))
    trait_json = json.dumps(trait_dims, separators=(",", ":"))
    state_json = json.dumps(state_dims, separators=(",", ":"))
    feats_json = json.dumps(sit_feats, separators=(",", ":"))
    domain_json = json.dumps(domain_info, separators=(",", ":"))
    domain_order_json = json.dumps(DOMAIN_ORDER, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Drivermap Explorer</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#fafafa;--surface:#fff;--border:#e5e5e5;--text:#1a1a1a;
  --text2:#666;--text3:#999;--accent:#4a90d9;
  --neutral-bg:#f0f0f0;--pos-bg:#e8f5e9;--neg-bg:#e3f2fd;
  --pos:#2e7d32;--neg:#1565c0;--neutral:#888;
  --font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  --mono:"SF Mono",Monaco,Consolas,"Liberation Mono",monospace;
  --radius:6px;
}}
body{{font-family:var(--font);background:var(--bg);color:var(--text);
  font-size:14px;line-height:1.5;min-height:100vh}}

/* Header */
.header{{background:var(--surface);border-bottom:1px solid var(--border);
  padding:16px 24px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100}}
.header h1{{font-size:18px;font-weight:700;letter-spacing:-0.02em}}
.header .meta{{font-size:13px;color:var(--text2)}}

/* Layout */
.container{{display:flex;max-width:1400px;margin:0 auto;padding:12px;gap:12px;
  min-height:calc(100vh - 56px)}}
.sidebar{{width:320px;flex-shrink:0;display:flex;flex-direction:column;gap:12px;
  position:sticky;top:68px;align-self:flex-start;max-height:calc(100vh - 80px);
  overflow-y:auto}}
.results{{flex:1;min-width:0}}

/* Panels */
.panel{{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);overflow:hidden}}
.panel-header{{padding:10px 14px;font-size:12px;font-weight:600;
  text-transform:uppercase;letter-spacing:0.05em;color:var(--text2);
  border-bottom:1px solid var(--border);display:flex;justify-content:space-between;
  align-items:center}}
.panel-header{{cursor:pointer}}
.panel-header .reset{{font-size:11px;text-transform:none;letter-spacing:0;
  color:var(--accent);cursor:pointer;font-weight:500}}
.panel-header .reset:hover{{text-decoration:underline}}
.panel-header .arrow{{font-size:10px;margin-right:4px;display:inline-block;
  transition:transform 0.15s ease}}
.panel.collapsed .arrow{{transform:rotate(-90deg)}}
.panel-body{{padding:8px 14px;max-height:calc((100vh - 360px) / 3);overflow-y:auto}}
.panel.collapsed .panel-body,.panel.collapsed .domain-legend{{display:none}}

/* Dimension toggles */
.dim-group{{margin-bottom:6px}}
.dim-row{{display:flex;align-items:center;padding:3px 0;gap:8px}}
.dim-label{{flex:1;font-size:12.5px;color:var(--text);cursor:default;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.dim-label:hover{{color:var(--accent)}}
.dim-toggle{{width:44px;height:22px;border-radius:11px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;font-size:11px;
  font-weight:700;user-select:none;transition:all 0.15s ease;flex-shrink:0;
  border:1.5px solid var(--border)}}
.dim-toggle[data-state="0"]{{background:var(--neutral-bg);color:var(--neutral)}}
.dim-toggle[data-state="+"]{{background:var(--pos-bg);color:var(--pos);
  border-color:var(--pos)}}
.dim-toggle[data-state="-"]{{background:var(--neg-bg);color:var(--neg);
  border-color:var(--neg)}}

/* Situation checkboxes */
.feat-row{{display:flex;align-items:center;padding:3px 0;gap:8px}}
.feat-row label{{flex:1;font-size:12.5px;cursor:pointer;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.feat-row input[type="checkbox"]{{width:15px;height:15px;cursor:pointer;
  accent-color:var(--accent);flex-shrink:0}}

/* Results */
.results-header{{padding:12px 16px;display:flex;justify-content:space-between;
  align-items:center;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);margin-bottom:8px}}
.results-header h2{{font-size:14px;font-weight:600}}
.results-count{{font-size:12px;color:var(--text2)}}

.empty-state{{text-align:center;padding:60px 20px;color:var(--text3)}}
.empty-state .icon{{font-size:40px;margin-bottom:12px;opacity:0.4}}
.empty-state p{{font-size:14px;max-width:320px;margin:0 auto}}

/* Result cards */
.result-card{{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);margin-bottom:6px;cursor:pointer;
  transition:border-color 0.15s ease}}
.result-card:hover{{border-color:#ccc}}
.result-card.expanded{{border-color:var(--accent)}}
.result-main{{padding:10px 14px;display:flex;align-items:center;gap:12px}}
.result-rank{{font-size:13px;color:var(--text3);font-weight:600;min-width:22px;
  text-align:right}}
.result-info{{flex:1;min-width:0}}
.result-name{{font-size:14px;font-weight:600;margin-bottom:2px}}
.result-chips{{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}}

/* Bar */
.result-bar-wrap{{width:140px;flex-shrink:0;display:flex;align-items:center;gap:8px}}
.result-bar{{flex:1;height:8px;border-radius:4px;background:var(--neutral-bg);
  overflow:hidden;display:flex}}
.result-bar .seg-person{{height:100%;border-radius:4px 0 0 4px;opacity:0.85}}
.result-bar .seg-situation{{height:100%;opacity:0.45}}
.result-score{{font-size:13px;font-weight:700;min-width:36px;text-align:right;
  font-variant-numeric:tabular-nums}}

/* Chips */
.chip{{display:inline-block;font-size:10.5px;padding:1px 7px;border-radius:10px;
  font-weight:500;line-height:1.6}}
.chip-domain{{color:#fff;opacity:0.85}}
.chip-dim-pos{{background:var(--pos-bg);color:var(--pos)}}
.chip-dim-neg{{background:var(--neg-bg);color:var(--neg)}}
.chip-dim-mixed{{background:var(--neutral-bg);color:var(--neutral)}}
.chip-feat{{background:#e0f2f1;color:#00695c}}
.chip-feat-damp{{background:#eceff1;color:#546e7a}}

/* Expanded detail */
.result-detail{{display:none;padding:0 14px 14px;border-top:1px solid var(--border)}}
.result-card.expanded .result-detail{{display:block}}
.detail-section{{margin-top:10px}}
.detail-section h4{{font-size:11px;text-transform:uppercase;letter-spacing:0.05em;
  color:var(--text3);margin-bottom:4px}}
.plo-list{{list-style:none;padding:0}}
.plo-list li{{font-size:12.5px;color:var(--text2);padding:2px 0;padding-left:12px;
  position:relative}}
.plo-list li::before{{content:"·";position:absolute;left:0;color:var(--text3)}}

/* Verbalization */
.verb-section{{margin-top:12px;background:#fafafa;border:1px solid var(--border);
  border-radius:var(--radius);padding:12px}}
.verb-section h4{{margin-bottom:8px}}
.verb-prompt{{padding:10px;background:#fff;border:1px solid var(--border);
  border-radius:var(--radius);font-family:var(--mono);font-size:11.5px;
  line-height:1.5;white-space:pre-wrap;color:var(--text2);max-height:320px;
  overflow-y:auto}}

/* Domain legend */
.domain-legend{{display:flex;flex-wrap:wrap;gap:6px;padding:8px 14px}}
.domain-chip{{font-size:10.5px;padding:2px 8px;border-radius:10px;color:#fff;
  opacity:0.8;cursor:pointer;transition:opacity 0.15s}}
.domain-chip:hover,.domain-chip.active{{opacity:1}}

/* Responsive */
@media(max-width:900px){{
  .container{{flex-direction:column}}
  .sidebar{{width:100%;position:static;max-height:none}}
  .panel-body{{max-height:40vh}}
}}
</style>
</head>
<body>

<div class="header">
  <h1>Drivermap Explorer</h1>
  <span class="meta">{n_mechs} mechanisms &middot; {n_dims} dimensions &middot; {n_feats} situation features</span>
</div>

<div class="container">
  <div class="sidebar">
    <!-- Trait dimensions -->
    <div class="panel" id="panel-trait">
      <div class="panel-header" onclick="togglePanel('panel-trait', event)">
        <span><span class="arrow">&#9660;</span>Trait Dimensions</span>
        <span class="reset" onclick="resetDims('trait')">Reset</span>
      </div>
      <div class="panel-body" id="trait-dims"></div>
    </div>

    <!-- State dimensions -->
    <div class="panel" id="panel-state">
      <div class="panel-header" onclick="togglePanel('panel-state', event)">
        <span><span class="arrow">&#9660;</span>State Dimensions</span>
        <span class="reset" onclick="resetDims('state')">Reset</span>
      </div>
      <div class="panel-body" id="state-dims"></div>
    </div>

    <!-- Situation features -->
    <div class="panel" id="panel-sit">
      <div class="panel-header" onclick="togglePanel('panel-sit', event)">
        <span><span class="arrow">&#9660;</span>Situation Features</span>
        <span class="reset" onclick="resetFeats()">Reset</span>
      </div>
      <div class="panel-body" id="sit-feats"></div>
    </div>

    <!-- Domain legend -->
    <div class="panel" id="panel-domains">
      <div class="panel-header" onclick="togglePanel('panel-domains', event)">
        <span><span class="arrow">&#9660;</span>Domains</span>
      </div>
      <div class="domain-legend" id="domain-legend"></div>
    </div>
  </div>

  <div class="results" id="results-area">
    <div class="results-header">
      <h2>Predictions</h2>
      <span class="results-count" id="results-count"></span>
    </div>
    <div id="results-list"></div>
  </div>
</div>

<script>
// ─── Embedded data ──────────────────────────────────────────────────────────
const MECHS = {mechs_json};
const PMODS = {pmods_json};
const SACTS = {sacts_json};
const TRAIT_DIMS = {trait_json};
const STATE_DIMS = {state_json};
const SIT_FEATS = {feats_json};
const DOMAINS = {domain_json};
const DOMAIN_ORDER = {domain_order_json};

// ─── State ──────────────────────────────────────────────────────────────────
const profile = {{}};     // dim → '+' | '-'
const situation = new Set();
let expandedId = null;

// ─── Scoring (faithful port of mcp_server._score_mechanisms) ────────────────
const STRENGTH_WEIGHT = {{"strong": 1.5, "moderate": 1.0, "weak": 0.5}};
const SITUATION_MULTIPLIER = 0.5;

function scoreMechanisms(filterDomain) {{
  const results = [];
  const sitSet = situation;

  for (const mech of MECHS) {{
    if (filterDomain && mech.domain !== filterDomain) continue;

    const mid = mech.id;
    let personScore = 0;
    let situationScore = 0;
    const personMatches = [];
    const situationMatches = [];
    let excluded = false;

    // Person moderators
    const pms = PMODS[mid] || [];
    for (const pm of pms) {{
      const dim = pm.dim;
      if (!(dim in profile)) continue;
      const w = STRENGTH_WEIGHT[pm.str] || 1.0;
      const userDir = profile[dim];
      const mechDir = pm.dir;

      if (mechDir === "mixed") {{
        personScore += 0.25;
        personMatches.push({{dimension: dim, direction: "mixed", weight: 0.25, effect: "mixed"}});
      }} else if (userDir === mechDir) {{
        personScore += w;
        personMatches.push({{dimension: dim, direction: mechDir, weight: w, effect: "amplifies"}});
      }} else {{
        personScore -= w * 0.5;
        personMatches.push({{dimension: dim, direction: mechDir, weight: -w * 0.5, effect: "dampens"}});
      }}
    }}

    // Situation activators
    const sas = SACTS[mid] || [];
    for (const sa of sas) {{
      const feat = sa.feat;
      const effect = sa.eff;

      if (effect === "required") {{
        if (sitSet.has(feat)) {{
          situationScore += 2.0;
          situationMatches.push({{feature: feat, effect: "required+present", weight: 2.0}});
        }} else {{
          excluded = true;
          break;
        }}
      }} else if (sitSet.has(feat)) {{
        if (effect === "activates") {{
          situationScore += 2.0;
          situationMatches.push({{feature: feat, effect: "activates", weight: 2.0}});
        }} else if (effect === "amplifies") {{
          situationScore += 1.0;
          situationMatches.push({{feature: feat, effect: "amplifies", weight: 1.0}});
        }} else if (effect === "dampens") {{
          situationScore -= 1.0;
          situationMatches.push({{feature: feat, effect: "dampens", weight: -1.0}});
        }}
      }}
    }}

    if (excluded) continue;
    if (personScore <= 0) continue;
    const total = personScore * (1 + situationScore * SITUATION_MULTIPLIER);
    if (total <= 0) continue;

    results.push({{
      id: mid,
      name: mech.name,
      domain: mech.domain,
      description: mech.description,
      plo: mech.plo,
      accuracy: mech.accuracy,
      score: Math.round(total * 100) / 100,
      personScore: Math.round(personScore * 100) / 100,
      situationScore: Math.round(situationScore * 100) / 100,
      personMatches,
      situationMatches,
    }});
  }}

  results.sort((a, b) => b.score - a.score);
  return results.slice(0, 15);
}}

// ─── Verbalization preview ──────────────────────────────────────────────────
function getVerbalizationPrompt(mechResult) {{
  // Score posthoc_rationalization domain mechanisms
  const ratResults = [];
  const sitSet = situation;

  for (const mech of MECHS) {{
    if (mech.domain !== "posthoc_rationalization") continue;
    const mid = mech.id;
    let ps = 0, ss = 0;
    let excl = false;

    for (const pm of (PMODS[mid] || [])) {{
      if (!(pm.dim in profile)) continue;
      const w = STRENGTH_WEIGHT[pm.str] || 1.0;
      if (pm.dir === "mixed") ps += 0.25;
      else if (profile[pm.dim] === pm.dir) ps += w;
      else ps -= w * 0.5;
    }}

    for (const sa of (SACTS[mid] || [])) {{
      if (sa.eff === "required") {{
        if (!sitSet.has(sa.feat)) {{ excl = true; break; }}
        ss += 2.0;
      }} else if (sitSet.has(sa.feat)) {{
        ss += sa.eff === "activates" ? 2.0 : sa.eff === "amplifies" ? 1.0 : -1.0;
      }}
    }}

    if (!excl && ps > 0) {{
      ratResults.push({{
        id: mid, name: mech.name, score: ps * (1 + ss * SITUATION_MULTIPLIER),
        description: mech.description, plo: mech.plo,
      }});
    }}
  }}

  // Pick template
  ratResults.sort((a, b) => b.score - a.score);
  let template = ratResults[0]?.score > 0 ? ratResults[0] :
    ratResults.find(r => r.id === "self_serving_bias") || ratResults[0];

  if (!template) return null;

  const hiddenPlo = (mechResult.plo || []).slice(0, 6).map(p => `"${{p}}"`).join(", ");
  const ratPlo = (template.plo || []).slice(0, 6).map(p => `"${{p}}"`).join(", ");

  return {{
    template,
    prompt: `You are generating dialogue for a character study in behavioral psychology.

ACTUAL HIDDEN DRIVER
Mechanism: ${{mechResult.name}}
What it produces: ${{hiddenPlo}}
Core dynamic: ${{(mechResult.description || "").slice(0, 280)}}

ACTION TAKEN: [describe what the person did]

RATIONALIZATION TEMPLATE
The character processes this via: ${{template.name}}
Verbal patterns for this rationalization: ${{ratPlo}}
Core pattern: ${{(template.description || "").slice(0, 200)}}

TASK
Generate 4-5 first-person statements (using I/me/my) this person would actually say out loud. They sound sincere and self-justifying.

Rules:
- Do NOT name or reference the actual psychological mechanism
- The character is NOT self-aware about the rationalization
- Statements should sound genuine, not rehearsed
- Draw on the verbal pattern vocabulary above
- Each statement 1-2 sentences max

Output: a JSON array of strings only. No commentary.`
  }};
}}

// ─── Rendering ──────────────────────────────────────────────────────────────
function formatDim(dim) {{
  return dim.replace(/_/g, " ");
}}

function renderDimToggles(containerId, dims, group) {{
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  for (const [key, desc] of Object.entries(dims)) {{
    const row = document.createElement("div");
    row.className = "dim-row";
    row.innerHTML = `
      <span class="dim-label" title="${{desc}}">${{formatDim(key)}}</span>
      <div class="dim-toggle" data-dim="${{key}}" data-group="${{group}}" data-state="0"
           onclick="cycleDim(this)" title="Click to cycle: neutral → + → − → neutral">·</div>
    `;
    container.appendChild(row);
  }}
}}

function renderSitFeats() {{
  const container = document.getElementById("sit-feats");
  container.innerHTML = "";
  for (const [key, desc] of Object.entries(SIT_FEATS)) {{
    const row = document.createElement("div");
    row.className = "feat-row";
    row.innerHTML = `
      <input type="checkbox" id="feat-${{key}}" data-feat="${{key}}"
             onchange="toggleFeat(this)">
      <label for="feat-${{key}}" title="${{desc}}">${{formatDim(key)}}</label>
    `;
    container.appendChild(row);
  }}
}}

function renderDomainLegend() {{
  const container = document.getElementById("domain-legend");
  for (const d of DOMAIN_ORDER) {{
    const info = DOMAINS[d];
    const chip = document.createElement("span");
    chip.className = "domain-chip";
    chip.style.background = info.color;
    chip.textContent = info.label;
    chip.title = d;
    container.appendChild(chip);
  }}
}}

function renderResults() {{
  const results = scoreMechanisms(null);
  const listEl = document.getElementById("results-list");
  const countEl = document.getElementById("results-count");
  const hasInput = Object.keys(profile).length > 0 || situation.size > 0;

  if (!hasInput) {{
    countEl.textContent = "";
    listEl.innerHTML = `
      <div class="empty-state">
        <div class="icon">&#9881;</div>
        <p>Set some dimensions or situation features to see predictions.</p>
      </div>`;
    return;
  }}

  if (results.length === 0) {{
    countEl.textContent = "0 results";
    listEl.innerHTML = `
      <div class="empty-state">
        <div class="icon">&#8709;</div>
        <p>No mechanisms scored above zero for this profile and situation.</p>
      </div>`;
    return;
  }}

  countEl.textContent = `${{results.length}} result${{results.length !== 1 ? "s" : ""}}`;
  const maxScore = results[0].score;

  listEl.innerHTML = results.map((r, i) => {{
    const domInfo = DOMAINS[r.domain] || {{color: "#888", label: r.domain}};
    const pPct = maxScore > 0 ? (Math.max(0, r.personScore) / maxScore * 100) : 0;
    const sPct = maxScore > 0 ? (Math.max(0, r.situationScore) / maxScore * 100) : 0;

    // Chips
    const ampDims = r.personMatches.filter(m => m.effect === "amplifies");
    const dmpDims = r.personMatches.filter(m => m.effect === "dampens");
    const mixDims = r.personMatches.filter(m => m.effect === "mixed");
    const sitFeats = r.situationMatches.filter(m => !m.effect.includes("dampens"));
    const sitDamp = r.situationMatches.filter(m => m.effect.includes("dampens"));

    let chips = `<span class="chip chip-domain" style="background:${{domInfo.color}}">${{domInfo.label}}</span>`;
    for (const m of ampDims) chips += `<span class="chip chip-dim-pos">+${{formatDim(m.dimension)}}</span>`;
    for (const m of mixDims) chips += `<span class="chip chip-dim-mixed">~${{formatDim(m.dimension)}}</span>`;
    for (const m of sitFeats) chips += `<span class="chip chip-feat">${{formatDim(m.feature)}}</span>`;
    for (const m of dmpDims) chips += `<span class="chip chip-dim-neg">&minus;${{formatDim(m.dimension)}}</span>`;
    for (const m of sitDamp) chips += `<span class="chip chip-feat-damp">&minus;${{formatDim(m.feature)}}</span>`;

    const expanded = expandedId === r.id;

    // PLO list
    let ploHtml = "";
    if (r.plo && r.plo.length > 0) {{
      ploHtml = `<div class="detail-section"><h4>Plain-language outputs</h4>
        <ul class="plo-list">${{r.plo.map(p => `<li>${{escHtml(p)}}</li>`).join("")}}</ul></div>`;
    }}

    // Narrative outputs
    let narrHtml = "";
    if (r.narrative && r.narrative.length > 0) {{
      narrHtml = `<div class="detail-section"><h4>Narrative outputs</h4>
        <ul class="plo-list" style="font-style:italic">${{r.narrative.map(n => `<li>${{escHtml(n)}}</li>`).join("")}}</ul></div>`;
    }}

    // Verbalization section — show prompt immediately on expand
    let verbHtml = "";
    if (expanded) {{
      const verb = getVerbalizationPrompt(r);
      if (verb) {{
        verbHtml = `<div class="verb-section">
          <h4>Verbalization prompt</h4>
          <div style="margin-bottom:8px;font-size:12px;color:var(--text2)">
            Rationalization template: <strong>${{escHtml(verb.template.name)}}</strong>
            <span style="color:var(--text3)">(score: ${{verb.template.score.toFixed(1)}})</span>
          </div>
          <div class="verb-prompt">${{escHtml(verb.prompt)}}</div>
        </div>`;
      }}
    }}

    return `<div class="result-card${{expanded ? " expanded" : ""}}" data-id="${{r.id}}"
                 onclick="toggleExpand('${{r.id}}', event)">
      <div class="result-main">
        <span class="result-rank">${{i + 1}}</span>
        <div class="result-info">
          <div class="result-name" style="color:${{domInfo.color}}">${{escHtml(r.name)}}</div>
          <div class="result-chips">${{chips}}</div>
        </div>
        <div class="result-bar-wrap">
          <div class="result-bar">
            <div class="seg-person" style="width:${{pPct}}%;background:${{domInfo.color}}"></div>
            <div class="seg-situation" style="width:${{sPct}}%;background:${{domInfo.color}}"></div>
          </div>
          <span class="result-score" style="color:${{domInfo.color}}">${{r.score.toFixed(1)}}</span>
        </div>
      </div>
      <div class="result-detail">
        ${{r.description ? `<div class="detail-section"><h4>Description</h4><p style="font-size:12.5px;color:var(--text2)">${{escHtml(r.description.slice(0, 400))}}</p></div>` : ""}}
        ${{ploHtml}}
        ${{narrHtml}}
        ${{verbHtml}}
      </div>
    </div>`;
  }}).join("");
}}

function escHtml(s) {{
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}}

// ─── Event handlers ─────────────────────────────────────────────────────────
function cycleDim(el) {{
  const dim = el.dataset.dim;
  const states = ["0", "+", "-"];
  const cur = states.indexOf(el.dataset.state);
  const next = states[(cur + 1) % 3];
  el.dataset.state = next;
  el.textContent = next === "0" ? "·" : next;

  if (next === "0") {{
    delete profile[dim];
  }} else {{
    profile[dim] = next;
  }}
  update();
}}

function toggleFeat(el) {{
  const feat = el.dataset.feat;
  if (el.checked) situation.add(feat);
  else situation.delete(feat);
  update();
}}

function toggleExpand(id, event) {{
  expandedId = expandedId === id ? null : id;
  renderResults();
}}

function togglePanel(panelId, event) {{
  if (event.target.classList.contains("reset")) return;
  document.getElementById(panelId).classList.toggle("collapsed");
}}

function resetDims(group) {{
  document.querySelectorAll(`.dim-toggle[data-group="${{group}}"]`).forEach(el => {{
    el.dataset.state = "0";
    el.textContent = "·";
    delete profile[el.dataset.dim];
  }});
  update();
}}

function resetFeats() {{
  document.querySelectorAll('#sit-feats input[type="checkbox"]').forEach(el => {{
    el.checked = false;
  }});
  situation.clear();
  update();
}}

// ─── URL hash for shareable links ───────────────────────────────────────────
function syncHash() {{
  const parts = [];
  for (const [dim, dir] of Object.entries(profile)) {{
    parts.push(`d.${{dim}}=${{encodeURIComponent(dir)}}`);
  }}
  for (const feat of situation) {{
    parts.push(`s.${{feat}}`);
  }}
  history.replaceState(null, "", parts.length ? "#" + parts.join("&") : location.pathname);
}}

function loadHash() {{
  const hash = location.hash.slice(1);
  if (!hash) return;
  for (const part of hash.split("&")) {{
    if (part.startsWith("d.")) {{
      const [key, val] = part.slice(2).split("=");
      if (key && (val === "+" || val === encodeURIComponent("+") || val === "-" || val === encodeURIComponent("-"))) {{
        profile[key] = decodeURIComponent(val);
      }}
    }} else if (part.startsWith("s.")) {{
      const feat = part.slice(2);
      if (feat) situation.add(feat);
    }}
  }}
  // Sync UI toggles
  document.querySelectorAll(".dim-toggle").forEach(el => {{
    const dim = el.dataset.dim;
    if (dim in profile) {{
      el.dataset.state = profile[dim];
      el.textContent = profile[dim];
    }}
  }});
  document.querySelectorAll('#sit-feats input[type="checkbox"]').forEach(el => {{
    if (situation.has(el.dataset.feat)) el.checked = true;
  }});
}}

function update() {{
  expandedId = null;
  syncHash();
  renderResults();
}}

// ─── Init ───────────────────────────────────────────────────────────────────
renderDimToggles("trait-dims", TRAIT_DIMS, "trait");
renderDimToggles("state-dims", STATE_DIMS, "state");
renderSitFeats();
renderDomainLegend();
loadHash();
renderResults();
window.addEventListener("hashchange", () => {{
  // Reset and reload from hash
  Object.keys(profile).forEach(k => delete profile[k]);
  situation.clear();
  loadHash();
  renderResults();
}});
</script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate static HTML explorer")
    parser.add_argument("--output", default="docs/index.html")
    args = parser.parse_args()

    mechs, person_mods, sit_acts, dims = load_data()
    html = build_html(mechs, person_mods, sit_acts, dims)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)

    size_kb = out.stat().st_size / 1024
    print(f"Saved → {out}  ({size_kb:.0f} KB)")
    print(f"  {len(mechs)} mechanisms embedded")
    print(f"  {sum(len(v) for v in person_mods.values())} person moderators")
    print(f"  {sum(len(v) for v in sit_acts.values())} situation activators")


if __name__ == "__main__":
    main()
