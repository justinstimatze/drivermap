#!/usr/bin/env python3
"""
visualize.py — Tufte-inspired heatmap of behavioral mechanisms.

Rows: mechanisms sorted by domain
Left panel : person moderator dimensions (sorted by coverage)
             domain color = amplifies (+)  |  steel blue = dampens (-)
             opacity encodes strength: strong > moderate > weak
Right panel: situation activators (sorted by coverage)
             domain color, opacity: required > activates > amplifies
             dampens shown in cool gray
Right edge : accuracy_score bar

Usage:
    python visualize.py
    python visualize.py --output viz/mechanisms.png --dpi 200
"""

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
DB_PATH = ROOT / "db" / "mechanisms.sqlite"

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
    "threat_affective_priming": (0.82, 0.22, 0.24),  # brick red
    "status_dominance": (0.52, 0.25, 0.72),  # purple
    "posthoc_rationalization": (0.88, 0.53, 0.10),  # amber
    "ingroup_outgroup": (0.22, 0.56, 0.28),  # forest green
    "social_influence_compliance": (0.12, 0.58, 0.58),  # teal
    "individual_variation": (0.16, 0.40, 0.70),  # steel blue
    "loss_aversion_reference": (0.68, 0.33, 0.14),  # sienna
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

# Direction → fill; strength → alpha
STRENGTH_A = {"strong": 0.92, "moderate": 0.60, "weak": 0.28}
NEG_RGB = (0.35, 0.45, 0.62)  # steel blue for "-" moderators
MIXED_RGB = (0.60, 0.60, 0.60)

# Effect → alpha; dampens uses a separate hue
EFFECT_A = {"required": 0.95, "activates": 0.72, "amplifies": 0.42, "dampens": 0.32}
DAMPENS_RGB = (0.30, 0.38, 0.55)

BG = "white"
GRID_COLOR = "#DDDDDD"
SEPARATOR_COLOR = "#888888"


def load_data(conn):
    mechs_by_domain = {}
    for d in DOMAIN_ORDER:
        rows = conn.execute(
            "SELECT id, name, accuracy_score FROM mechanisms WHERE domain=? ORDER BY name",
            (d,),
        ).fetchall()
        mechs_by_domain[d] = [dict(r) for r in rows]

    mech_list = []  # [(mech_dict, domain_str)]
    for d in DOMAIN_ORDER:
        for m in mechs_by_domain[d]:
            mech_list.append((m, d))

    # {mechanism_id: {dimension: (direction, strength)}}
    pm = {}
    for row in conn.execute(
        "SELECT mechanism_id, dimension, direction, strength FROM person_moderators"
    ):
        pm.setdefault(row["mechanism_id"], {})[row["dimension"]] = (
            row["direction"],
            row["strength"],
        )

    # {mechanism_id: {feature: effect}}
    sa = {}
    for row in conn.execute("SELECT mechanism_id, feature, effect FROM situation_activators"):
        sa.setdefault(row["mechanism_id"], {})[row["feature"]] = row["effect"]

    # Sort dimensions by how many mechanisms have them (desc)
    dim_cov = {}
    for mid_pm in pm.values():
        for dim in mid_pm:
            dim_cov[dim] = dim_cov.get(dim, 0) + 1
    dimensions = sorted(dim_cov, key=lambda d: -dim_cov[d])

    feat_cov = {}
    for mid_sa in sa.values():
        for feat in mid_sa:
            feat_cov[feat] = feat_cov.get(feat, 0) + 1
    features = sorted(feat_cov, key=lambda f: -feat_cov[f])

    return mech_list, pm, sa, dimensions, features, dim_cov, feat_cov, mechs_by_domain


def draw_cell(ax, col, row, rgb, alpha, cell_pad=0.07):
    """Draw a filled rectangle at grid position (col, row) [bottom-left origin]."""
    rect = plt.Rectangle(
        (col + cell_pad, row + cell_pad),
        1 - 2 * cell_pad,
        1 - 2 * cell_pad,
        color=rgb,
        alpha=alpha,
        zorder=2,
        linewidth=0,
    )
    ax.add_patch(rect)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="assets/mechanisms.png")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    mech_list, pm, sa, dimensions, features, dim_cov, feat_cov, mechs_by_domain = load_data(conn)

    n_mechs = len(mech_list)
    n_dims = len(dimensions)
    n_feats = len(features)

    # ── Layout constants ──────────────────────────────────────────────────────
    CELL = 0.26  # inches per cell (square)
    NAME_W = 2.9  # inches for mechanism name column
    DOMBAR_W = 0.10  # colored domain bar width
    GAP_W = 0.30  # gap between pm panel and sa panel
    SCORE_W = 0.55  # accuracy bar panel width
    COL_H = 1.70  # height reserved for column labels
    TITLE_H = 0.45  # title height
    GAP_TITLE = 0.60  # gap between column labels top and title
    LEGEND_H = 0.50  # legend height at bottom
    GAP_LEGEND = 0.60  # gap between grid bottom and legend
    PAD_R = 0.10  # right padding

    grid_w = n_dims * CELL  # width of pm panel in inches
    grid_w2 = n_feats * CELL  # width of sa panel in inches
    total_h = n_mechs * CELL  # height of cell grid

    fig_w = NAME_W + DOMBAR_W + grid_w + GAP_W + grid_w2 + SCORE_W + PAD_R
    fig_h = TITLE_H + GAP_TITLE + COL_H + total_h + GAP_LEGEND + LEGEND_H

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    # Helper: convert inches to figure fractions
    def ix(x):
        return x / fig_w

    def iy(y):
        return y / fig_h

    # Shared y-origin for the cell rows (from bottom)
    y0 = LEGEND_H + GAP_LEGEND  # bottom of cell grid in inches

    # ── Axes: main pm heatmap ─────────────────────────────────────────────────
    ax_pm = fig.add_axes(
        [
            ix(NAME_W + DOMBAR_W),
            iy(y0),
            ix(grid_w),
            iy(total_h),
        ]
    )
    ax_pm.set_xlim(0, n_dims)
    ax_pm.set_ylim(0, n_mechs)
    ax_pm.set_aspect("equal")
    ax_pm.axis("off")

    # ── Axes: situation activators heatmap ────────────────────────────────────
    ax_sa = fig.add_axes(
        [
            ix(NAME_W + DOMBAR_W + grid_w + GAP_W),
            iy(y0),
            ix(grid_w2),
            iy(total_h),
        ]
    )
    ax_sa.set_xlim(0, n_feats)
    ax_sa.set_ylim(0, n_mechs)
    ax_sa.set_aspect("equal")
    ax_sa.axis("off")

    # ── Axes: mechanism names + domain bar ────────────────────────────────────
    ax_names = fig.add_axes([0, iy(y0), ix(NAME_W + DOMBAR_W), iy(total_h)])
    ax_names.set_xlim(0, NAME_W + DOMBAR_W)
    ax_names.set_ylim(0, n_mechs)
    ax_names.axis("off")

    # ── Axes: accuracy score bar ──────────────────────────────────────────────
    ax_score = fig.add_axes(
        [
            ix(NAME_W + DOMBAR_W + grid_w + GAP_W + grid_w2),
            iy(y0),
            ix(SCORE_W),
            iy(total_h),
        ]
    )
    ax_score.set_xlim(0, 1)
    ax_score.set_ylim(0, n_mechs)
    ax_score.axis("off")

    # ── Axes: pm column labels ────────────────────────────────────────────────
    ax_pm_hdr = fig.add_axes(
        [
            ix(NAME_W + DOMBAR_W),
            iy(y0 + total_h),
            ix(grid_w),
            iy(COL_H),
        ]
    )
    ax_pm_hdr.set_xlim(0, n_dims)
    ax_pm_hdr.set_ylim(0, COL_H / CELL)
    ax_pm_hdr.axis("off")

    # ── Axes: sa column labels ────────────────────────────────────────────────
    ax_sa_hdr = fig.add_axes(
        [
            ix(NAME_W + DOMBAR_W + grid_w + GAP_W),
            iy(y0 + total_h),
            ix(grid_w2),
            iy(COL_H),
        ]
    )
    ax_sa_hdr.set_xlim(0, n_feats)
    ax_sa_hdr.set_ylim(0, COL_H / CELL)
    ax_sa_hdr.axis("off")

    # ═════════════════════════════════════════════════════════════════════════
    # Draw cells
    # ═════════════════════════════════════════════════════════════════════════
    for row_idx, (mech, domain) in enumerate(mech_list):
        mid = mech["id"]
        row = n_mechs - row_idx - 1  # y coordinate (top-to-bottom reading)
        dom_rgb = DOMAIN_RGB[domain]

        # Person moderators
        mid_pm = pm.get(mid, {})
        for col_idx, dim in enumerate(dimensions):
            if dim in mid_pm:
                direction, strength = mid_pm[dim]
                alpha = STRENGTH_A.get(strength or "moderate", 0.60)
                if direction == "+":
                    rgb = dom_rgb
                elif direction == "-":
                    rgb = NEG_RGB
                else:
                    rgb = MIXED_RGB
                    alpha = 0.35
                draw_cell(ax_pm, col_idx, row, rgb, alpha)

        # Situation activators
        mid_sa = sa.get(mid, {})
        for col_idx, feat in enumerate(features):
            if feat in mid_sa:
                effect = mid_sa[feat]
                alpha = EFFECT_A.get(effect, 0.42)
                rgb = DAMPENS_RGB if effect == "dampens" else dom_rgb
                draw_cell(ax_sa, col_idx, row, rgb, alpha)
                # Required: thin white border on top
                if effect == "required":
                    border = plt.Rectangle(
                        (col_idx + 0.07, row + 0.07),
                        0.86,
                        0.86,
                        fill=False,
                        edgecolor="white",
                        linewidth=0.8,
                        zorder=3,
                    )
                    ax_sa.add_patch(border)

    # ═════════════════════════════════════════════════════════════════════════
    # Domain separator lines and row background tints
    # ═════════════════════════════════════════════════════════════════════════
    current_row = 0
    for d_idx, domain in enumerate(DOMAIN_ORDER):
        n = len(mechs_by_domain[domain])
        # Faint alternating band
        if d_idx % 2 == 1:
            band_y = n_mechs - current_row - n
            for ax_ in (ax_pm, ax_sa):
                band = plt.Rectangle(
                    (0, band_y),
                    9999,
                    n,
                    color="#F5F5F5",
                    zorder=0,
                    linewidth=0,
                )
                ax_.add_patch(band)

        # Separator line between domains
        if current_row > 0:
            sep_y = n_mechs - current_row
            for ax_ in (ax_pm, ax_sa):
                ax_.axhline(sep_y, color=SEPARATOR_COLOR, linewidth=0.6, zorder=4, alpha=0.6)
            ax_names.axhline(sep_y, color=SEPARATOR_COLOR, linewidth=0.6, zorder=4, alpha=0.6)

        current_row += n

    # ═════════════════════════════════════════════════════════════════════════
    # Mechanism names + domain color bar
    # ═════════════════════════════════════════════════════════════════════════

    # One solid rectangle per domain block (no per-row gaps)
    current_row = 0
    for domain in DOMAIN_ORDER:
        n = len(mechs_by_domain[domain])
        band_y = n_mechs - current_row - n
        dom_rgb = DOMAIN_RGB[domain]
        ax_names.add_patch(
            plt.Rectangle(
                (NAME_W, band_y),
                DOMBAR_W,
                n,
                color=dom_rgb,
                alpha=0.90,
                zorder=2,
                linewidth=0,
            )
        )
        if n >= 3:
            ax_names.text(
                NAME_W + DOMBAR_W / 2,
                band_y + n / 2,
                DOMAIN_LABEL[domain].upper(),
                ha="center",
                va="center",
                fontsize=3.8,
                color="white",
                fontweight="bold",
                rotation=90,
                clip_on=True,
            )
        current_row += n

    # Mechanism names
    for row_idx, (mech, _domain) in enumerate(mech_list):
        row = n_mechs - row_idx - 1
        yc = row + 0.5
        name = mech["name"]
        if len(name) > 40:
            name = name[:38] + "…"
        ax_names.text(
            NAME_W - 0.08,
            yc,
            name,
            ha="right",
            va="center",
            fontsize=6.5,
            color="#1A1A1A",
            fontfamily="sans-serif",
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Column labels
    # ═════════════════════════════════════════════════════════════════════════
    COL_H_UNITS = COL_H / CELL

    # Section headers
    ax_pm_hdr.text(
        n_dims / 2,
        COL_H_UNITS - 0.4,
        "PERSON MODERATORS",
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color="#222222",
    )
    ax_sa_hdr.text(
        n_feats / 2,
        COL_H_UNITS - 0.4,
        "SITUATION ACTIVATORS",
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color="#222222",
    )

    # Dimension labels with coverage mini-bars
    max_dim_cov = max(dim_cov.values())
    for col_idx, dim in enumerate(dimensions):
        x = col_idx + 0.5
        label = dim.replace("_", " ")
        ax_pm_hdr.text(
            x,
            0.5,
            label,
            ha="left",
            va="bottom",
            fontsize=5.5,
            rotation=60,
            color="#333333",
        )
        # Coverage tick
        bar_h = (dim_cov[dim] / max_dim_cov) * 0.45
        ax_pm_hdr.add_patch(
            plt.Rectangle(
                (col_idx + 0.15, 0.05),
                0.7,
                bar_h,
                color="#AAAAAA",
                alpha=0.6,
                zorder=1,
            )
        )

    max_feat_cov = max(feat_cov.values())
    for col_idx, feat in enumerate(features):
        x = col_idx + 0.5
        label = feat.replace("_", " ")
        ax_sa_hdr.text(
            x,
            0.5,
            label,
            ha="left",
            va="bottom",
            fontsize=5.5,
            rotation=60,
            color="#333333",
        )
        bar_h = (feat_cov[feat] / max_feat_cov) * 0.45
        ax_sa_hdr.add_patch(
            plt.Rectangle(
                (col_idx + 0.15, 0.05),
                0.7,
                bar_h,
                color="#AAAAAA",
                alpha=0.6,
                zorder=1,
            )
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Accuracy score bars (right)
    # ═════════════════════════════════════════════════════════════════════════
    ax_score.axvline(0.05, color="#CCCCCC", linewidth=0.5, zorder=1)
    for row_idx, (mech, domain) in enumerate(mech_list):
        row = n_mechs - row_idx - 1
        score = mech["accuracy_score"] or 0.85
        dom_rgb = DOMAIN_RGB[domain]
        # Map 0.70–1.00 to 0–0.85 bar width
        bar_w = max(0, (score - 0.70) / 0.30) * 0.85
        ax_score.add_patch(
            plt.Rectangle(
                (0.05, row + 0.15),
                bar_w,
                0.70,
                color=dom_rgb,
                alpha=0.65,
                zorder=2,
                linewidth=0,
            )
        )
        ax_score.text(
            bar_w + 0.07,
            row + 0.50,
            f"{score:.2f}",
            ha="left",
            va="center",
            fontsize=4.0,
            color="#444444",
        )

    # Score header
    ax_score.text(
        0.40,
        n_mechs + 0.1,
        "accuracy",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#555555",
    )
    ax_score.text(
        0.05,
        n_mechs + 0.1,
        "0.70",
        ha="left",
        va="bottom",
        fontsize=4.0,
        color="#999999",
    )
    ax_score.text(
        0.90,
        n_mechs + 0.1,
        "1.0",
        ha="right",
        va="bottom",
        fontsize=4.0,
        color="#999999",
    )

    # ═════════════════════════════════════════════════════════════════════════
    # Legend
    # ═════════════════════════════════════════════════════════════════════════
    ax_legend = fig.add_axes([0, 0, 1, iy(LEGEND_H)])
    ax_legend.set_xlim(0, fig_w)
    ax_legend.set_ylim(0, LEGEND_H)
    ax_legend.axis("off")

    lx = NAME_W + DOMBAR_W  # start legend under pm panel
    ly = LEGEND_H * 0.65

    # Person moderator legend
    ax_legend.text(
        lx,
        ly + 0.12,
        "PERSON MODERATOR DIRECTION:",
        fontsize=5.5,
        fontweight="bold",
        color="#333333",
        va="bottom",
    )
    pm_items = [
        ("amplifies (+)", (0.60, 0.25, 0.65), 0.90),
        ("dampens (−)", NEG_RGB, 0.90),
        ("mixed", MIXED_RGB, 0.35),
    ]
    for i, (label, rgb, alpha) in enumerate(pm_items):
        bx = lx + i * 1.6
        ax_legend.add_patch(
            plt.Rectangle((bx, ly - 0.17), 0.28, 0.22, color=rgb, alpha=alpha, linewidth=0)
        )
        ax_legend.text(bx + 0.32, ly - 0.06, label, fontsize=5.0, color="#444444", va="center")

    # Strength legend
    bx0 = lx + 5.5
    ax_legend.text(
        bx0, ly + 0.12, "STRENGTH:", fontsize=5.5, fontweight="bold", color="#333333", va="bottom"
    )
    for i, (label, alpha) in enumerate([("strong", 0.92), ("moderate", 0.60), ("weak", 0.28)]):
        bx = bx0 + i * 1.3
        ax_legend.add_patch(
            plt.Rectangle(
                (bx, ly - 0.17), 0.28, 0.22, color=(0.52, 0.25, 0.72), alpha=alpha, linewidth=0
            )
        )
        ax_legend.text(bx + 0.32, ly - 0.06, label, fontsize=5.0, color="#444444", va="center")

    # Situation activator legend
    bx0 = lx + 9.8
    ax_legend.text(
        bx0,
        ly + 0.12,
        "SITUATION EFFECT:",
        fontsize=5.5,
        fontweight="bold",
        color="#333333",
        va="bottom",
    )
    sa_items = [
        ("required", (0.12, 0.58, 0.58), 0.95),
        ("activates", (0.12, 0.58, 0.58), 0.72),
        ("amplifies", (0.12, 0.58, 0.58), 0.42),
        ("dampens", DAMPENS_RGB, 0.32),
    ]
    for i, (label, rgb, alpha) in enumerate(sa_items):
        bx = bx0 + i * 1.4
        ax_legend.add_patch(
            plt.Rectangle((bx, ly - 0.17), 0.28, 0.22, color=rgb, alpha=alpha, linewidth=0)
        )
        ax_legend.text(bx + 0.32, ly - 0.06, label, fontsize=5.0, color="#444444", va="center")

    # ═════════════════════════════════════════════════════════════════════════
    # Title
    # ═════════════════════════════════════════════════════════════════════════
    fig.text(
        0.5,
        1.0 - 0.005,
        f"Behavioral Mechanisms Knowledge Base · {n_mechs} mechanisms · "
        f"{n_dims} person dimensions · {n_feats} situation features",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#111111",
    )
    fig.text(
        0.5,
        1.0 - 0.025,
        "Rows sorted by domain. "
        "Left panel: person moderator dimensions (coverage-sorted). "
        "Right panel: situation activators. "
        "Accuracy bar (0.70–1.00) at far right.",
        ha="center",
        va="top",
        fontsize=6,
        color="#666666",
    )

    # ═════════════════════════════════════════════════════════════════════════
    # Save
    # ═════════════════════════════════════════════════════════════════════════
    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out}  ({out.stat().st_size // 1024} KB)")
    plt.close()


def _pred_chip(ax, x, yc, text, bg, fg="white", fs=4.8, alpha=0.80, max_x=None):
    """Draw a rounded chip; returns new x position. Returns None if it would overflow."""
    tw = len(text) * 0.055 + 0.14
    if max_x is not None and x + tw > max_x:
        return None
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, yc - 0.13),
            tw,
            0.26,
            boxstyle="round,pad=0.018",
            color=bg,
            alpha=alpha,
            zorder=3,
            linewidth=0,
        )
    )
    ax.text(x + tw / 2, yc, text, ha="center", va="center", fontsize=fs, color=fg, zorder=4)
    return x + tw + 0.10


def render_prediction(
    results: list[dict],
    profile: dict,
    situation: list[str],
    output_path: str,
    title: str = None,
    dpi: int = 150,
):
    """
    Render a prediction query result as a horizontal bar chart.

    Two sub-lines per row:
      top    — rank · name · bar · score
      bottom — [domain]  +dim chips  situation chips  −dim chips
    Chips live in the bar panel's lower sub-line so they get full panel width.
    """
    if not results:
        print("No results to render.")
        return

    n = len(results)
    max_score = max(r["score"] for r in results)

    # ── Layout (inches) ───────────────────────────────────────────────────────
    LPAD = 0.22  # left/right page margin
    NAME_W = 3.20  # mechanism name column
    H_GAP = 0.22  # gap between name and bar columns
    BAR_W = 5.20  # bar + chip sub-line (chips use same full width)
    DOM_GAP = 0.22  # gap before domain sidebar
    DOM_W = 2.50  # domain sidebar (wide enough for full names)
    ROW_IN = 0.78  # inches per row (two sub-lines + breathing room)
    HDR_H = 1.55  # header: title + profile row + situation row + axis legend
    FTR_H = 0.25

    # Data-unit row layout  (row height = 1.0 data units)
    BAR_YC = 0.67  # bar centre in upper portion of row
    CHIP_YC = 0.24  # chip centre in lower portion
    BAR_H = 0.28  # bar height (leaving clear air above and below)
    BAR_XLIM = max_score * 1.18 + 0.6

    fig_w = LPAD + NAME_W + H_GAP + BAR_W + DOM_GAP + DOM_W + LPAD
    body_h = n * ROW_IN
    fig_h = HDR_H + body_h + FTR_H

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    fx = lambda x: x / fig_w
    fy = lambda y: y / fig_h
    y0 = FTR_H  # bottom of body in inches

    # ── Body axes ─────────────────────────────────────────────────────────────
    ax_n = fig.add_axes([fx(LPAD), fy(y0), fx(NAME_W), fy(body_h)])
    ax_n.set_xlim(0, NAME_W)
    ax_n.set_ylim(0, n)
    ax_n.axis("off")

    ax_b = fig.add_axes([fx(LPAD + NAME_W + H_GAP), fy(y0), fx(BAR_W), fy(body_h)])
    ax_b.set_xlim(0, BAR_XLIM)
    ax_b.set_ylim(0, n)
    ax_b.axis("off")

    ax_d = fig.add_axes(
        [fx(LPAD + NAME_W + H_GAP + BAR_W + DOM_GAP), fy(y0), fx(DOM_W), fy(body_h)]
    )
    ax_d.set_xlim(0, DOM_W)
    ax_d.set_ylim(0, n)
    ax_d.axis("off")

    # ── Header axis ───────────────────────────────────────────────────────────
    ax_h = fig.add_axes([0, fy(y0 + body_h), 1.0, fy(HDR_H)])
    ax_h.set_xlim(0, fig_w)
    ax_h.set_ylim(0, HDR_H)
    ax_h.axis("off")

    # ── Draw rows ─────────────────────────────────────────────────────────────
    for i, r in enumerate(results):
        y = n - i - 1
        dom = r["domain"]
        rgb = DOMAIN_RGB.get(dom, (0.5, 0.5, 0.5))
        ps = max(0.0, r["person_score"])
        ss = max(0.0, r["situation_score"])

        # Subtle alternating band (even rows = light)
        if i % 2 == 0:
            for ax_ in (ax_n, ax_b, ax_d):
                ax_.add_patch(
                    plt.Rectangle((0, y), 9999, 1, color="#F8F8F8", zorder=0, linewidth=0)
                )

        # ── Name column, top sub-line ─────────────────────────────────────
        ax_n.text(
            0.26, y + BAR_YC, f"{i + 1}.", ha="right", va="center", fontsize=6.5, color="#C0C0C0"
        )
        name = r["name"] if len(r["name"]) <= 42 else r["name"][:40] + "…"
        ax_n.text(
            0.34,
            y + BAR_YC,
            name,
            ha="left",
            va="center",
            fontsize=7.2,
            color=rgb,
            fontweight="bold",
        )

        # ── Name column, bottom sub-line — domain tag ─────────────────────
        dtag = f"[{dom.replace('_', ' ')}]"
        ax_n.text(0.34, y + CHIP_YC, dtag, ha="left", va="center", fontsize=4.8, color="#BBBBBB")

        # ── Bar (top sub-line) ────────────────────────────────────────────
        # Person segment
        ax_b.barh(y + BAR_YC, ps, height=BAR_H, color=rgb, alpha=0.88, zorder=2, left=0)
        # Situation segment (slightly thinner, lighter)
        if ss > 0:
            ax_b.barh(y + BAR_YC, ss, height=BAR_H * 0.72, color=rgb, alpha=0.40, zorder=2, left=ps)
        # Score label (right of bar)
        ax_b.text(
            ps + ss + 0.22,
            y + BAR_YC,
            f"{r['score']:.1f}",
            ha="left",
            va="center",
            fontsize=7.0,
            color="#444444",
            fontweight="bold",
        )
        # P/S inline labels (only if segment is wide enough)
        if ps > BAR_XLIM * 0.09:
            ax_b.text(
                ps / 2,
                y + BAR_YC,
                f"P {ps:.1f}",
                ha="center",
                va="center",
                fontsize=5.0,
                color="white",
                fontweight="bold",
                zorder=3,
            )
        if ss > BAR_XLIM * 0.07:
            ax_b.text(
                ps + ss / 2,
                y + BAR_YC,
                f"S {ss:.1f}",
                ha="center",
                va="center",
                fontsize=5.0,
                color=rgb,
                fontweight="bold",
                zorder=3,
            )

        # ── Chips (bottom sub-line, inside bar panel) ─────────────────────
        amp_dims = [m["dimension"] for m in r["person_matches"] if m.get("effect") == "amplifies"]
        dmp_dims = [m["dimension"] for m in r["person_matches"] if m.get("effect") == "dampens"]
        sit_feats = [
            m["feature"] for m in r["situation_matches"] if "dampens" not in m.get("effect", "")
        ]

        cx = 0.05
        limit = BAR_XLIM - 0.3

        for dim in amp_dims[:6]:
            lbl = "+" + dim.replace("_", " ")
            nx = _pred_chip(ax_b, cx, y + CHIP_YC, lbl, rgb, max_x=limit)
            if nx is None:
                break
            cx = nx

        for feat in sit_feats[:5]:
            lbl = feat.replace("_", " ")
            nx = _pred_chip(ax_b, cx, y + CHIP_YC, lbl, (0.12, 0.58, 0.58), max_x=limit)
            if nx is None:
                break
            cx = nx

        if dmp_dims:
            cx += 0.08  # small visual gap before dampening chips
            for dim in dmp_dims[:4]:
                lbl = "−" + dim.replace("_", " ")
                nx = _pred_chip(ax_b, cx, y + CHIP_YC, lbl, NEG_RGB, alpha=0.65, max_x=limit)
                if nx is None:
                    break
                cx = nx

        # Thin separator line between rows
        ax_b.axhline(y, color="#E8E8E8", linewidth=0.4, zorder=1)
        ax_n.axhline(y, color="#E8E8E8", linewidth=0.4, zorder=1)

    # ── Domain sidebar ────────────────────────────────────────────────────────
    dom_counts: dict[str, int] = {}
    for r in results:
        dom_counts[r["domain"]] = dom_counts.get(r["domain"], 0) + 1

    sorted_doms = sorted(dom_counts.items(), key=lambda x: -x[1])
    max_cnt = max(dom_counts.values()) if dom_counts else 1
    n_doms = len(sorted_doms)
    # Distribute entries evenly across body height (top-aligned with padding)
    step = max(1.10, n / max(n_doms, 1) * 0.80)
    dy = n - 0.70
    for dom, cnt in sorted_doms:
        rgb = DOMAIN_RGB.get(dom, (0.5, 0.5, 0.5))
        bar_w = cnt / max_cnt * (DOM_W - 0.60)  # longest bar fills ~76% of column
        # Color bar
        ax_d.add_patch(
            plt.Rectangle(
                (0.10, dy - 0.18), bar_w, 0.34, color=rgb, alpha=0.85, linewidth=0, zorder=2
            )
        )
        # Count badge (inside bar if wide enough, else to the right)
        if bar_w > 0.45:
            ax_d.text(
                0.10 + bar_w - 0.10,
                dy,
                str(cnt),
                ha="right",
                va="center",
                fontsize=7.0,
                color="white",
                fontweight="bold",
            )
        else:
            ax_d.text(
                0.10 + bar_w + 0.10,
                dy,
                str(cnt),
                ha="left",
                va="center",
                fontsize=7.0,
                color=rgb,
                fontweight="bold",
            )
        # Domain label (below bar, slightly larger)
        short = dom.replace("_", " ")
        ax_d.text(0.10, dy - 0.36, short, ha="left", va="top", fontsize=5.5, color="#666666")
        dy -= step

    # ── Header: title ─────────────────────────────────────────────────────────
    t = title or "Prediction"
    ax_h.text(
        LPAD,
        HDR_H - 0.12,
        t,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#111111",
    )

    # Score axis ticks (drawn at the bottom of the header, aligned to bar column)
    bar_x0 = LPAD + NAME_W + H_GAP  # left edge of bar column in figure inches
    tick_y = 0.48
    ax_h.plot(
        [bar_x0, bar_x0 + BAR_W],
        [tick_y, tick_y],
        color="#DDDDDD",
        linewidth=0.8,
        transform=ax_h.transData,
    )
    step = max(1, int(max_score / 5))
    for xv in range(0, int(BAR_XLIM) + 1, step):
        xpos = bar_x0 + xv / BAR_XLIM * BAR_W
        ax_h.plot([xpos, xpos], [tick_y - 0.05, tick_y + 0.0], color="#CCCCCC", linewidth=0.5)
        ax_h.text(
            xpos, tick_y + 0.06, str(xv), ha="center", va="bottom", fontsize=5.0, color="#AAAAAA"
        )

    # P/S legend in header (right of ticks)
    lx = bar_x0
    ly = tick_y - 0.30
    ax_h.add_patch(
        plt.Rectangle((lx, ly - 0.07), 0.32, 0.14, color=(0.5, 0.5, 0.5), alpha=0.88, linewidth=0)
    )
    ax_h.text(lx + 0.38, ly, "person score", ha="left", va="center", fontsize=5.5, color="#777777")
    lx += 1.55
    ax_h.add_patch(
        plt.Rectangle((lx, ly - 0.07), 0.32, 0.14, color=(0.5, 0.5, 0.5), alpha=0.38, linewidth=0)
    )
    ax_h.text(
        lx + 0.38, ly, "situation score", ha="left", va="center", fontsize=5.5, color="#777777"
    )

    # ── Header: profile chips ─────────────────────────────────────────────────
    chip_row_h = 0.36  # vertical space per chip row
    base_y = HDR_H - 0.52 - 0.0  # first chip row y-centre

    if profile:
        cy = base_y
        ax_h.text(
            LPAD,
            cy,
            "Profile:",
            ha="left",
            va="center",
            fontsize=6.5,
            color="#666666",
            fontweight="bold",
        )
        cx = LPAD + 0.72
        for dim, val in profile.items():
            lbl = f"{'+' if val == '+' else '−'} {dim.replace('_', ' ')}"
            bg = (0.22, 0.56, 0.28) if val == "+" else NEG_RGB
            nx = _pred_chip(ax_h, cx, cy, lbl, bg, fs=5.8, max_x=fig_w - LPAD)
            if nx is None:
                break
            cx = nx

    if situation:
        cy = base_y - chip_row_h * (1 if profile else 0) - chip_row_h * 0.10
        ax_h.text(
            LPAD,
            cy,
            "Situation:",
            ha="left",
            va="center",
            fontsize=6.5,
            color="#666666",
            fontweight="bold",
        )
        cx = LPAD + 0.88
        for feat in situation:
            lbl = feat.replace("_", " ")
            nx = _pred_chip(ax_h, cx, cy, lbl, (0.12, 0.58, 0.58), fs=5.8, max_x=fig_w - LPAD)
            if nx is None:
                break
            cx = nx

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
