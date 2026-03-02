#!/usr/bin/env python3
"""
evolution.py — Logan Roy through Season 1: mechanism activation curves.

Traits stay fixed throughout.  State dimensions and situation features
shift across five episode moments.  Shows which mechanisms peak when he
is weakest vs. when he has fully reclaimed control.

Usage:
    python evolution.py
    python evolution.py --output assets/logan_evolution.png --dpi 180
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from mcp_server import _score_mechanisms, get_conn  # noqa: E402

# ── Fixed trait profile (stable across all of S1) ────────────────────────────

TRAITS = {
    "social_dominance_orientation": "+",  # hierarchy is the only real structure
    "dark_triad_narcissism": "+",  # Waystar is not a company he built — it is him
    "dark_triad_machiavellianism": "+",  # everyone in the room is a piece to be moved
    "attachment_avoidant": "+",  # closeness = leverage = threat
    "hexaco_H": "-",  # rules are for people he controls
    "bis_sensitivity": "-",  # threat → attack, not freeze
    "need_for_closure": "+",  # ambiguity about succession is intolerable
    "big_five_A": "-",  # does not accommodate, ever
}

# ── Five moments — state dims + situation features change per episode ─────────

MOMENTS = [
    {
        "label": "Ep 1\nBirthday party",
        "sublabel": "Collapses; kids\nsmell blood",
        "state": {
            "power_state": "-",
            "threat_appraisal": "+",
            "affective_valence": "-",
            "affective_arousal": "+",
        },
        "situation": [
            "stakes",
            "power_differential",
            "surveillance",
            "social_visibility",
            "prior_commitment",
        ],
    },
    {
        "label": "Ep 3\nHospitalized",
        "sublabel": "Concedes succession;\nreverses overnight",
        "state": {
            "power_state": "-",
            "threat_appraisal": "+",
            "affective_valence": "-",
            "affective_arousal": "+",
        },
        "situation": [
            "stakes",
            "power_differential",
            "surveillance",
            "prior_commitment",
        ],
    },
    {
        "label": "Ep 5\nBack at office",
        "sublabel": "Returns unannounced;\nKendall maneuvering",
        "state": {
            "power_state": "-",
            "threat_appraisal": "+",
        },
        "situation": [
            "stakes",
            "conflict_present",
            "power_differential",
            "social_visibility",
        ],
    },
    {
        "label": "Ep 7–8\nConsolidating",
        "sublabel": "Deal in play;\ngrip tightening",
        "state": {
            "threat_appraisal": "+",
        },
        "situation": [
            "stakes",
            "prior_commitment",
            "social_visibility",
        ],
    },
    {
        "label": 'Ep 10\n"No deal"',
        "sublabel": "Kills the GoJo sale;\nfully dominant again",
        "state": {
            "power_state": "+",
        },
        "situation": [
            "stakes",
            "social_visibility",
            "prior_commitment",
        ],
    },
]

TOP_N = 15  # fetch this many per moment so scores exist for near-miss mechanisms

# ─────────────────────────────────────────────────────────────────────────────


def build_profile(state: dict) -> dict:
    return {**TRAITS, **state}


def spread_labels(
    items: list[tuple[float, str]], min_gap: float = 0.55
) -> list[tuple[float, float, str]]:
    """
    Given [(y_actual, name), ...] sorted ascending, return [(y_actual, y_label, name)]
    with y_label spread so no two are closer than min_gap.
    """
    items = sorted(items, key=lambda t: t[0])
    adjusted = [(y, y, name) for y, name in items]

    # Forward pass: push up
    for i in range(1, len(adjusted)):
        y_act, y_lab, name = adjusted[i]
        prev_lab = adjusted[i - 1][1]
        if y_lab < prev_lab + min_gap:
            adjusted[i] = (y_act, prev_lab + min_gap, name)

    # Backward pass: pull down
    for i in range(len(adjusted) - 2, -1, -1):
        y_act, y_lab, name = adjusted[i]
        next_lab = adjusted[i + 1][1]
        if y_lab > next_lab - min_gap:
            adjusted[i] = (y_act, next_lab - min_gap, name)

    return adjusted


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="assets/logan_evolution.png")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    conn = get_conn()

    # ── Collect scores across moments ────────────────────────────────────────
    all_scores: dict[str, list] = {}  # name → [score | None] * n_moments
    all_domains: dict[str, str] = {}
    moment_results = []

    for moment in MOMENTS:
        profile = build_profile(moment["state"])
        results = _score_mechanisms(conn, profile, moment["situation"], top_n=TOP_N)
        moment_results.append(results)
        for r in results:
            name = r["name"]
            all_domains[name] = r["domain"]
            if name not in all_scores:
                all_scores[name] = [None] * len(MOMENTS)

    for m_idx, results in enumerate(moment_results):
        for r in results:
            all_scores[r["name"]][m_idx] = r["score"]

    # Featured = mechanisms in top-5 for at least one moment
    featured: set[str] = set()
    for results in moment_results:
        for r in results[:5]:
            featured.add(r["name"])

    def peak(name):
        vals = [v for v in all_scores[name] if v is not None]
        return max(vals) if vals else 0.0

    featured_sorted = sorted(featured, key=peak, reverse=True)

    print(f"\nTracking {len(featured_sorted)} mechanisms across {len(MOMENTS)} moments\n")
    header = "  {:<48}".format("Mechanism") + "  ".join(
        f"Ep{i + 1:>2}" for i in range(len(MOMENTS))
    )
    print(header)
    print("  " + "─" * (len(header) - 2))
    for name in featured_sorted:
        scores = all_scores[name]
        cols = [f"{s:>4.1f}" if s is not None else "  — " for s in scores]
        print(f"  {name:<48}  {'  '.join(cols)}")
    print()

    # ── Layout ───────────────────────────────────────────────────────────────
    n_moments = len(MOMENTS)
    # Right portion of the x-axis is blank label space (data lives at x 0..n-1)
    X_LABEL_START = n_moments - 0.55  # where leader lines end
    X_RIGHT = n_moments + 3.5  # x-axis limit; labels fill the gap
    fig_w, fig_h = 12.5, 7.5
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[5.5, 2.0],
        hspace=0.0,
        left=0.06,
        right=0.96,
        top=0.90,
        bottom=0.03,
    )
    ax_main = fig.add_subplot(gs[0])
    ax_ann = fig.add_subplot(gs[1])

    # ── Main line chart ───────────────────────────────────────────────────────
    ax_main.set_facecolor("white")
    ax_main.set_xlim(-0.35, X_RIGHT)

    all_vals = [v for scores in all_scores.values() for v in scores if v is not None]
    max_y = max(all_vals) if all_vals else 12.0
    ax_main.set_ylim(0, max_y * 1.08)
    ax_main.set_xticks([])

    # Minimal y-axis: a few reference ticks, no label clutter
    tick_step = 4 if max_y > 8 else 2
    ax_main.set_yticks(range(0, int(max_y) + tick_step, tick_step))
    ax_main.yaxis.set_tick_params(labelsize=6.5, color="#CCCCCC")
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["bottom"].set_visible(False)
    ax_main.spines["left"].set_color("#DDDDDD")
    ax_main.tick_params(axis="y", color="#DDDDDD")

    # Gridlines only over data range (not into label space)
    for yv in range(0, int(max_y) + tick_step, tick_step):
        ax_main.axhline(
            yv,
            color="#F2F2F2",
            linewidth=0.5,
            zorder=0,
            xmin=0,
            xmax=(n_moments - 0.65 + 0.35) / (X_RIGHT + 0.35),
        )
    for xi in range(n_moments):
        ax_main.axvline(xi, color="#EBEBEB", linewidth=0.8, zorder=0)

    # One unique color per mechanism (tab20 = 20 perceptually distinct hues)
    cmap = matplotlib.colormaps["tab20"]
    mechanism_color = {
        name: cmap(i / max(len(featured_sorted) - 1, 1)) for i, name in enumerate(featured_sorted)
    }

    # Track rightmost point per mechanism for label anchoring
    right_edge_points: list[tuple[float, str]] = []

    for name in featured_sorted:
        scores = all_scores[name]
        rgb = mechanism_color[name]

        # Plot connected segments; None = gap, line breaks there
        xs_seg, ys_seg = [], []
        for xi, sc in enumerate(scores):
            if sc is not None:
                xs_seg.append(xi)
                ys_seg.append(sc)
            else:
                if xs_seg:
                    ax_main.plot(
                        xs_seg,
                        ys_seg,
                        color=rgb,
                        linewidth=1.8,
                        alpha=0.9,
                        marker="o",
                        markersize=5.0,
                        markerfacecolor=rgb,
                        markeredgecolor="white",
                        markeredgewidth=0.9,
                        zorder=3,
                        solid_capstyle="round",
                    )
                xs_seg, ys_seg = [], []
        if xs_seg:
            ax_main.plot(
                xs_seg,
                ys_seg,
                color=rgb,
                linewidth=1.8,
                alpha=0.9,
                marker="o",
                markersize=5.0,
                markerfacecolor=rgb,
                markeredgecolor="white",
                markeredgewidth=0.9,
                zorder=3,
                solid_capstyle="round",
            )

        last_valid = max(
            (xi for xi, sc in enumerate(scores) if sc is not None),
            default=None,
        )
        if last_valid is not None:
            right_edge_points.append((scores[last_valid], name))

    # ── Right-side labels (spread, fully inside xlim so no clipping) ─────────
    spread = spread_labels(right_edge_points, min_gap=0.52)

    for y_actual, y_label, name in spread:
        rgb = mechanism_color[name]
        last_x = max(xi for xi, sc in enumerate(all_scores[name]) if sc is not None)

        # Thin leader: endpoint dot → label
        ax_main.plot(
            [last_x, X_LABEL_START],
            [y_actual, y_label],
            color=rgb,
            linewidth=0.5,
            alpha=0.45,
            zorder=2,
        )
        ax_main.text(
            X_LABEL_START + 0.08,
            y_label,
            name,
            ha="left",
            va="center",
            fontsize=6.2,
            color=rgb,
            fontweight="bold",
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(
        0.06,
        0.94,
        "Logan Roy — Mechanism Activation Across Season 1",
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color="#111111",
    )
    fig.text(
        0.06,
        0.915,
        "Traits fixed.  State dimensions and situation features shift by episode.",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color="#888888",
    )

    # ── Annotation strip ─────────────────────────────────────────────────────
    ax_ann.set_facecolor("#F7F7F7")
    ax_ann.set_xlim(-0.35, X_RIGHT)
    ax_ann.set_ylim(0, 1)
    ax_ann.axis("off")

    for xi, moment in enumerate(MOMENTS):
        # Vertical rule at each episode
        ax_ann.axvline(xi, color="#DDDDDD", linewidth=0.8)

        # Episode label
        ax_ann.text(
            xi,
            0.93,
            moment["label"],
            ha="center",
            va="top",
            fontsize=7.5,
            fontweight="bold",
            color="#222222",
            multialignment="center",
        )
        ax_ann.text(
            xi,
            0.52,
            moment["sublabel"],
            ha="center",
            va="top",
            fontsize=5.8,
            color="#777777",
            style="italic",
            multialignment="center",
        )

        # State dimension chips
        chips = []
        for dim, val in moment["state"].items():
            sign = "+" if val == "+" else "−"
            short = dim.replace("affective_", "").replace("_", " ")
            chips.append((f"{sign} {short}", val == "+"))

        n_chips = len(chips)
        chip_w = 0.28
        chip_gap = 0.04
        total_chip_w = n_chips * chip_w + (n_chips - 1) * chip_gap
        chip_x0 = xi - total_chip_w / 2

        for ci, (label, is_pos) in enumerate(chips):
            cx = chip_x0 + ci * (chip_w + chip_gap)
            color = "#228822" if is_pos else "#CC3333"
            ax_ann.add_patch(
                plt.Rectangle(
                    (cx, 0.06),
                    chip_w,
                    0.15,
                    facecolor=color,
                    alpha=0.15,
                    edgecolor=color,
                    linewidth=0.5,
                    zorder=2,
                )
            )
            ax_ann.text(
                cx + chip_w / 2,
                0.135,
                label,
                ha="center",
                va="center",
                fontsize=4.2,
                color=color,
                fontweight="bold",
                zorder=3,
            )

    # ── Save ─────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    conn.close()
    kb = out.stat().st_size // 1024
    print(f"Saved → {out}  ({kb} KB)")
    plt.close()


if __name__ == "__main__":
    run()
