#!/usr/bin/env python3
"""Shared matplotlib style for every paper figure.

One palette, one font, one line weight, one legend convention.  Every fig_*.py does
    import figstyle as fs; fs.setup(); fig, ax = fs.figure("full", 2.4) ...; fs.save(fig, "fig3_cost")
and never sets colours or font sizes on its own.

Widths follow a 5.5 in text block: "full" = one text-width figure,
"half" = one of two side-by-side panels (3.3 in incl. gutter slack), "wide" = full width
but taller aspect budget.  Output is vector PDF (fonts embedded as Type 42) plus a 200 dpi
PNG preview for the readout page.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "paper", "figures"))

# ---- harness identity: fixed across all figures -------------------------------------------
HARNESS_ORDER = ["claude-code", "mini-swe-agent", "opencode"]
HARNESS_SHORT = {"claude-code": "CC", "mini-swe-agent": "mini", "opencode": "OC", "dsh": "dsh"}
EXTRA_HARNESSES = ["dsh"]   # vendor harness, DS-V4-Flash only; drawn after the three in Fig 4
HARNESS_LONG = {"claude-code": "Claude Code", "mini-swe-agent": "mini-SWE-agent", "opencode": "OpenCode"}
# Okabe–Ito (colour-blind safe, prints in greyscale as three distinct values)
HARNESS_COLOR = {"claude-code": "#0072B2",     # blue
                 "mini-swe-agent": "#E69F00",  # orange
                 "opencode": "#009E73"}        # bluish green
HARNESS_LIGHT = {"claude-code": "#9CC6E6", "mini-swe-agent": "#F3D08A", "opencode": "#8FD5BF"}

# ---- token / outcome segment colours (Fig 3, Fig 4) ----------------------------------------
TOKEN_COLOR = {"input_miss": "#3B3F58",   # input, cache miss   (dark slate)
               "input_hit": "#B9BDD3",    # input, cache hit    (light slate)
               "output": "#C8553D"}       # output incl. reasoning (brick)
TOKEN_LABEL = {"input_miss": "input · cache miss", "input_hit": "input · cache hit",
               "output": "output (incl. reasoning)"}
OUTCOME_ORDER = ["pass", "fail_patch", "early_stop", "overflow", "step_cap", "void"]
OUTCOME_COLOR = {"pass": "#3D8F5F",        # green
                 "fail_patch": "#C9CFD6",  # grey: tried, failed
                 "early_stop": "#E8A33D",  # amber: model stopped, no edit
                 "overflow": "#B5485D",    # wine: hit the context window
                 "step_cap": "#6B4E9B",    # violet: step / wall cap
                 "void": "#F2F2F2"}        # near white: infra, no outcome
OUTCOME_LABEL = {"pass": "pass", "fail_patch": "fail, patch submitted",
                 "early_stop": "early stop (no edit)", "overflow": "context overflow",
                 "step_cap": "step / time cap", "void": "infra void"}

MODEL_ORDER = ["qwen3.6-35b-a3b", "qwen3.8-27b", "ds-v4-flash", "glm-5.3-flash", "hy4-preview", "opus5"]
MODEL_LABEL = {"qwen3.6-35b-a3b": "Qwen3.6-35B-A3B", "qwen3.8-27b": "Qwen3.8-27B",
               "ds-v4-flash": "DeepSeek-V4-Flash", "glm-5.3-flash": "GLM-5.3-Flash",
               "hy4-preview": "HY4-Preview", "opus5": "Claude Opus 5"}
MODEL_SHORT = {"qwen3.6-35b-a3b": "Qwen3.6", "qwen3.8-27b": "Qwen3.8",
               "ds-v4-flash": "DS Flash", "glm-5.3-flash": "GLM Flash", "hy4-preview": "HY4",
               "opus5": "Opus 5"}

INK = "#222222"
MUTED = "#6B6B6B"
GRID = "#E3E3E3"
ZERO = "#111111"

WIDTH = {"full": 5.5, "half": 2.7, "wide": 5.5, "twothird": 3.6}


def _font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Lato", "Source Sans 3", "Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"):
        if cand in names:
            return cand
    return "DejaVu Sans"


def setup():
    fam = _font()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [fam, "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "patch.linewidth": 0.5,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 200,
        "figure.dpi": 110,
        "text.color": INK,
    })
    return fam


def figure(width="full", height=2.6, **kw):
    w = WIDTH.get(width, width)
    return plt.subplots(figsize=(w, height), **kw)


def despine(ax, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def hgrid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def note(ax, text, xy=(0.02, 0.98), ha="left", va="top", size=6.8, color=MUTED, **kw):
    """Explanatory text inside the figure whitespace (annotations live in the figure)."""
    return ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va,
                   fontsize=size, color=color, linespacing=1.25, **kw)


def panel_label(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom", color=INK)


def save(fig, name, png=True):
    os.makedirs(FIG_DIR, exist_ok=True)
    pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    if png:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    return pdf
