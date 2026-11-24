"""
Step 3: turn summary.json into slide-ready graphs (PNG, 16:9-friendly).

    .venv/bin/python benchmarks/ocr_eval/plots.py
    .venv/bin/python benchmarks/ocr_eval/plots.py --summary benchmarks/results/update2/summary.json \\
                                                 --out benchmarks/figures/update2

Graphs (each skipped if the summary lacks its data):
    error_free.png        error-free rate (main measure) by photo group
    cer_by_type.png       character error rate by photo type
    time_by_type.png      seconds per photo by photo type
    hidden_errors.png     H4: low-CER readings that still had a critical error
    error_types.png       which kinds of critical errors each model made
    cer_vs_critical.png   every reading: CER vs. critical errors (H4, per reading)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.ticker import PercentFormatter      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Validated two-series palette (blue / orange: CVD ΔE 24.7, all checks pass)
COLOR = {"easyocr": "#2a78d6", "pix2text": "#eb6834"}
LABEL = {"easyocr": "EasyOCR", "pix2text": "Pix2Text"}
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CLEAN, CRITICAL = "#c3c2b7", "#d03b3b"

plt.rcParams.update({
    "font.family": ["Arial", "Arial Unicode MS", "DejaVu Sans"],
    "font.size": 13,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": MUTED, "xtick.labelsize": 13, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 13,
})
SIZE = (11, 6)


def _title(fig, title, subtitle=None):
    h = fig.get_figheight()                     # place by inches so short figures don't collide
    fig.text(0.04, 1 - 0.3 / h, title, fontsize=20, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.04, 1 - 0.72 / h, subtitle, fontsize=13, color=INK2, va="top")


def _legend(ax, models):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in models]
    ax.legend(handles, [LABEL.get(m, m) for m in models], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=len(models), handlelength=1.1, borderaxespad=0.2)


def _grouped_bars(ax, groups, models, values, fmt, sublabels=None):
    """values[g][m] -> bar height (None = no bar). 2px surface gap between bars."""
    n = len(models)
    width = 0.78 / n
    for j, m in enumerate(models):
        xs = [i + (j - (n - 1) / 2) * width for i in range(len(groups))]
        for x, g in zip(xs, groups):
            v = values[g].get(m)
            if v is None:
                continue
            ax.bar(x, v, width, color=COLOR[m], edgecolor=SURFACE, linewidth=2)
            txt = fmt(v)
            if sublabels and sublabels[g].get(m):
                txt += f"\n{sublabels[g][m]}"
            ax.annotate(txt, (x, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12, color=INK, linespacing=1.15)
    ax.set_xticks(range(len(groups)), groups)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.spines["bottom"].set_color(AXIS)


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=200)
    plt.close(fig)
    print("wrote", out / name)


def fig_error_free(s, out):
    models = s["models"]
    groups = [g for g in s["error_free"] if any(n for _, n in s["error_free"][g].values())]
    vals = {g: {m: (k / n if n else None) for m, (k, n) in s["error_free"][g].items()} for g in groups}
    subs = {g: {m: f"{k}/{n}" for m, (k, n) in s["error_free"][g].items() if n} for g in groups}
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.07, right=0.98)
    _grouped_bars(ax, groups, models, vals, lambda v: f"{v:.0%}", subs)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _legend(ax, models)
    _title(fig, "Share of images read with zero critical errors",
           s.get("error_free_subtitle", "Higher is better. A critical error changes a number or math symbol."))
    _save(fig, out, "error_free.png")


def fig_by_type(s, out, key, name, title, subtitle, fmt, ymax_pad=1.18, yfmt=None):
    data = s.get(key) or {}
    if not data:
        return
    models = s["models"]
    groups = list(data)
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.07, right=0.98)
    _grouped_bars(ax, groups, models, data, fmt)
    top = max(v for g in data.values() for v in g.values())
    ax.set_ylim(0, top * ymax_pad)
    if yfmt:
        ax.yaxis.set_major_formatter(yfmt)
    _legend(ax, models)
    _title(fig, title, subtitle)
    _save(fig, out, name)


def fig_hidden(s, out):
    h = s.get("hidden")
    if not h or not h.get("low_cer"):
        return
    n, k = h["low_cer"], h["with_critical"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    fig.subplots_adjust(top=0.70, bottom=0.12, left=0.04, right=0.96)
    ax.barh(0, n - k, color=CLEAN, edgecolor=SURFACE, linewidth=2, height=0.6)
    ax.barh(0, k, left=n - k, color=CRITICAL, edgecolor=SURFACE, linewidth=2, height=0.6)
    ax.text((n - k) / 2, 0, f"{n - k} had every number and symbol right", ha="center", va="center",
            color=INK, fontsize=14)
    ax.text(n - k / 2, 0, f"{k}", ha="center", va="center", color="#ffffff", fontsize=15, fontweight="bold")
    ax.annotate(f"⚠ {k} of {n} ({k / n:.0%}) still had a wrong\nor extra number or symbol",
                (n - k / 2, 0.3), xytext=(n - k / 2, 0.62), ha="right", va="bottom", fontsize=13, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_xlim(0, n)
    ax.set_ylim(-0.5, 1.15)
    ax.axis("off")
    _title(fig, "A low character error rate can hide a wrong number",
           f"Readings with under {h['cer_threshold']:.0%} of characters wrong, both models, "
           f"{s.get('pool_label', 'real photos')}")
    _save(fig, out, "hidden_errors.png")


def fig_error_types(s, out):
    cats = s.get("categories")
    if not cats or not any(sum(v.values()) for v in cats.values()):
        return
    models = s["models"]
    names = list(next(iter(cats.values())))
    data = {c: {m: cats[m][c] for m in models} for c in names}
    fig_by_type({"models": models, "x": data}, out, "x", "error_types.png",
                "What kind of critical errors each model made",
                f"Count across all {s.get('pool_label', 'real photos')}. Lower is better.", lambda v: f"{v:d}")


def fig_scatter(s, out):
    pts = s.get("points")
    if not pts:
        return
    rnd = random.Random(0)
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.08, right=0.98)
    thr = s["hidden"]["cer_threshold"]
    ax.axvspan(0, thr, color=GRID, alpha=0.6, lw=0)
    for m in s["models"]:
        p = [q for q in pts if q["model"] == m]
        ax.scatter([min(q["cer"], 1.0) for q in p], [q["critical"] + rnd.uniform(-.15, .15) for q in p],
                   s=64, color=COLOR[m], edgecolor=SURFACE, linewidth=1.5, alpha=0.9, label=LABEL[m])
    ax.set_xlim(left=0)
    ax.set_ylim(-0.5, max(q["critical"] for q in pts) + 0.8)
    ax.text(thr / 2, ax.get_ylim()[1], f"CER < {thr:.0%}", ha="center", va="top", color=INK2, fontsize=12)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Character error rate (capped at 100%)")
    ax.set_ylabel("Critical errors in the reading")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.3)
    _title(fig, "Every reading: character errors vs. critical errors",
           "Points in the shaded band look accurate but can still have wrong numbers")
    _save(fig, out, "cer_vs_critical.png")


def make_all(s, out: Path):
    fig_error_free(s, out)
    fig_by_type(s, out, "cer", "cer_by_type.png", "Character error rate by image type",
                s.get("cer_subtitle", "Share of letters and symbols read wrong. Lower is better."),
                lambda v: f"{v:.1%}", yfmt=PercentFormatter(1.0, decimals=0))
    fig_by_type(s, out, "time_s", "time_by_type.png", "Time to read one image",
                s.get("time_subtitle", "Seconds per image on CPU. Lower is better."),
                lambda v: f"{v:.2f} s" if v < 1 else f"{v:.1f} s")
    fig_hidden(s, out)
    fig_error_types(s, out)
    fig_scatter(s, out)


def fig_energy(e, out):
    """energy.json from energy.py: median joules per image, with the min-max range across rounds."""
    if e.get("fake_sampler"):
        print("energy.json came from the fake power meter; skipping energy.png")
        return
    models = [m for m in COLOR if m in e["summary"]]
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.07, right=0.98)
    for i, m in enumerate(models):
        s = e["summary"][m]
        med, lo, hi = s["joules_per_image_median"], s["joules_per_image_min"], s["joules_per_image_max"]
        ax.bar(i, med, 0.55, color=COLOR[m], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], color=INK2, capsize=8, lw=1.5)
        ax.annotate(f"{med:.2f} J" if med < 10 else f"{med:.1f} J", (i, hi), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom", fontsize=13, color=INK)
    ax.set_xticks(range(len(models)), [LABEL[m] for m in models])
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_xlim(-0.7, len(models) - 0.3)
    ax.set_ylim(0, max(e["summary"][m]["joules_per_image_max"] for m in models) * 1.2)
    ax.set_ylabel("Joules per image")
    _title(fig, "Energy to read one image",
           f"Chip energy above idle (CPU + GPU + ANE), median of {e['repeats']} rounds over "
           f"{e['images']} images; lines show the range. Lower is better.")
    _save(fig, out, "energy.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=ROOT / "benchmarks" / "results" / "summary.json")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "figures")
    a = ap.parse_args()
    make_all(json.loads(a.summary.read_text(encoding="utf-8")), a.out)
    energy = a.summary.parent / "energy.json"
    if energy.exists():
        fig_energy(json.loads(energy.read_text(encoding="utf-8")), a.out)


if __name__ == "__main__":
    main()
