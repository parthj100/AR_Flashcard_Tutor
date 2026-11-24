"""
Draw the energy graphs from energy.py's results (PNG, 16:9-friendly).

    .venv/bin/python benchmarks/ocr_eval/plots.py

Draws one graph per benchmarks/results/energy*.json:
    energy.json        -> energy.png         (first run: 60 rendered images)
    energy_pilot.json  -> energy_pilot.png   (73 real photos from data/pilot)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Validated two-series palette (blue / orange: CVD ΔE 24.7, all checks pass)
COLOR = {"easyocr": "#2a78d6", "pix2text": "#eb6834"}
LABEL = {"easyocr": "EasyOCR", "pix2text": "Pix2Text"}
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

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


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=200)
    plt.close(fig)
    print("wrote", out / name)


def fig_energy(e, out, name="energy.png"):
    """energy.json from energy.py: median joules per image, with the min-max range across rounds."""
    if e.get("fake_sampler"):
        print(f"{name}: data came from the fake power meter; skipping")
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
           f"Chip energy above idle (CPU + GPU + ANE), median of {e['repeats']} rounds; lines show the range. "
           f"Lower is better.\n{e['images']} images: {', '.join(e['types'])}.")
    _save(fig, out, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "benchmarks" / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "figures")
    a = ap.parse_args()
    files = sorted(a.results.glob("energy*.json"))
    if not files:
        raise SystemExit(f"no energy*.json in {a.results}; run energy.py first")
    for f in files:
        fig_energy(json.loads(f.read_text(encoding="utf-8")), a.out, f.stem + ".png")


if __name__ == "__main__":
    main()
