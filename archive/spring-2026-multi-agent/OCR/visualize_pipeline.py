"""
Visualize benchmark results for AR Flashcards pipeline.
Usage (run from repo root):
  python OCR/visualize_pipeline.py
  python OCR/visualize_pipeline.py --csv OCR/results/e2e_agents.csv
"""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path

DEFAULT_RESULTS_DIR = "OCR/results"

def load_agent_csv(path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "agent":    row["agent"],
                "avg_ms":   float(row["avg_ms"]),
                "accuracy": float(row["accuracy"]),
                "fps":      float(row["fps"]),
                "avg_conf": float(row["avg_conf"]),
            })
    return rows

def find_latest_csv(d):
    csvs = sorted(d.glob("e2e_*_agents.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None

def generate_heatmap(rows, out_path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("  pip install matplotlib --break-system-packages"); return

    agents  = [r["agent"].split("(")[0].strip() for r in rows]
    metrics = ["Avg Latency (ms)", "Accuracy (%)", "Throughput (FPS)", "Confidence (%)"]
    # Direction note shown next to each metric label
    directions = ["↓ lower is better", "↑ higher is better",
                  "↑ higher is better", "↑ higher is better"]

    raw = np.array([
        [r["avg_ms"]         for r in rows],
        [r["accuracy"] * 100 for r in rows],
        [r["fps"]            for r in rows],
        [r["avg_conf"] * 100 for r in rows],
    ])

    # Normalize to [0,1] where 1 = best performance for every metric
    norm = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        mn, mx = raw[i].min(), raw[i].max()
        norm[i] = 0.5 if mx - mn < 1e-9 else (raw[i] - mn) / (mx - mn)
    norm[0] = 1.0 - norm[0]  # latency: lower ms = better

    cmap = plt.get_cmap("YlOrRd_r")  # yellow = best, dark red = worst

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": "#1A1A1A", "font.family": "sans-serif", "font.size": 12,
    })

    n_a, n_m = len(agents), len(metrics)
    fig, ax = plt.subplots(figsize=(max(9, n_a * 2.5), max(6, n_m * 1.6)))
    im = ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # --- X axis (agents) ---
    ax.set_xticks(range(n_a))
    ax.set_xticklabels(agents, fontsize=13, fontweight="bold", color="#1A1A1A")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel("Pipeline Agent", fontsize=12, labelpad=10, color="#1A1A1A")

    # --- Y axis (metrics + direction hint) ---
    y_labels = [f"{m}\n{d}" for m, d in zip(metrics, directions)]
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(y_labels, fontsize=10, color="#1A1A1A", linespacing=1.4)
    ax.set_ylabel("Benchmark Metric", fontsize=12, labelpad=12, color="#1A1A1A")
    ax.tick_params(length=0)

    # --- Cell value annotations ---
    fmt = [
        lambda v: f"{v:.0f} ms",
        lambda v: f"{v:.1f}%",
        lambda v: f"{v:.1f}" if v > 0 else "N/A",
        lambda v: f"{v:.1f}%" if v > 0 else "N/A",
    ]
    for i in range(n_m):
        for j in range(n_a):
            nval = norm[i, j]
            color = "#1A1A1A" if nval > 0.45 else "white"
            ax.text(j, i, fmt[i](raw[i, j]),
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    # --- Clean borders ---
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(n_a)  - 0.5, minor=True)
    ax.set_yticks(np.arange(n_m) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    # --- Colorbar — descriptive only, no fake numbers ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Relative Performance\n(per metric, normalized independently)",
                   fontsize=9, color="#444444", labelpad=10)
    cbar.set_ticks([0.03, 0.5, 0.97])
    cbar.set_ticklabels([
        "Worst\nperformance",
        "Average",
        "Best\nperformance",
    ], fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#888888", length=0)
    cbar.outline.set_edgecolor("#DDDDDD")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#444444", linespacing=1.4)

    # --- Title ---
    ax.set_title(
        "AR Flashcards — Agent Benchmark Heatmap",
        fontsize=15, fontweight="bold", pad=16, color="#1A1A1A",
    )
    fig.text(
        0.5, 0.98,
        f"{n_m} metrics  ×  {n_a} agents   |   "
        f"each metric normalized independently   |   yellow = best,  dark red = worst",
        ha="center", fontsize=8.5, color="#888888", style="italic",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Heatmap saved: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    csv_path    = Path(args.csv) if args.csv else find_latest_csv(results_dir)
    if not csv_path:
        print(f"  No CSVs found in {results_dir}. Run pipeline_benchmark.py first.")
        return

    stamp    = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else \
               csv_path.parent / f"e2e_{stamp}_heatmap.png"

    print(f"  CSV : {csv_path}")
    print(f"  Out : {out_path}\n")

    rows = load_agent_csv(csv_path)
    if rows:
        generate_heatmap(rows, out_path)

if __name__ == "__main__":
    main()