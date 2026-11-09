"""
Visualize YOLO benchmark results for AR Flashcards.

Generates two charts from YOLOv8-Detection/results/:

  1. Per-class heatmap (benchmark_results.csv)
     — latency, FPS, avg detections, confidence per object class

  2. Model comparison bar chart (compare_*_summary.csv)
     — YOLOv8n vs YOLOv8s across avg ms, p50, p95, FPS, confidence

Usage (run from repo root):
  python YOLOv8-Detection/visualize_yolo.py

Output saved to YOLOv8-Detection/results/:
  yolo_class_heatmap.png
  yolo_model_comparison.png
"""
from __future__ import annotations
import csv, time
from pathlib import Path

RESULTS_DIR    = Path("YOLOv8-Detection/results")
BENCHMARK_CSV  = RESULTS_DIR / "benchmark_results.csv"
COMPARE_GLOB   = "compare_*_summary.csv"

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_benchmark(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["class"] == "OVERALL":
                continue
            rows.append({
                "class":    row["class"].replace("_", " ").title(),
                "avg_ms":   float(row["avg_ms"]),
                "fps":      float(row["fps"]),
                "avg_dets": float(row["avg_dets"]) if row["avg_dets"] else 0,
                "avg_conf": float(row["avg_conf"]) * 100,
            })
    return rows


def load_summary(results_dir: Path) -> list[dict]:
    csvs = sorted(results_dir.glob(COMPARE_GLOB),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        return []
    rows = []
    with csvs[0].open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "model":   row["model"].replace(".pt", ""),
                "avg_ms":  float(row["avg_ms"]),
                "p50_ms":  float(row["p50_ms"]),
                "p95_ms":  float(row["p95_ms"]),
                "fps":     float(row["fps"]),
                "avg_conf": float(row["avg_conf"]) * 100,
            })
    return rows

# ---------------------------------------------------------------------------
# Chart 1 — Per-class heatmap
# ---------------------------------------------------------------------------

def plot_class_heatmap(rows: list[dict], out_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    classes = [r["class"] for r in rows]
    metrics = ["Avg Latency (ms)", "Throughput (FPS)",
               "Avg Detections", "Confidence (%)"]
    directions = ["↓ lower is better", "↑ higher is better",
                  "↑ higher is better", "↑ higher is better"]

    raw = np.array([
        [r["avg_ms"]   for r in rows],
        [r["fps"]      for r in rows],
        [r["avg_dets"] for r in rows],
        [r["avg_conf"] for r in rows],
    ])  # shape: (4 metrics, n_classes)

    # Normalize each metric to [0,1] where 1 = best
    norm = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        mn, mx = raw[i].min(), raw[i].max()
        norm[i] = 0.5 if mx - mn < 1e-9 else (raw[i] - mn) / (mx - mn)
    norm[0] = 1.0 - norm[0]  # latency: lower = better

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": "#1A1A1A", "font.family": "sans-serif", "font.size": 11,
    })

    n_c, n_m = len(classes), len(metrics)
    fig, ax = plt.subplots(figsize=(max(10, n_c * 1.6), max(5, n_m * 1.6)))
    cmap = plt.get_cmap("YlOrRd_r")
    im   = ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # X axis — classes
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold",
                       color="#1A1A1A", rotation=15, ha="right")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel("Object Class", fontsize=12, labelpad=10, color="#1A1A1A")

    # Y axis — metrics + direction
    y_labels = [f"{m}\n{d}" for m, d in zip(metrics, directions)]
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(y_labels, fontsize=10, color="#1A1A1A", linespacing=1.4)
    ax.set_ylabel("Benchmark Metric", fontsize=12, labelpad=12, color="#1A1A1A")
    ax.tick_params(length=0)

    # Cell annotations
    fmt = [
        lambda v: f"{v:.0f} ms",
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.1f}%",
    ]
    for i in range(n_m):
        for j in range(n_c):
            nval  = norm[i, j]
            color = "#1A1A1A" if nval > 0.45 else "white"
            ax.text(j, i, fmt[i](raw[i, j]),
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(n_c) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_m) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Relative Performance\n(per metric, normalized independently)",
                   fontsize=9, color="#444444", labelpad=10)
    cbar.set_ticks([0.03, 0.5, 0.97])
    cbar.set_ticklabels(["Worst\nperformance", "Average", "Best\nperformance"],
                        fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#888888", length=0)
    cbar.outline.set_edgecolor("#DDDDDD")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#444444", linespacing=1.4)

    ax.set_title("YOLOv8n — Per-Class Benchmark Heatmap",
                 fontsize=14, fontweight="bold", pad=16, color="#1A1A1A")
    fig.text(0.5, 0.98,
             f"{n_m} metrics  ×  {n_c} object classes   |   "
             f"yellow = best,  dark red = worst",
             ha="center", fontsize=8.5, color="#888888", style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Class heatmap saved: {out_path}")

# ---------------------------------------------------------------------------
# Chart 2 — Model comparison grouped bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(rows: list[dict], out_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    models  = [r["model"] for r in rows]
    metrics = ["Avg Latency (ms)", "p50 Latency (ms)",
               "p95 Latency (ms)", "FPS", "Confidence (%)"]
    keys    = ["avg_ms", "p50_ms", "p95_ms", "fps", "avg_conf"]

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": "#1A1A1A", "font.family": "sans-serif", "font.size": 11,
    })

    # Color per model
    MODEL_COLORS = ["#4A90D9", "#E8714A"]

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(16, 5), sharey=False)
    fig.suptitle("YOLOv8n vs YOLOv8s — Model Comparison",
                 fontsize=14, fontweight="bold", color="#1A1A1A", y=1.02)

    x = np.arange(len(models))
    bar_w = 0.5

    for ax, metric, key in zip(axes, metrics, keys):
        vals = [r[key] for r in rows]
        bars = ax.bar(x, vals, bar_w, color=MODEL_COLORS, alpha=0.9,
                      edgecolor="white", linewidth=1.2)

        # Value labels on top of bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#1A1A1A")

        # Real-time line only on latency and FPS charts
        if "ms" in metric.lower():
            ax.axhline(y=66.7, color="#E53935", linestyle="--",
                       linewidth=1.2, label="RT ceiling (67 ms)")
        if metric == "FPS":
            ax.axhline(y=15, color="#E53935", linestyle="--",
                       linewidth=1.2, label="RT target (15 FPS)")

        ax.set_title(metric, fontsize=11, fontweight="bold",
                     color="#1A1A1A", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10, fontweight="bold",
                           color="#1A1A1A")
        ax.tick_params(length=0)
        ax.set_ylim(0, max(vals) * 1.25)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.yaxis.set_tick_params(labelsize=9, color="#888888")
        ax.grid(axis="y", color="#EEEEEE", linewidth=1)

        if "ms" in metric.lower() or metric == "FPS":
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    # Shared legend for model colors
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.9)
               for c in MODEL_COLORS]
    fig.legend(handles, models,
               loc="lower center", ncol=len(models),
               fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.06))

    fig.text(0.5, 1.04,
             "Red dashed line = real-time threshold   |   "
             "lower latency and higher FPS = better",
             ha="center", fontsize=8.5, color="#888888", style="italic")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Model comparison saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 56)
    print("  AR Flashcards — YOLO Benchmark Visualizations")
    print("=" * 56)

    try:
        import matplotlib
    except ImportError:
        print("  pip install matplotlib --break-system-packages")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")

    # Chart 1 — per-class heatmap
    if BENCHMARK_CSV.exists():
        rows = load_benchmark(BENCHMARK_CSV)
        if rows:
            out = RESULTS_DIR / f"yolo_class_heatmap_{stamp}.png"
            plot_class_heatmap(rows, out)
    else:
        print(f"  ⚠ {BENCHMARK_CSV} not found — skipping class heatmap")

    # Chart 2 — model comparison
    summary_rows = load_summary(RESULTS_DIR)
    if summary_rows:
        out = RESULTS_DIR / f"yolo_model_comparison_{stamp}.png"
        plot_model_comparison(summary_rows, out)
    else:
        print("  ⚠ No compare summary CSV found — skipping model comparison")

    print("\n" + "=" * 56)

if __name__ == "__main__":
    main()
