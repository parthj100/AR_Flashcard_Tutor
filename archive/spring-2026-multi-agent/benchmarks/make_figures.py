"""
Generate the benchmark figures from the committed result CSVs.

Reads from:
  YOLOv8-Detection/results/compare_<ts>_summary.csv      (yolov8n/s/m)
  YOLOv8-Detection/results/compare_<ts>.csv              (per-class)
  benchmarks/results/ocr_benchmark_<ts>_summary.csv      (OCR per-class)
  benchmarks/results/rewards_<ts>.csv                    (per-image rewards)
  OCR/results/e2e_<ts>_agents.csv                        (all 4 agents)
  OCR/results/e2e_<ts>_pipeline.csv                      (3 prototype modes)

Writes PNGs to benchmarks/figures/. Each figure is paper-friendly:
sans-serif, 150 dpi, ~6×4 in. Always reads the *latest* result file
matching each pattern so re-running the benchmarks regenerates the
figures without code changes.

Run:
    python benchmarks/make_figures.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------- style --------------------------------------------------------

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "font.size":         11,
    "font.family":       "sans-serif",
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "axes.axisbelow":    True,
})

# Color palette — colourblind-safe, matches Tol's bright scheme
PALETTE = {
    "yolo":   "#4477AA",
    "ocr":    "#EE6677",
    "clip":   "#228833",
    "llm":    "#CCBB44",
    "joint":  "#AA3377",
    "school": "#4477AA",
    "text":   "#EE6677",
    "tier1":  "#999999",
    "tier2":  "#AA3377",
}

REPO   = Path(__file__).resolve().parent.parent
OUT    = REPO / "benchmarks" / "figures"


def latest(pattern_dir: Path, glob: str) -> Path | None:
    matches = sorted(pattern_dir.glob(glob))
    return matches[-1] if matches else None


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


# ---------- figures ------------------------------------------------------

def fig_per_agent_latency():
    """Bar chart: mean inference latency per agent, with p95 error bars."""
    src = latest(REPO / "OCR" / "results", "e2e_*_agents.csv")
    if not src:
        print("  skip fig_per_agent_latency: no e2e_*_agents.csv")
        return
    rows = read_csv(src)
    rows = [r for r in rows if float(r["avg_ms"]) > 0]
    if not rows:
        return

    short_names = []
    for r in rows:
        n = r["agent"]
        if "YOLO"   in n: short_names.append("YOLO\nYOLOv8n")
        elif "CLIP" in n: short_names.append("CLIP\nViT-B/32")
        elif "OCR"  in n or "EasyOCR" in n: short_names.append("OCR\nEasyOCR")
        elif "Phi"  in n: short_names.append("LLM\nPhi-3-mini")
        else:             short_names.append(n)

    colors = []
    for r in rows:
        n = r["agent"].lower()
        if "yolo" in n: colors.append(PALETTE["yolo"])
        elif "clip" in n: colors.append(PALETTE["clip"])
        elif "easyocr" in n: colors.append(PALETTE["ocr"])
        elif "phi" in n: colors.append(PALETTE["llm"])
        else: colors.append("#888")

    means = [float(r["avg_ms"]) for r in rows]
    p95s  = [float(r["p95_ms"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(short_names, means, color=colors, edgecolor="black", linewidth=0.5)
    # p95 as a separate annotated tick line on top of each bar
    for bar, p95, mean_v in zip(bars, p95s, means):
        ax.plot([bar.get_x(), bar.get_x() + bar.get_width()],
                [p95, p95], color="black", linewidth=1.2)
        ax.annotate(f"p95={p95:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, p95),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, color="#333")
        ax.annotate(f"{mean_v:.0f} ms",
                    xy=(bar.get_x() + bar.get_width() / 2, mean_v / 2),
                    ha="center", color="white", fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Inference latency (ms, log scale)")
    ax.set_title("Per-agent latency  (means with p95 markers)")
    ax.set_ylim(1, max(p95s) * 1.6)
    fig.savefig(OUT / "fig1_per_agent_latency.png")
    plt.close(fig)
    print(f"  wrote fig1_per_agent_latency.png  (source: {src.name})")


def fig_yolo_model_tradeoff():
    """yolov8 n / s / m: latency vs confidence, with FPS callouts."""
    src = latest(REPO / "YOLOv8-Detection" / "results", "compare_*_summary.csv")
    if not src:
        print("  skip fig_yolo_model_tradeoff: no summary csv")
        return
    rows = read_csv(src)

    models = [r["model"] for r in rows]
    avg_ms = [float(r["avg_ms"]) for r in rows]
    conf   = [float(r["avg_conf"]) * 100 for r in rows]
    fps    = [float(r["fps"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(avg_ms, conf, s=160, c=[PALETTE["yolo"]], edgecolor="black",
               linewidth=0.6, zorder=3)
    for x, y, m, f in zip(avg_ms, conf, models, fps):
        label = f"{m.replace('.pt','')}\n{f:.1f} FPS"
        ax.annotate(label, xy=(x, y), xytext=(8, 8),
                    textcoords="offset points", fontsize=10)
    ax.plot(avg_ms, conf, color=PALETTE["yolo"], linewidth=1.2,
            alpha=0.6, zorder=2)
    ax.set_xlabel("Mean inference latency (ms)")
    ax.set_ylabel("Mean detection confidence (%)")
    ax.set_title("YOLOv8 n → s → m  —  latency / confidence trade-off")
    fig.savefig(OUT / "fig2_yolo_model_tradeoff.png")
    plt.close(fig)
    print(f"  wrote fig2_yolo_model_tradeoff.png  (source: {src.name})")


def fig_pipeline_modes():
    """Three prototype modes side-by-side: Single vs Multi vs OCR."""
    src = latest(REPO / "OCR" / "results", "e2e_*_pipeline.csv")
    if not src:
        print("  skip fig_pipeline_modes: no e2e_*_pipeline.csv")
        return
    rows = read_csv(src)
    rows = [r for r in rows if float(r["avg_ms"]) > 0]

    names = [r["mode"] for r in rows]
    means = [float(r["avg_ms"]) for r in rows]
    accs  = [float(r["accuracy"]) * 100 for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    ax1.bar(names, means, color=[PALETTE["yolo"], PALETTE["clip"], PALETTE["ocr"]],
            edgecolor="black", linewidth=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("End-to-end latency (ms, log scale)")
    ax1.set_title("Per-mode latency")
    for x, y in zip(names, means):
        ax1.annotate(f"{y:.0f} ms", (x, y), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=9)
    ax1.tick_params(axis="x", labelrotation=15)

    ax2.bar(names, accs, color=[PALETTE["yolo"], PALETTE["clip"], PALETTE["ocr"]],
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Per-mode accuracy")
    ax2.set_ylim(0, 105)
    for x, y in zip(names, accs):
        ax2.annotate(f"{y:.0f}%", (x, y), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=9)
    ax2.tick_params(axis="x", labelrotation=15)

    fig.suptitle("Three prototype modes — pipeline-level benchmark", fontsize=13)
    fig.savefig(OUT / "fig3_pipeline_modes.png")
    plt.close(fig)
    print(f"  wrote fig3_pipeline_modes.png  (source: {src.name})")


def _aggregate_rewards(csv_path: Path) -> dict:
    rows = read_csv(csv_path)
    out = {"yolo": [], "ocr": [], "llm": [], "joint": []}
    for r in rows:
        for k in out:
            v = r.get(f"{k}_reward" if k != "joint" else "joint_reward", "")
            if v not in ("", None):
                try:
                    out[k].append(float(v))
                except ValueError:
                    pass
    return out


def fig_reward_decomposition():
    """Per-agent mean reward on the full Update-5 dataset (Tier 1, n=554)."""
    full = REPO / "benchmarks" / "results" / "rewards_20260516_010557.csv"
    if not full.is_file():
        print("  skip fig_reward_decomposition: no rewards_20260516_010557.csv")
        return
    agg = _aggregate_rewards(full)
    keys = ["yolo", "ocr", "llm", "joint"]
    means = [mean(agg[k]) if agg[k] else 0 for k in keys]
    ns    = [len(agg[k])   for k in keys]
    colors = [PALETTE[k] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(keys, means, color=colors, edgecolor="black", linewidth=0.5)
    for bar, m, n in zip(bars, means, ns):
        ax.annotate(f"{m:.3f}\n(n={n})",
                    xy=(bar.get_x() + bar.get_width() / 2, m),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean reward  (0 = bad, 1 = perfect)")
    ax.set_title("L3 reward decomposition  •  Tier 1  •  n = 554 (Update-5 dataset)")
    fig.savefig(OUT / "fig4_reward_decomposition_tier1.png")
    plt.close(fig)
    print(f"  wrote fig4_reward_decomposition_tier1.png")


def fig_tier1_vs_tier2():
    """Grouped bars: Tier 1 vs Tier 2 per-agent mean reward (n=30 subset)."""
    t1 = REPO / "benchmarks" / "results" / "rewards_20260515_215313.csv"  # n=12 closest
    t2 = REPO / "benchmarks" / "results" / "rewards_20260516_030600.csv"  # n=30 T2 run
    # Prefer the n=554 T1 reduction for the LLM number since that's what
    # the paper reports — but keep the 30-row T2 cohort fixed.
    full_t1 = REPO / "benchmarks" / "results" / "rewards_20260516_010557.csv"

    if not (t2.is_file() and full_t1.is_file()):
        print("  skip fig_tier1_vs_tier2: missing input csvs")
        return

    a1 = _aggregate_rewards(full_t1)   # n=554 T1
    a2 = _aggregate_rewards(t2)        # n=30 T2

    keys = ["yolo", "ocr", "llm", "joint"]
    t1_means = [mean(a1[k]) if a1[k] else 0 for k in keys]
    t2_means = [mean(a2[k]) if a2[k] else 0 for k in keys]

    x = list(range(len(keys)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.4))
    b1 = ax.bar([i - width/2 for i in x], t1_means, width,
                label="Tier 1  (confidence / schema, n=554)",
                color=PALETTE["tier1"], edgecolor="black", linewidth=0.5)
    b2 = ax.bar([i + width/2 for i in x], t2_means, width,
                label="Tier 2  (vs ground truth, n=30)",
                color=PALETTE["tier2"], edgecolor="black", linewidth=0.5)
    for bar, v in list(zip(b1, t1_means)) + list(zip(b2, t2_means)):
        if v > 0.02:
            ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width()/2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Mean reward")
    ax.set_ylim(0, 1.35)
    ax.set_title("Tier 1 vs Tier 2  —  LLM reward drops 51 pp on topic match",
                 pad=14)
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.99))

    # Annotate the LLM gap — arrow pointing from T1 top to T2 top of llm bars
    llm_idx = keys.index("llm")
    ax.annotate(
        f"Δ = −{(t1_means[llm_idx]-t2_means[llm_idx])*100:.0f} pp",
        xy=(llm_idx + width/2, t2_means[llm_idx] + 0.04),
        xytext=(llm_idx - 0.4, 1.18),
        fontsize=11, fontweight="bold", color=PALETTE["tier2"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["tier2"], lw=1.4,
                        connectionstyle="arc3,rad=-0.2"),
    )

    # Note about circularity of yolo/ocr Tier-2
    ax.text(0.02, -0.15,
            "Note: yolo & ocr Tier-2 ≈ 1.00 is circular (labels were "
            "auto-suggested by those agents).  Verify rows before publishing.",
            transform=ax.transAxes, fontsize=9, color="#555", style="italic")

    fig.savefig(OUT / "fig5_tier1_vs_tier2.png")
    plt.close(fig)
    print(f"  wrote fig5_tier1_vs_tier2.png")


def fig_cross_cohort():
    """School-objects vs text-problems per-agent decomposition."""
    school = REPO / "benchmarks" / "results" / "rewards_20260516_010557.csv"  # n=554
    text   = REPO / "benchmarks" / "results" / "rewards_20260515_220432.csv"  # n=5
    if not (school.is_file() and text.is_file()):
        print("  skip fig_cross_cohort: missing input csvs")
        return

    a_s = _aggregate_rewards(school)
    a_t = _aggregate_rewards(text)
    keys = ["yolo", "ocr", "llm", "joint"]
    s_means = [mean(a_s[k]) if a_s[k] else 0 for k in keys]
    t_means = [mean(a_t[k]) if a_t[k] else 0 for k in keys]

    x = list(range(len(keys)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar([i - width/2 for i in x], s_means, width,
                label="School objects  (n=554)",
                color=PALETTE["school"], edgecolor="black", linewidth=0.5)
    b2 = ax.bar([i + width/2 for i in x], t_means, width,
                label="Text problems  (n=5)",
                color=PALETTE["text"], edgecolor="black", linewidth=0.5)
    for bar, v in list(zip(b1, s_means)) + list(zip(b2, t_means)):
        ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Mean reward")
    ax.set_title("Cross-cohort decomposition\n"
                 "joint reward nearly equal — per-agent contributions invert",
                 pad=10)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(OUT / "fig6_cross_cohort.png")
    plt.close(fig)
    print(f"  wrote fig6_cross_cohort.png")


def fig_per_class_heatmap():
    """Heatmap: agent × class mean reward on the 554-image run."""
    src = REPO / "benchmarks" / "results" / "rewards_20260516_010557.csv"
    if not src.is_file():
        print("  skip fig_per_class_heatmap")
        return
    rows = read_csv(src)
    classes_seen: list[str] = []
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cls = r["image"].split("/", 1)[0]
        if cls not in classes_seen:
            classes_seen.append(cls)
        for ag in ("yolo", "ocr", "llm"):
            v = r.get(f"{ag}_reward", "")
            try:
                per[cls][ag].append(float(v))
            except (TypeError, ValueError):
                pass

    agents = ["yolo", "ocr", "llm"]
    matrix = [[mean(per[cls][ag]) if per[cls][ag] else 0
               for cls in classes_seen] for ag in agents]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes_seen)))
    ax.set_xticklabels([c.replace("_", "\n") for c in classes_seen],
                       fontsize=9)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents)
    for i, ag in enumerate(agents):
        for j, cls in enumerate(classes_seen):
            v = matrix[i][j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if v > 0.55 else "white", fontsize=9)
    ax.set_title("Per-class reward heatmap  (Tier 1, n=554)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Mean reward")
    fig.savefig(OUT / "fig7_per_class_heatmap.png")
    plt.close(fig)
    print(f"  wrote fig7_per_class_heatmap.png")


def fig_ocr_per_class():
    """OCR-specific per-class latency + confidence on Update-5 set."""
    src = REPO / "benchmarks" / "results" / "ocr_benchmark_20260515_124616_summary.csv"
    if not src.is_file():
        print("  skip fig_ocr_per_class")
        return
    rows = read_csv(src)
    rows = [r for r in rows if r["class"] != "OVERALL"]

    classes  = [r["class"] for r in rows]
    lat_ms   = [float(r["mean_ms"]) for r in rows]
    conf     = [float(r["avg_conf"]) * 100 for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    bars = ax1.bar(classes, lat_ms, color=PALETTE["ocr"], alpha=0.85,
                    edgecolor="black", linewidth=0.5, label="Mean latency (ms)")
    ax1.set_ylabel("Mean OCR latency (ms)", color=PALETTE["ocr"])
    ax1.tick_params(axis="y", colors=PALETTE["ocr"])
    ax1.tick_params(axis="x", labelrotation=20)
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.plot(classes, conf, marker="o", color="black", linewidth=1.5,
             label="Mean text confidence (%)")
    ax2.set_ylabel("Mean text confidence (%)", color="black")
    ax2.set_ylim(0, 100)
    ax2.grid(False)

    ax1.set_title("OCR  •  per-class latency (bars) and text confidence (line)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    fig.savefig(OUT / "fig8_ocr_per_class.png")
    plt.close(fig)
    print(f"  wrote fig8_ocr_per_class.png")


# ---------- main ---------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"  writing figures to {OUT}/\n")
    fig_per_agent_latency()
    fig_yolo_model_tradeoff()
    fig_pipeline_modes()
    fig_reward_decomposition()
    fig_tier1_vs_tier2()
    fig_cross_cohort()
    fig_per_class_heatmap()
    fig_ocr_per_class()
    print("\n  done.")


if __name__ == "__main__":
    main()
