# Benchmark figures

Eight paper-ready PNGs, each generated from a committed CSV in the
repo. Regenerate any time the underlying CSVs change:

```
python benchmarks/make_figures.py
```

The script auto-picks the *latest* file matching each pattern, so a
fresh benchmark run drops in without code edits.

## Figures

| File | What it shows | Source |
|---|---|---|
| `fig1_per_agent_latency.png` | Mean inference latency per agent on a log scale, with p95 markers. The latency-spread story in one chart: detection in tens of ms, OCR in hundreds, LLM in seconds. | `OCR/results/e2e_*_agents.csv` |
| `fig2_yolo_model_tradeoff.png` | YOLOv8 n → s → m latency-vs-confidence scatter with FPS callouts. The cost of going bigger. | `YOLOv8-Detection/results/compare_*_summary.csv` |
| `fig3_pipeline_modes.png` | Per-mode latency (left) and accuracy (right) for the three prototype modes (Single = CLIP, Multi = YOLO+CLIP, OCR = EasyOCR+Phi-3). | `OCR/results/e2e_*_pipeline.csv` |
| `fig4_reward_decomposition_tier1.png` | The headline L3 number from BENCHMARKS.md: per-agent + joint reward, n=554, all four sidecars live. LLM near-perfect, YOLO and OCR mid-50s. | `benchmarks/results/rewards_20260516_010557.csv` |
| `fig5_tier1_vs_tier2.png` | Tier 1 vs Tier 2 grouped bars. The headline finding: LLM reward drops 51 pp when "schema valid" is replaced with "topic match". Annotated with the circularity caveat for YOLO/OCR Tier-2. | Tier 1: rewards_20260516_010557.csv. Tier 2: rewards_20260516_030600.csv |
| `fig6_cross_cohort.png` | School-objects (n=554) vs text-problems (n=5) per-agent rewards. Joint reward almost equal, per-agent contributions invert. Direct support for "the decomposition does work a single number can't." | rewards_20260516_010557.csv + rewards_20260515_220432.csv |
| `fig7_per_class_heatmap.png` | Agent × class reward grid on the full Update-5 dataset. Surfaces *which classes broke which agent* — e.g. `periodic_table_poster` drags YOLO down to 0.31; `backpack` drags OCR down to 0.33. | rewards_20260516_010557.csv |
| `fig8_ocr_per_class.png` | OCR-only per-class latency (bars, log scale) and text confidence (line). `periodic_table_poster` is both the slowest *and* the highest-confidence class — text-heavy images stress EasyOCR but reward the effort. | `benchmarks/results/ocr_benchmark_*_summary.csv` |

## Where each figure goes in the paper

| Section | Figure |
|---|---|
| Methods → benchmark hierarchy | fig1, fig3 |
| Detection model selection | fig2 |
| Per-agent rewards (Tier 1) | fig4 |
| Tier 2 + the schema-validity ceiling | fig5 |
| Why per-agent decomposition matters | fig6 |
| Where the pipeline breaks per class | fig7, fig8 |

## Style notes

- 150 dpi, ~7×4.5 in (slightly wider for grouped bars), Tol-style
  colorblind-safe palette
- Latency axes are log-scale because the agents span four orders of
  magnitude
- Sources are noted in console output when `make_figures.py` runs, so
  reviewers can trace any number to its CSV
- Each figure stands alone — title + a paragraph of context (in the
  paper) is enough to interpret it
