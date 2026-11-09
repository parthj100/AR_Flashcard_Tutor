"""
End-to-end pipeline benchmark — measures the cascading latency of the
full multi-agent pipeline (YOLO → OCR → LLM) on a folder of images.

Whichever sidecars are reachable get exercised; missing servers are skipped
and reported as such. The point is to produce a number that supports the
project title — "Standardized Benchmarking of Multi-Agent Distributed
Machine Learning" — at the *system* level, not just per agent.

Outputs (in benchmarks/results/ by default):
  pipeline_<timestamp>.csv           - one row per image (per-agent latency)
  pipeline_<timestamp>.md            - human-readable summary
  pipeline_<timestamp>_summary.csv   - one row per agent + overall

Run:
  # All servers available
  python benchmarks/benchmark_pipeline.py \\
      --images-dir YOLOv8-Detection/dataset/images/raw \\
      --classes microscope calculator backpack \\
      --limit 30

  # Subset (just YOLO + OCR, skip LLM)
  python benchmarks/benchmark_pipeline.py --skip-llm
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from agents import (
    DEFAULT_YOLO, DEFAULT_OCR, DEFAULT_OLLAMA, DEFAULT_LLM_MODEL,
    yolo_health, yolo_detect,
    ocr_health, ocr_extract,
    ollama_health, llm_generate,
)

DEFAULT_IMAGES_DIR = "YOLOv8-Detection/dataset/images/raw"
DEFAULT_CLASSES = [
    "microscope", "calculator", "backpack",
    "periodic_table_poster", "globe_model", "safety_goggles",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def collect_images(images_dir: Path, classes: list[str], limit: int) -> list[tuple[str, Path]]:
    out = []
    for cls in classes:
        d = images_dir / cls
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                out.append((cls, p))
    if limit and limit > 0:
        # keep a balanced subset across classes
        per_class: dict[str, list[Path]] = defaultdict(list)
        for cls, p in out:
            per_class[cls].append(p)
        n_per = max(1, limit // max(1, len(per_class)))
        balanced = []
        for cls, paths in per_class.items():
            for p in paths[:n_per]:
                balanced.append((cls, p))
            if len(balanced) >= limit:
                break
        return balanced[:limit]
    return out


def run(args):
    print("=" * 64)
    print("  End-to-end pipeline benchmark")
    print("=" * 64)

    # Probe sidecars
    enabled = {"yolo": not args.skip_yolo, "ocr": not args.skip_ocr, "llm": not args.skip_llm}
    health = {}
    if enabled["yolo"]:
        health["yolo"] = yolo_health(args.yolo)
    if enabled["ocr"]:
        health["ocr"] = ocr_health(args.ocr)
    if enabled["llm"]:
        health["llm"] = ollama_health(args.ollama, args.llm_model)
    for k, v in health.items():
        if v.get("ok"):
            print(f"  {k:<5}: UP    {v}")
        else:
            print(f"  {k:<5}: DOWN  ({v.get('error')[:80]})")
            enabled[k] = False

    images = collect_images(Path(args.images_dir), args.classes, args.limit)
    if not images:
        print(f"\n  ERROR: no images under {args.images_dir}")
        return 1
    print(f"\n  Images : {len(images)} across {len(set(c for c, _ in images))} classes")
    print(f"  Agents enabled: {[k for k, v in enabled.items() if v]}\n")

    rows = []
    per_agent_lat = {"yolo": [], "ocr": [], "llm": []}
    per_agent_ok  = {"yolo": 0, "ocr": 0, "llm": 0}

    t_wall = time.perf_counter()
    for i, (cls, p) in enumerate(images, 1):
        row = {"class": cls, "image": p.name}

        # --- YOLO -------------------------------------------------------
        if enabled["yolo"]:
            r = yolo_detect(p, conf=args.yolo_conf, endpoint=args.yolo)
            if r.get("ok"):
                per_agent_lat["yolo"].append(r["wall_ms"])
                per_agent_ok["yolo"] += 1
                row["yolo_wall_ms"] = round(r["wall_ms"], 2)
                row["yolo_inf_ms"]  = round(r["inference_ms"], 2)
                row["yolo_n_det"]   = r["n_detections"]
                row["yolo_top"]     = r["top_class"]
                row["yolo_conf"]    = round(r["top_conf"], 4)
            else:
                row["yolo_error"] = r.get("error", "")[:80]

        # --- OCR --------------------------------------------------------
        if enabled["ocr"]:
            r = ocr_extract(p, conf=args.ocr_conf, endpoint=args.ocr)
            if r.get("ok"):
                per_agent_lat["ocr"].append(r["wall_ms"])
                per_agent_ok["ocr"] += 1
                row["ocr_wall_ms"]  = round(r["wall_ms"], 2)
                row["ocr_inf_ms"]   = round(r["inference_ms"], 2)
                row["ocr_n_lines"]  = r["n_lines"]
                row["ocr_mean_conf"] = round(r["mean_conf"], 4)
            else:
                row["ocr_error"] = r.get("error", "")[:80]

        # --- LLM --------------------------------------------------------
        if enabled["llm"]:
            # Use the YOLO top-class as a topic hint, falling back to the folder name
            topic = (row.get("yolo_top") or cls).replace("_", " ")
            r = llm_generate(topic, host=args.ollama, model=args.llm_model)
            if r.get("ok"):
                per_agent_lat["llm"].append(r["wall_ms"])
                per_agent_ok["llm"] += 1 if r["schema_valid"] else 0
                row["llm_wall_ms"] = round(r["wall_ms"], 2)
                row["llm_eval_ms"] = round(r["eval_ms"], 2)
                row["llm_tokens"]  = r["tokens"]
                row["llm_tps"]     = r["tokens_per_sec"]
                row["llm_topic"]   = topic
                row["llm_schema_valid"] = r["schema_valid"]
            else:
                row["llm_error"] = r.get("error", "")[:80]

        # End-to-end wall
        e2e = sum(row.get(k, 0) for k in ("yolo_wall_ms", "ocr_wall_ms", "llm_wall_ms"))
        row["e2e_wall_ms"] = round(e2e, 2)
        rows.append(row)

        if i % 5 == 0 or i == len(images):
            print(f"    [{i}/{len(images)}] {cls:<22}  e2e={row['e2e_wall_ms']:.0f} ms")

    total_wall_s = time.perf_counter() - t_wall

    # ------------------------------------------------------------------
    # Aggregate and write
    # ------------------------------------------------------------------
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = out_dir / f"pipeline_{ts}"

    # Per-image CSV (union of keys, written in deterministic order)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with stem.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary_rows = []
    for agent in ("yolo", "ocr", "llm"):
        if not enabled[agent]:
            summary_rows.append({"agent": agent, "status": "SKIPPED",
                                  "n": 0, "mean_ms": "", "p50_ms": "", "p95_ms": "",
                                  "fps_estimate": "", "ok_rate": ""})
            continue
        lats = per_agent_lat[agent]
        n = len(lats)
        if n == 0:
            summary_rows.append({"agent": agent, "status": "ALL_FAILED",
                                  "n": 0, "mean_ms": "", "p50_ms": "", "p95_ms": "",
                                  "fps_estimate": "", "ok_rate": ""})
            continue
        m = mean(lats)
        ok = per_agent_ok[agent] / n
        summary_rows.append({
            "agent":        agent,
            "status":       "OK",
            "n":            n,
            "mean_ms":      round(m, 2),
            "p50_ms":       round(percentile(lats, 50), 2),
            "p95_ms":       round(percentile(lats, 95), 2),
            "fps_estimate": round(1000.0 / m, 2),
            "ok_rate":      round(ok, 4),
        })

    e2e_lats = [r.get("e2e_wall_ms", 0.0) for r in rows]
    e2e_mean = mean(e2e_lats) if e2e_lats else 0.0
    e2e_summary = {
        "agent":        "e2e",
        "status":       "OK",
        "n":            len(e2e_lats),
        "mean_ms":      round(e2e_mean, 2),
        "p50_ms":       round(percentile(e2e_lats, 50), 2),
        "p95_ms":       round(percentile(e2e_lats, 95), 2),
        "fps_estimate": round(1000.0 / e2e_mean, 2) if e2e_mean else 0.0,
        "ok_rate":      "",
    }
    summary_rows.append(e2e_summary)

    with (out_dir / f"pipeline_{ts}_summary.csv").open("w", newline="") as f:
        keys = ["agent", "status", "n", "mean_ms", "p50_ms", "p95_ms",
                "fps_estimate", "ok_rate"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary_rows)

    md = []
    md.append(f"# End-to-end pipeline benchmark\n")
    md.append(f"- images   : `{len(rows)}`\n")
    md.append(f"- classes  : `{', '.join(args.classes)}`\n")
    md.append(f"- enabled  : `{[k for k, v in enabled.items() if v]}`\n")
    md.append(f"- wall_s   : `{total_wall_s:.2f}`\n")
    md.append(f"- timestamp: `{ts}`\n\n")
    md.append("## Per-agent latency\n\n")
    md.append("| agent | status | n | mean ms | p50 ms | p95 ms | FPS est | ok rate |\n")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in summary_rows:
        md.append(f"| `{r['agent']}` | {r['status']} | {r['n']} | {r['mean_ms']} "
                  f"| {r['p50_ms']} | {r['p95_ms']} | {r['fps_estimate']} | {r['ok_rate']} |\n")
    md.append("\n## End-to-end wall clock\n\n")
    md.append(f"Mean per-image: **{e2e_mean:.2f} ms**  "
              f"(p50 {percentile(e2e_lats, 50):.2f}, p95 {percentile(e2e_lats, 95):.2f})\n\n")
    md.append("Note: e2e here is the *sum* of per-agent wall_ms in the order YOLO → OCR → "
              "LLM. The agents in the live prototype run partly in parallel, so this is an "
              "upper bound, not a measured wall-clock for the live UI.\n")
    (stem.with_suffix(".md")).write_text("".join(md))

    print("\n" + "=" * 64)
    print(f"  Done. {len(rows)} images in {total_wall_s:.2f} s")
    for r in summary_rows:
        if r["status"] == "OK":
            print(f"    {r['agent']:<5} {r['mean_ms']:>8} ms  (p95 {r['p95_ms']})  "
                  f"ok={r['ok_rate']}")
        else:
            print(f"    {r['agent']:<5} [{r['status']}]")
    print(f"  Results : {stem.with_suffix('.md')}")
    print("=" * 64)
    return 0


def main():
    p = argparse.ArgumentParser(description="End-to-end pipeline benchmark")
    p.add_argument("--images-dir",  default=DEFAULT_IMAGES_DIR)
    p.add_argument("--classes",     nargs="+", default=DEFAULT_CLASSES)
    p.add_argument("--limit",       type=int, default=30)
    p.add_argument("--yolo",        default=DEFAULT_YOLO)
    p.add_argument("--ocr",         default=DEFAULT_OCR)
    p.add_argument("--ollama",      default=DEFAULT_OLLAMA)
    p.add_argument("--llm-model",   default=DEFAULT_LLM_MODEL)
    p.add_argument("--yolo-conf",   type=float, default=0.25)
    p.add_argument("--ocr-conf",    type=float, default=0.5)
    p.add_argument("--skip-yolo",   action="store_true")
    p.add_argument("--skip-ocr",    action="store_true")
    p.add_argument("--skip-llm",    action="store_true")
    p.add_argument("--results-dir", default="benchmarks/results")
    args = p.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
