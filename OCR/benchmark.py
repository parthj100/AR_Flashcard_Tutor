"""
OCR agent benchmark.

Last semester's OCR speed/confidence benchmark (per-class folders of
school-object photos). Kept as a starting point; the Fall 2026 study needs a
new harness that scores CER/WER and critical errors against reference
transcripts. The default --images-dir below pointed at the YOLO dataset,
which is now archived, so pass --images-dir explicitly.

Hits the running EasyOCR server (default http://127.0.0.1:8766) with every
image under a dataset directory, records per-image latency + text-confidence
+ lines-extracted, then aggregates per-class and overall.

Outputs (next to this file in ../benchmarks/results/ by default):
  ocr_benchmark_<timestamp>.csv  - one row per image
  ocr_benchmark_<timestamp>.md   - human-readable summary
  ocr_benchmark_<timestamp>_summary.csv - one row per class + overall

Run:
  python OCR/benchmark.py \\
      --images-dir YOLOv8-Detection/dataset/images/raw \\
      --classes microscope calculator backpack periodic_table_poster globe_model safety_goggles \\
      --endpoint http://127.0.0.1:8766
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path
from statistics import mean

DEFAULT_ENDPOINT = "http://127.0.0.1:8766"
DEFAULT_IMAGES_DIR = "YOLOv8-Detection/dataset/images/raw"
DEFAULT_CLASSES = [
    "microscope",
    "calculator",
    "backpack",
    "periodic_table_poster",
    "globe_model",
    "safety_goggles",
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


def collect_images(images_dir: Path, classes: list[str]) -> list[tuple[str, Path]]:
    out = []
    for cls in classes:
        d = images_dir / cls
        if not d.is_dir():
            print(f"  WARN: missing class dir {d}")
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                out.append((cls, p))
    return out


def ocr_one(endpoint: str, img: Path, conf: float, timeout: float = 60.0) -> dict:
    raw = img.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    body = json.dumps({"image": b64, "conf": conf}).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/ocr",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    wall_ms = (time.perf_counter() - t0) * 1000
    lines = payload.get("lines", [])
    confs = [ln["confidence"] for ln in lines]
    return {
        "lines":         len(lines),
        "mean_conf":     mean(confs) if confs else 0.0,
        "inference_ms":  payload.get("inference_ms", 0.0),
        "wall_ms":       wall_ms,
        "raw_text":      payload.get("raw_text", "")[:120],
    }


def check_health(endpoint: str) -> dict:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/health", timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def run(args):
    endpoint = args.endpoint
    print("=" * 62)
    print("  OCR agent benchmark — EasyOCR sidecar")
    print("=" * 62)

    try:
        h = check_health(endpoint)
        print(f"\n  Endpoint : {endpoint}")
        print(f"  Health   : {h}")
    except (urllib.error.URLError, ConnectionError) as e:
        print(f"\n  ERROR: OCR server unreachable at {endpoint}: {e}")
        print(f"  Start it with: python OCR/ocr_serve.py")
        return 1

    images_dir = Path(args.images_dir)
    images = collect_images(images_dir, args.classes)
    if not images:
        print(f"\n  ERROR: no images found under {images_dir.resolve()}")
        return 1

    print(f"\n  Conf     : {args.conf}")
    print(f"  Images   : {len(images)} across {len(args.classes)} classes")

    # Warmup
    print("\n  Warming up...", end=" ", flush=True)
    for _, p in images[: args.warmup]:
        try:
            ocr_one(endpoint, p, args.conf)
        except Exception:
            pass
    print("done.")

    # Main loop
    print("\n  Running OCR over dataset...")
    rows = []  # one per image
    per_class_lat: dict[str, list[float]] = defaultdict(list)
    per_class_conf: dict[str, list[float]] = defaultdict(list)
    per_class_lines: dict[str, list[int]] = defaultdict(list)
    all_lat: list[float] = []

    t_wall = time.perf_counter()
    for i, (cls, p) in enumerate(images, 1):
        try:
            r = ocr_one(endpoint, p, args.conf)
        except Exception as e:
            print(f"    [{i}/{len(images)}] FAIL {p.name}: {e}")
            continue
        rows.append({
            "class": cls,
            "image": str(p.relative_to(images_dir)),
            "lines": r["lines"],
            "mean_conf": round(r["mean_conf"], 4),
            "inference_ms": round(r["inference_ms"], 2),
            "wall_ms": round(r["wall_ms"], 2),
            "raw_text": r["raw_text"],
        })
        per_class_lat[cls].append(r["inference_ms"])
        per_class_conf[cls].append(r["mean_conf"])
        per_class_lines[cls].append(r["lines"])
        all_lat.append(r["inference_ms"])
        if i % 50 == 0:
            print(f"    [{i}/{len(images)}]")

    total_wall_s = time.perf_counter() - t_wall

    if not rows:
        print("\n  ERROR: no successful OCR runs")
        return 1

    # Aggregate
    summary = []
    for cls in args.classes:
        lats = per_class_lat[cls]
        if not lats:
            continue
        summary.append({
            "class":        cls,
            "images":       len(lats),
            "mean_ms":      round(mean(lats), 2),
            "p50_ms":       round(percentile(lats, 50), 2),
            "p95_ms":       round(percentile(lats, 95), 2),
            "fps":          round(1000.0 / mean(lats), 2),
            "avg_lines":    round(mean(per_class_lines[cls]), 2),
            "avg_conf":     round(mean(per_class_conf[cls]), 4),
        })

    overall = {
        "class":     "OVERALL",
        "images":    len(rows),
        "mean_ms":   round(mean(all_lat), 2),
        "p50_ms":    round(percentile(all_lat, 50), 2),
        "p95_ms":    round(percentile(all_lat, 95), 2),
        "fps":       round(1000.0 / mean(all_lat), 2) if all_lat else 0.0,
        "avg_lines": round(mean([r["lines"] for r in rows]), 2),
        "avg_conf":  round(mean([r["mean_conf"] for r in rows]), 4),
    }

    # Write outputs
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = out_dir / f"ocr_benchmark_{ts}"

    with (stem.with_suffix(".csv")).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (out_dir / f"ocr_benchmark_{ts}_summary.csv").open("w", newline="") as f:
        keys = ["class", "images", "mean_ms", "p50_ms", "p95_ms", "fps", "avg_lines", "avg_conf"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary + [overall])

    md_lines: list[str] = []
    md_lines.append(f"# OCR agent benchmark")
    md_lines.append("")
    md_lines.append(f"- server   : `{endpoint}`")
    md_lines.append(f"- conf     : `{args.conf}`")
    md_lines.append(f"- warmup   : `{args.warmup}`")
    md_lines.append(f"- classes  : `{', '.join(args.classes)}`")
    md_lines.append(f"- images   : `{len(rows)}`")
    md_lines.append(f"- wall_s   : `{total_wall_s:.2f}`")
    md_lines.append(f"- timestamp: `{ts}`")
    md_lines.append("")
    md_lines.append("## Overall")
    md_lines.append("")
    md_lines.append("| metric | value |")
    md_lines.append("|---|---:|")
    md_lines.append(f"| images | {overall['images']} |")
    md_lines.append(f"| mean ms | {overall['mean_ms']} |")
    md_lines.append(f"| p50 ms | {overall['p50_ms']} |")
    md_lines.append(f"| p95 ms | {overall['p95_ms']} |")
    md_lines.append(f"| FPS | {overall['fps']} |")
    md_lines.append(f"| avg lines / image | {overall['avg_lines']} |")
    md_lines.append(f"| mean text conf | {overall['avg_conf']:.4f} |")
    md_lines.append("")
    md_lines.append("## Per class")
    md_lines.append("")
    md_lines.append("| class | images | mean ms | p50 ms | p95 ms | FPS | lines/img | mean conf |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        md_lines.append(
            f"| `{row['class']}` | {row['images']} | {row['mean_ms']} | {row['p50_ms']} | "
            f"{row['p95_ms']} | {row['fps']} | {row['avg_lines']} | {row['avg_conf']:.4f} |"
        )
    md = "\n".join(md_lines) + "\n"
    (stem.with_suffix(".md")).write_text(md)

    # Final printout
    print("\n" + "=" * 62)
    print(f"  Done. {len(rows)} images in {total_wall_s:.2f} s")
    print(f"  Mean inference: {overall['mean_ms']} ms  ({overall['fps']} FPS)")
    print(f"  Mean text conf: {overall['avg_conf']:.4f}")
    print(f"  Avg lines/img : {overall['avg_lines']}")
    print(f"  Results       : {stem.with_suffix('.md')}")
    print("=" * 62)
    return 0


def main():
    p = argparse.ArgumentParser(description="OCR agent benchmark")
    p.add_argument("--endpoint",  default=DEFAULT_ENDPOINT)
    p.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    p.add_argument("--classes",   nargs="+", default=DEFAULT_CLASSES)
    p.add_argument("--conf",      type=float, default=0.5)
    p.add_argument("--warmup",    type=int, default=3)
    p.add_argument("--results-dir", default="benchmarks/results")
    args = p.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
