"""
Multi-model comparative benchmark for Update 5.

Extends benchmark.py beyond a single pretrained YOLOv8n to any number of
YOLO checkpoints. For each checkpoint it reports per-class latency, FPS,
detection counts, and confidence, then writes:

  results/compare_<timestamp>.csv          <- long-format rows, one per (model,class)
  results/compare_<timestamp>_summary.csv  <- one row per model (overall stats)
  results/compare_<timestamp>.md           <- human-readable markdown table

Per-image timings (p50, p95, p99) are computed from the raw measurement array
so we can talk about tail latency, not just the mean.

Usage:

    # Default: YOLOv8n vs YOLOv8s vs YOLOv8m
    python YOLOv8-Detection/compare_benchmarks.py

    # Explicit checkpoint list (paths or Ultralytics hub names)
    python YOLOv8-Detection/compare_benchmarks.py \\
        --models yolov8n.pt yolov8s.pt yolov8m.pt runs/detect/train/weights/best.pt

    # Use a specific dataset root and class subset (defaults match Update 4)
    python YOLOv8-Detection/compare_benchmarks.py \\
        --images-dir YOLOv8-Detection/dataset/images/raw \\
        --classes textbook whiteboard desk_chair chemistry_flask laptop_computer ruler_pencil_pen

    # Override conf/imgsz/warmup
    python YOLOv8-Detection/compare_benchmarks.py --imgsz 640 --conf 0.25 --warmup 5

The script is intentionally dependency-light: it reuses whatever torch and
ultralytics are already installed in the project venv.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")


DEFAULT_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
DEFAULT_IMAGES_DIR = "YOLOv8-Detection/dataset/images/raw"
DEFAULT_CLASSES = [
    "textbook",
    "whiteboard",
    "desk_chair",
    "chemistry_flask",
    "laptop_computer",
    "ruler_pencil_pen",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def percentile(values, pct: float) -> float:
    """Simple linear-interpolation percentile. pct in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def gather_images(base: Path, classes: list[str]) -> list[tuple[str, Path]]:
    pairs = []
    for class_dir in sorted(base.iterdir()):
        if class_dir.is_dir() and class_dir.name in classes:
            for img in class_dir.iterdir():
                if img.suffix.lower() in IMAGE_EXTS:
                    pairs.append((class_dir.name, img))
    return pairs


def benchmark_model(model_path: str, images: list[tuple[str, Path]],
                    device: str, imgsz: int, conf: float, warmup: int):
    """Run one model against the shared image list. Returns a dict of metrics."""
    print(f"\n  Loading model: {model_path}")
    model = YOLO(model_path)
    model.to(device)

    # Warmup — throw away these timings so we measure steady-state.
    warm_imgs = [str(p) for _, p in images[:max(1, warmup)]]
    for img in warm_imgs:
        model.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False)

    per_image_ms: list[float] = []
    class_times = defaultdict(list)
    class_dets = defaultdict(list)
    class_confs = defaultdict(list)

    n = len(images)
    # Cheap progress ticker so long runs don't look stuck.
    tick_every = max(1, n // 10)
    start = time.perf_counter()
    for i, (class_name, img_path) in enumerate(images, 1):
        t0 = time.perf_counter()
        results = model.predict(
            str(img_path),
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        per_image_ms.append(dt_ms)
        class_times[class_name].append(dt_ms)

        boxes = results[0].boxes
        n_dets = len(boxes) if boxes is not None else 0
        class_dets[class_name].append(n_dets)
        if boxes is not None and n_dets > 0:
            class_confs[class_name].extend(boxes.conf.cpu().tolist())

        if i % tick_every == 0:
            print(f"   ... {i}/{n} images", flush=True)

    wall_s = time.perf_counter() - start

    # Per-class summary
    per_class = []
    for cname in sorted(class_times):
        ts = class_times[cname]
        ds = class_dets[cname]
        cs = class_confs[cname]
        avg_ms = mean(ts)
        per_class.append({
            "class": cname,
            "images": len(ts),
            "avg_ms": round(avg_ms, 2),
            "p50_ms": round(percentile(ts, 50), 2),
            "p95_ms": round(percentile(ts, 95), 2),
            "fps": round(1000 / avg_ms, 2),
            "avg_dets": round(mean(ds), 2),
            "avg_conf": round(mean(cs), 4) if cs else 0.0,
        })

    # Overall summary
    all_confs = [c for cs in class_confs.values() for c in cs]
    overall = {
        "images": len(images),
        "wall_s": round(wall_s, 2),
        "avg_ms": round(mean(per_image_ms), 2),
        "p50_ms": round(percentile(per_image_ms, 50), 2),
        "p95_ms": round(percentile(per_image_ms, 95), 2),
        "p99_ms": round(percentile(per_image_ms, 99), 2),
        "fps": round(1000 / mean(per_image_ms), 2),
        "avg_conf": round(mean(all_confs), 4) if all_confs else 0.0,
    }

    return {
        "model": model_path,
        "per_class": per_class,
        "overall": overall,
    }


def write_long_csv(path: Path, model_results: list[dict]):
    fields = ["model", "class", "images", "avg_ms", "p50_ms", "p95_ms",
              "fps", "avg_dets", "avg_conf"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in model_results:
            for row in r["per_class"]:
                w.writerow({"model": r["model"], **row})


def write_summary_csv(path: Path, model_results: list[dict]):
    fields = ["model", "images", "wall_s", "avg_ms", "p50_ms", "p95_ms",
              "p99_ms", "fps", "avg_conf"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in model_results:
            w.writerow({"model": r["model"], **r["overall"]})


def write_markdown(path: Path, model_results: list[dict], meta: dict):
    lines = []
    lines.append("# YOLOv8 comparative benchmark\n")
    lines.append(f"- device: `{meta['device']}`")
    lines.append(f"- imgsz: `{meta['imgsz']}`")
    lines.append(f"- conf: `{meta['conf']}`")
    lines.append(f"- warmup: `{meta['warmup']}`")
    lines.append(f"- classes: `{', '.join(meta['classes'])}`")
    lines.append(f"- total images: `{meta['total_images']}`")
    lines.append(f"- timestamp: `{meta['timestamp']}`\n")
    lines.append("## Overall\n")
    lines.append("| model | images | wall s | mean ms | p50 ms | p95 ms | p99 ms | FPS | mean conf |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in model_results:
        o = r["overall"]
        lines.append(
            f"| `{r['model']}` | {o['images']} | {o['wall_s']} | {o['avg_ms']} | "
            f"{o['p50_ms']} | {o['p95_ms']} | {o['p99_ms']} | {o['fps']} | {o['avg_conf']:.2%} |"
        )
    lines.append("")
    lines.append("## Per-class mean latency (ms)\n")
    # Column per class, row per model
    class_order = [c["class"] for c in model_results[0]["per_class"]]
    header = "| model | " + " | ".join(class_order) + " |"
    sep = "|---|" + "---|" * len(class_order)
    lines.append(header)
    lines.append(sep)
    for r in model_results:
        by_class = {c["class"]: c for c in r["per_class"]}
        cells = " | ".join(
            f"{by_class[c]['avg_ms']}" if c in by_class else "—"
            for c in class_order
        )
        lines.append(f"| `{r['model']}` | {cells} |")
    lines.append("")
    lines.append("## Per-class FPS\n")
    lines.append(header)
    lines.append(sep)
    for r in model_results:
        by_class = {c["class"]: c for c in r["per_class"]}
        cells = " | ".join(
            f"{by_class[c]['fps']}" if c in by_class else "—"
            for c in class_order
        )
        lines.append(f"| `{r['model']}` | {cells} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="YOLO checkpoints to compare. Paths or hub names.")
    ap.add_argument("--images-dir", type=str, default=DEFAULT_IMAGES_DIR,
                    help="Root containing <class>/*.jpg subfolders.")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="Subset of class folders to benchmark. "
                         "Default: the six classes from the Update 4 baseline.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out-dir", type=str, default="YOLOv8-Detection/results",
                    help="Where to write CSVs and the markdown summary.")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional suffix appended to the output filenames.")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"Images dir not found: {images_dir.resolve()}")

    images = gather_images(images_dir, args.classes)
    if not images:
        raise SystemExit(
            f"No images found under {images_dir.resolve()} for classes "
            f"{args.classes}. Run collect_dataset.py first."
        )

    device = pick_device()
    print("=" * 64)
    print("  YOLOv8 comparative benchmark")
    print("=" * 64)
    print(f"  Device   : {device}")
    print(f"  Classes  : {', '.join(args.classes)}")
    print(f"  Images   : {len(images)}")
    print(f"  Imgsz    : {args.imgsz}")
    print(f"  Conf     : {args.conf}")
    print(f"  Warmup   : {args.warmup}")
    print(f"  Models   :")
    for m in args.models:
        print(f"     - {m}")

    model_results = []
    for m in args.models:
        try:
            r = benchmark_model(
                m, images, device,
                imgsz=args.imgsz, conf=args.conf, warmup=args.warmup,
            )
        except Exception as e:
            print(f"   !! {m} failed: {e}")
            continue
        model_results.append(r)

    if not model_results:
        raise SystemExit("All models failed to benchmark.")

    # Output files
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    stem = f"compare_{stamp}{suffix}"
    long_csv = out_dir / f"{stem}.csv"
    summary_csv = out_dir / f"{stem}_summary.csv"
    md_path = out_dir / f"{stem}.md"

    write_long_csv(long_csv, model_results)
    write_summary_csv(summary_csv, model_results)
    write_markdown(md_path, model_results, {
        "device": device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "warmup": args.warmup,
        "classes": args.classes,
        "total_images": len(images),
        "timestamp": stamp,
    })

    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  {'model':<36} {'mean ms':>8} {'p95 ms':>8} {'FPS':>6} {'conf':>8}")
    for r in model_results:
        o = r["overall"]
        print(f"  {r['model']:<36} {o['avg_ms']:>8.1f} {o['p95_ms']:>8.1f} "
              f"{o['fps']:>6.1f} {o['avg_conf']*100:>7.1f}%")
    print("=" * 64)
    print(f"  Long CSV : {long_csv}")
    print(f"  Summary  : {summary_csv}")
    print(f"  Markdown : {md_path}")

    # Stash a compact JSON for anyone wiring this into a pipeline
    (out_dir / f"{stem}.json").write_text(json.dumps({
        "meta": {
            "device": device,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "warmup": args.warmup,
            "classes": args.classes,
            "total_images": len(images),
            "timestamp": stamp,
        },
        "results": model_results,
    }, indent=2))


if __name__ == "__main__":
    main()
