import time
import csv
import warnings
from pathlib import Path
from collections import defaultdict
 
import torch
from ultralytics import YOLO
from ultralytics.utils import DATASETS_DIR
 
warnings.filterwarnings("ignore")
 
# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "yolov8n.pt"       # downloaded automatically if not present
CONF_THRESH = 0.25               # minimum confidence to count a detection
IMG_SIZE    = 640                # standard YOLO input size
WARMUP_RUNS = 5                  # throwaway runs before timing starts
OUTPUT_CSV  = Path(__file__).resolve().parent / "results" / "benchmark_results_coco128.csv"
 
# COCO128 lives here after first download
COCO128_IMAGES = DATASETS_DIR / "coco128" / "images" / "train2017"
 
 
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"        # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"
 
 
def ensure_coco128() -> Path:
    """
    Download COCO128 (~6 MB) if it isn't already cached, by calling
    Ultralytics' ``check_det_dataset`` on the ``coco128.yaml`` manifest —
    that helper resolves the dataset entry and triggers the download as a
    side effect. Returns the path to the train2017 image folder.
    """
    if not COCO128_IMAGES.exists():
        print("  Downloading COCO128 dataset (~6 MB)...", flush=True)
        from ultralytics.data.utils import check_det_dataset
        check_det_dataset("coco128.yaml")
    return COCO128_IMAGES
 
 
def collect_images(images_dir: Path) -> list[Path]:
    """Return all images in the COCO128 train2017 folder."""
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in exts
    ])
    return images
 
 
def run_benchmark():
    print("\n" + "=" * 62)
    print("  YOLOv8n Baseline Benchmark — COCO128")
    print("=" * 62)
 
    device = get_device()
    print(f"\n  Device   : {device.upper()}")
    print(f"  Model    : {MODEL_PATH}")
    print(f"  Img size : {IMG_SIZE}px")
    print(f"  Conf     : {CONF_THRESH}")
 
    # Load model (downloads yolov8n.pt automatically if missing)
    model = YOLO(MODEL_PATH)
    model.to(device)
 
    # Ensure COCO128 is downloaded
    images_dir = ensure_coco128()
    all_images = collect_images(images_dir)
 
    if not all_images:
        print(f"\n  ERROR: No images found in {images_dir.resolve()}")
        print("  Try deleting ~/.cache/ultralytics and re-running.")
        return
 
    print(f"\n  Images   : {len(all_images)} total (COCO128)")
 
    # ── Warmup ────────────────────────────────────────────────────────────────
    print("\n  Warming up model...", end=" ", flush=True)
    for img in all_images[:WARMUP_RUNS]:
        model.predict(str(img), imgsz=IMG_SIZE, conf=CONF_THRESH,
                      device=device, verbose=False)
    print("done\n")
 
    # ── Run inference on every image ──────────────────────────────────────────
    all_times = []
    all_confs = []
    all_det_counts = []
 
    # Also track per-detected-class stats for the top classes
    class_confs   = defaultdict(list)

    coco_names = model.names  # dict {int: str}

    for img_path in all_images:
        t0 = time.perf_counter()
        results = model.predict(
            str(img_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            device=device,
            verbose=False,
        )
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        all_times.append(elapsed_ms)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            confs       = boxes.conf.cpu().tolist()
            class_ids   = boxes.cls.cpu().tolist()
            all_confs.extend(confs)
            all_det_counts.append(len(boxes))

            for cid, conf in zip(class_ids, confs):
                cname = coco_names[int(cid)]
                class_confs[cname].append(conf)
        else:
            all_det_counts.append(0)
 
    # ── Overall stats ─────────────────────────────────────────────────────────
    overall_ms   = sum(all_times) / len(all_times)
    overall_fps  = 1000 / overall_ms
    overall_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
    overall_dets = sum(all_det_counts) / len(all_det_counts)
    total_dets   = sum(all_det_counts)
 
    # ── Top 15 most-detected classes ──────────────────────────────────────────
    top_classes = sorted(class_confs.items(),
                         key=lambda x: len(x[1]), reverse=True)[:15]
 
    print(f"  {'Class':<22} {'Detections':>11} {'Avg conf':>9}")
    print("  " + "-" * 44)
 
    rows = []
    for cname, confs in top_classes:
        avg_c = sum(confs) / len(confs)
        print(f"  {cname:<22} {len(confs):>11} {avg_c:>8.2%}")
        rows.append({
            "class":      cname,
            "detections": len(confs),
            "avg_conf":   round(avg_c, 4),
        })
 
    print("  " + "-" * 44)
    print(f"  {'OVERALL':<22} {total_dets:>11} {overall_conf:>8.2%}")
 
    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = Path(OUTPUT_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "detections", "avg_conf"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "class":      "OVERALL",
            "detections": total_dets,
            "avg_conf":   round(overall_conf, 4),
        })
    print(f"\n  Results saved → {csv_path.resolve()}")
 
    # ── Summary ───────────────────────────────────────────────────────────────
    rt_status = (
        "MEETS real-time target (≥15 FPS)"
        if overall_fps >= 15
        else "BELOW real-time target (≥15 FPS)"
    )
 
    print("\n" + "=" * 62)
    print("  BENCHMARK SUMMARY")
    print("=" * 62)
    print(f"  Model        : YOLOv8n (pretrained COCO baseline)")
    print(f"  Hardware     : Apple M2 ({device.upper()} backend)")
    print(f"  Dataset      : COCO128 ({len(all_images)} images, 80 classes)")
    print(f"  Avg latency  : {overall_ms:.1f} ms/image")
    print(f"  Avg FPS      : {overall_fps:.1f} frames/sec")
    print(f"  Avg conf     : {overall_conf:.2%}")
    print(f"  Avg dets     : {overall_dets:.2f} objects/image")
    print(f"  Total dets   : {total_dets}")
    print(f"  Real-time    : {rt_status}")
    print("=" * 62)
    print()
    print("  Next step: full pipeline benchmarks live in benchmarks/benchmark_pipeline.py")
    print("             and OCR/pipeline_benchmark.py (see BENCHMARKS.md)")
    print()
 
 
if __name__ == "__main__":
    run_benchmark()