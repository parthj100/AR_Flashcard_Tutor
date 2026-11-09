

import time
import csv
import warnings
from pathlib import Path
from collections import defaultdict

import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")


MODEL_PATH   = "ultralytics/yolov8n.pt"
IMAGES_DIR   = "dataset/images/raw"
OUTPUT_CSV   = "benchmark_results.csv"
CONF_THRESH  = 0.25   # minimum confidence to count a detection
IMG_SIZE     = 640    # standard YOLO input size
WARMUP_RUNS  = 5      # throwaway runs before timing starts

# Classes we collected 
OUR_CLASSES = [
    "textbook",
    "whiteboard",
    "desk_chair",
    "chemistry_flask",
    "laptop_computer",
    "ruler_pencil_pen",
]


def get_device():
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def collect_images(base: Path) -> list:
    """Gather all images from all class subfolders."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = []
    for class_dir in sorted(base.iterdir()):
        if class_dir.is_dir() and class_dir.name in OUR_CLASSES:
            for img in class_dir.iterdir():
                if img.suffix.lower() in exts:
                    images.append((class_dir.name, img))
    return images



def run_benchmark():
    print("=" * 62)
    print("  YOLOv8n Baseline Benchmark — Apple M2")
    print("=" * 62)


    device = get_device()
    print(f"\n  Device   : {device.upper()}")
    print(f"  Model    : {MODEL_PATH}")
    print(f"  Img size : {IMG_SIZE}px")
    print(f"  Conf     : {CONF_THRESH}")

    model = YOLO(MODEL_PATH)
    model.to(device)

    # Collect images
    base = Path(IMAGES_DIR)
    all_images = collect_images(base)
    if not all_images:
        print(f"\n No images found in {base.resolve()}")
        return
    print(f"\n  Images   : {len(all_images)} across {len(OUR_CLASSES)} classes\n")

    # ── Warmup ────────────────────────────────────────────────────────────────
    print("  Warming up model...", end=" ", flush=True)
    warmup_imgs = [str(p) for _, p in all_images[:WARMUP_RUNS]]
    for img in warmup_imgs:
        model.predict(img, imgsz=IMG_SIZE, conf=CONF_THRESH,
                      device=device, verbose=False)
    print("done\n")


    class_times   = defaultdict(list)   
    class_dets    = defaultdict(list)   
    class_confs   = defaultdict(list)   

    print(f"  {'Class':<22} {'Images':>7} {'Avg ms':>8} {'FPS':>7} {'Avg dets':>9} {'Avg conf':>9}")
    print("  " + "-" * 58)

    rows = []  

    for class_name in OUR_CLASSES:
        imgs = [(c, p) for c, p in all_images if c == class_name]
        if not imgs:
            continue

        for _, img_path in imgs:
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
            class_times[class_name].append(elapsed_ms)

            # Parse detections
            boxes = results[0].boxes
            n_dets = len(boxes) if boxes is not None else 0
            class_dets[class_name].append(n_dets)

            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().tolist()
                class_confs[class_name].extend(confs)

   
        times  = class_times[class_name]
        avg_ms = sum(times) / len(times)
        fps    = 1000 / avg_ms
        avg_d  = sum(class_dets[class_name]) / len(class_dets[class_name])
        avg_c  = (sum(class_confs[class_name]) / len(class_confs[class_name])
                  if class_confs[class_name] else 0.0)

        print(f"  {class_name:<22} {len(imgs):>7} {avg_ms:>7.1f}ms "
              f"{fps:>6.1f} {avg_d:>9.2f} {avg_c:>8.2%}")

        rows.append({
            "class":        class_name,
            "images":       len(imgs),
            "avg_ms":       round(avg_ms, 2),
            "fps":          round(fps, 2),
            "avg_dets":     round(avg_d, 2),
            "avg_conf":     round(avg_c, 4),
        })


    all_times = [t for ts in class_times.values() for t in ts]
    all_confs = [c for cs in class_confs.values() for c in cs]
    overall_ms   = sum(all_times) / len(all_times)
    overall_fps  = 1000 / overall_ms
    overall_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

    print("  " + "-" * 58)
    print(f"  {'OVERALL':<22} {len(all_images):>7} {overall_ms:>7.1f}ms "
          f"{overall_fps:>6.1f} {'':>9} {overall_conf:>8.2%}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "class": "OVERALL",
            "images": len(all_images),
            "avg_ms": round(overall_ms, 2),
            "fps": round(overall_fps, 2),
            "avg_dets": "",
            "avg_conf": round(overall_conf, 4),
        })

    print(f"\n  📊 Results saved to: {Path(OUTPUT_CSV).resolve()}")


    print("\n" + "=" * 62)
    print("  BENCHMARK SUMMARY")
    print("=" * 62)
    print(f"  Model        : YOLOv8n (pretrained COCO baseline)")
    print(f"  Hardware     : Apple M2 ({device.upper()} backend)")
    print(f"  Dataset      : {len(all_images)} school object images (6 classes)")
    print(f"  Avg latency  : {overall_ms:.1f} ms/image")
    print(f"  Avg FPS      : {overall_fps:.1f} frames/sec")
    print(f"  Avg conf     : {overall_conf:.2%}")
    rt_status = " MEETS real-time target (≥15 FPS)" if overall_fps >= 15 else "⚠️  BELOW real-time target (≥15 FPS)"
    print(f"  Real-time    : {rt_status}")
    print("=" * 62)
    print("\n  Note: This is a COCO-pretrained baseline. Custom fine-tuned")
    print("  model results will be compared against these numbers in PR5.")
    print()


if __name__ == "__main__":
    run_benchmark()
