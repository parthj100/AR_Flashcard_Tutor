"""
detect.py — Single Image Object Detection
------------------------------------------
Uses YOLOv8n (pretrained on COCO) to detect objects in a given image.
Weights are downloaded automatically on first run (~6 MB).

Usage:
    python detect.py --image path/to/image.jpg
    python detect.py --image path/to/image.jpg --conf 0.4
    python detect.py --image path/to/image.jpg --save

Output:
    - Prints detected objects with confidence scores + bounding boxes
    - Optionally saves an annotated image to results/
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "yolov8n.pt"   # auto-downloaded on first run
IMG_SIZE    = 640
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"        # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def detect(image_path: str, conf_thresh: float = 0.25, save: bool = False) -> dict:
    """
    Run YOLOv8n detection on a single image.

    Args:
        image_path:  Path to input image.
        conf_thresh: Minimum confidence to include a detection.
        save:        If True, saves annotated image to results/.

    Returns:
        A dict with keys:
          - image        : input path
          - device       : hardware used
          - latency_ms   : inference time in milliseconds
          - detections   : list of {label, confidence, bbox}
    """
    device = get_device()
    model  = YOLO(MODEL_PATH)
    model.to(device)

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ── Run inference ─────────────────────────────────────────────────────────
    t0      = time.perf_counter()
    results = model.predict(
        str(image_path),
        imgsz=IMG_SIZE,
        conf=conf_thresh,
        device=device,
        verbose=False,
    )
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000

    # ── Parse results ─────────────────────────────────────────────────────────
    detections = []
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            label      = model.names[int(box.cls)]
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label":      label,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                }
            })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    output = {
        "image":      str(image_path),
        "device":     device.upper(),
        "latency_ms": round(latency_ms, 2),
        "detections": detections,
    }

    # ── Save annotated image (optional) ───────────────────────────────────────
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        annotated = results[0].plot()          # numpy array with boxes drawn
        import cv2
        out_path = RESULTS_DIR / f"detected_{image_path.name}"
        cv2.imwrite(str(out_path), annotated)
        output["saved_to"] = str(out_path)

    return output


def print_results(output: dict):
    """Pretty-print detection results to the console."""
    print("\n" + "=" * 52)
    print("  DETECTION RESULTS")
    print("=" * 52)
    print(f"  Image    : {output['image']}")
    print(f"  Device   : {output['device']}")
    print(f"  Latency  : {output['latency_ms']} ms")
    print(f"  Objects  : {len(output['detections'])} detected")
    print("-" * 52)

    if not output["detections"]:
        print("  No objects detected above confidence threshold.")
    else:
        print(f"  {'#':<4} {'Label':<20} {'Confidence':>11}  Bounding Box")
        print(f"  {'-'*4} {'-'*20} {'-'*11}  {'-'*24}")
        for i, det in enumerate(output["detections"], 1):
            bb = det["bbox"]
            bbox_str = f"({bb['x1']}, {bb['y1']}) → ({bb['x2']}, {bb['y2']})"
            print(f"  {i:<4} {det['label']:<20} {det['confidence']:>10.2%}  {bbox_str}")

    if "saved_to" in output:
        print(f"\n  Annotated image saved → {output['saved_to']}")

    print("=" * 52 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8n object detection on a single image.")
    parser.add_argument("--image", required=True,  help="Path to input image")
    parser.add_argument("--conf",  type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save",  action="store_true", help="Save annotated image to results/")
    parser.add_argument("--json",  action="store_true", help="Also print raw JSON output")
    args = parser.parse_args()

    output = detect(args.image, conf_thresh=args.conf, save=args.save)
    print_results(output)

    if args.json:
        print("  Raw JSON output:")
        print(json.dumps(output, indent=2))
