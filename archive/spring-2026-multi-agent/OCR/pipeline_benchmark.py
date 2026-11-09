"""
End-to-End Pipeline Benchmark for AR Flashcards (Lens prototype).

Research question:
  Can a multi-agent, edge-distributed AR pipeline deliver accurate,
  context-aware educational overlays in real time on commodity hardware?

Measures 4 metrics across all 4 agents and all 3 pipeline modes:
  1. Latency    — avg, p50, p95 (ms)
  2. Accuracy   — did the agent return the correct/expected output?
  3. Throughput — FPS (1000 / avg_ms)
  4. Confidence — agent-reported confidence score (0..1)

Agents:
  YOLO      — calls serve.py      on port 8765
  EasyOCR   — calls ocr_serve.py  on port 8766
  CLIP      — runs in-process     (transformers / ONNX)
  Phi-3     — calls Ollama        on port 11434

Pipeline modes:
  Single    — CLIP score -> flashcard
  Multi     — YOLO detect -> CLIP re-ID -> flashcard
  OCR       — EasyOCR extract -> Phi-3 generate -> flashcard

Prerequisites:
  pip install requests Pillow torch transformers numpy --break-system-packages
  Terminal 1: python YOLOv8-Detection/serve.py
  Terminal 2: python OCR/ocr_serve.py
  Terminal 3: OLLAMA_ORIGINS='*' ollama serve

Usage (run from repo root):
  python OCR/benchmark_e2e.py

Output — all saved to OCR/results/:
  e2e_<timestamp>_agents.csv
  e2e_<timestamp>_pipeline.csv
  e2e_<timestamp>.md
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import time
import warnings
from pathlib import Path
from statistics import mean
from typing import Optional

import requests
from PIL import Image

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
YOLO_URL     = "http://127.0.0.1:8765"
OCR_URL      = "http://127.0.0.1:8766"
OLLAMA_URL   = "http://127.0.0.1:11434"
OLLAMA_MODEL = "phi3:mini"

DEFAULT_OBJECT_DIR = "YOLOv8-Detection/dataset/images/raw"
DEFAULT_OCR_DIR    = "OCR/test_images"
DEFAULT_OUT_DIR    = "OCR/results"
IMAGE_EXTS         = {".jpg", ".jpeg", ".png", ".webp"}

CLIP_LABEL_MAP = {
    # Update-4 classes
    "textbook":         ["book", "textbook"],
    "whiteboard":       ["whiteboard"],
    "desk_chair":       ["chair", "desk_chair"],
    "chemistry_flask":  ["beaker", "erlenmeyer", "test-tube", "chemistry_flask"],
    "laptop_computer":  ["laptop", "laptop_computer"],
    "ruler_pencil_pen": ["pencil", "ruler_pencil_pen"],
    # Update-5 expansion classes (current dataset on disk)
    "microscope":              ["microscope"],
    "calculator":              ["calculator"],
    "backpack":                ["backpack"],
    "periodic_table_poster":   ["periodic table", "chemistry poster", "periodic_table_poster"],
    "globe_model":             ["globe", "world globe", "globe_model"],
    "safety_goggles":          ["safety goggles", "lab goggles", "safety_goggles"],
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def encode_image(path: Path, max_side: int = 720) -> str:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def gather_images(base: Path, classes: Optional[list] = None) -> list[tuple[str, Path]]:
    pairs = []
    if not base.exists():
        return pairs
    for d in sorted(base.iterdir()):
        if d.is_dir() and (classes is None or d.name in classes):
            for img in d.iterdir():
                if img.suffix.lower() in IMAGE_EXTS:
                    pairs.append((d.name, img))
    for img in base.iterdir():
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            pairs.append(("ocr", img))
    return pairs


def server_ok(url: str, path: str = "/health") -> bool:
    try:
        return requests.get(url + path, timeout=3).ok
    except Exception:
        return False


def ollama_ok() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if not r.ok:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        return any(m.startswith(OLLAMA_MODEL.split(":")[0]) for m in models)
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Shared result builders
# ---------------------------------------------------------------------------

def _agent_result(name, latencies, confs, correct, total) -> dict:
    avg_ms = mean(latencies) if latencies else 0
    return {
        "agent":    name,
        "images":   total,
        "avg_ms":   round(avg_ms, 2),
        "p50_ms":   round(percentile(latencies, 50), 2),
        "p95_ms":   round(percentile(latencies, 95), 2),
        "fps":      round(1000 / avg_ms, 2) if avg_ms else 0,
        "accuracy": round(correct / total, 4) if total else 0,
        "avg_conf": round(mean(confs), 4) if confs else 0,
    }


def _pipeline_result(name, latencies, correct, total) -> dict:
    avg_ms = mean(latencies) if latencies else 0
    return {
        "mode":     name,
        "runs":     total,
        "avg_ms":   round(avg_ms, 2),
        "p50_ms":   round(percentile(latencies, 50), 2),
        "p95_ms":   round(percentile(latencies, 95), 2),
        "fps":      round(1000 / avg_ms, 2) if avg_ms else 0,
        "accuracy": round(correct / total, 4) if total else 0,
        "realtime": "YES" if avg_ms and (1000 / avg_ms) >= 15 else "NO",
    }


def _empty_pipeline(name) -> dict:
    return {"mode": name, "runs": 0, "avg_ms": 0, "p50_ms": 0,
            "p95_ms": 0, "fps": 0, "accuracy": 0, "realtime": "N/A"}

# ---------------------------------------------------------------------------
# Agent benchmarks (isolation)
# ---------------------------------------------------------------------------

def bench_yolo(images, warmup, runs) -> dict:
    print("\n  [1/4] Benchmarking YOLO agent...")
    latencies, confs, correct, total = [], [], 0, 0

    for _, p in images[:warmup]:
        try:
            requests.post(f"{YOLO_URL}/detect",
                          json={"image": encode_image(p), "conf": 0.25, "max_dets": 6},
                          timeout=30)
        except Exception:
            pass

    for class_name, p in images[:runs]:
        try:
            t0   = time.perf_counter()
            r    = requests.post(f"{YOLO_URL}/detect",
                                 json={"image": encode_image(p),
                                       "conf": 0.25, "max_dets": 6},
                                 timeout=30)
            dt   = (time.perf_counter() - t0) * 1000
            latencies.append(dt)
            dets = r.json().get("detections", [])
            expected = CLIP_LABEL_MAP.get(class_name, [class_name])
            hit = any(any(e in d["class_name"].lower() for e in expected) for d in dets)
            correct += int(hit)
            total   += 1
            if dets:
                confs.extend(d["confidence"] for d in dets)
        except Exception as e:
            print(f"    YOLO error on {p.name}: {e}")

    return _agent_result("YOLO (YOLOv8n)", latencies, confs, correct, total)


def bench_ocr(images, warmup, runs) -> dict:
    print("\n  [2/4] Benchmarking EasyOCR agent...")
    latencies, confs, correct, total = [], [], 0, 0

    for _, p in images[:warmup]:
        try:
            requests.post(f"{OCR_URL}/ocr",
                          json={"image": encode_image(p), "conf": 0.5}, timeout=60)
        except Exception:
            pass

    for _, p in images[:runs]:
        try:
            t0    = time.perf_counter()
            r     = requests.post(f"{OCR_URL}/ocr",
                                  json={"image": encode_image(p), "conf": 0.5},
                                  timeout=60)
            dt    = (time.perf_counter() - t0) * 1000
            latencies.append(dt)
            j     = r.json()
            lines = j.get("lines", [])
            raw   = j.get("raw_text", "").strip()
            correct += int(len(raw) > 0)
            total   += 1
            if lines:
                confs.extend(l["confidence"] for l in lines)
        except Exception as e:
            print(f"    OCR error on {p.name}: {e}")

    return _agent_result("EasyOCR", latencies, confs, correct, total)


def bench_clip(images, warmup, runs) -> dict:
    print("\n  [3/4] Benchmarking CLIP agent...")
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
    except ImportError:
        print("    transformers not installed — skipping CLIP")
        return _agent_result("CLIP", [], [], 0, 0)

    model_id  = "openai/clip-vit-base-patch32"
    print(f"    Loading {model_id}…")
    processor = CLIPProcessor.from_pretrained(model_id)
    model     = CLIPModel.from_pretrained(model_id)
    model.eval()

    prompts        = [f"a photo of a {v}"
                      for variants in CLIP_LABEL_MAP.values() for v in variants]
    prompt_classes = [cls for cls, variants in CLIP_LABEL_MAP.items()
                      for _ in variants]
    latencies, confs, correct, total = [], [], 0, 0

    for class_name, p in images[:runs]:
        try:
            im   = Image.open(p).convert("RGB")
            w, h = im.size
            side = min(w, h)
            im   = im.crop(((w-side)//2, (h-side)//2,
                             (w+side)//2, (h+side)//2))
            im   = im.resize((224, 224), Image.LANCZOS)
            t0     = time.perf_counter()
            inputs = processor(text=prompts, images=im,
                               return_tensors="pt", padding=True)
            with torch.no_grad():
                probs = (model(**inputs)
                         .logits_per_image[0]
                         .softmax(dim=0)
                         .cpu().tolist())
            latencies.append((time.perf_counter() - t0) * 1000)
            top_idx = probs.index(max(probs))
            confs.append(probs[top_idx])
            correct += int(prompt_classes[top_idx] == class_name)
            total   += 1
        except Exception as e:
            print(f"    CLIP error on {p.name}: {e}")

    return _agent_result("CLIP (ViT-B/32)", latencies, confs, correct, total)


def bench_phi3(n_runs: int = 5) -> dict:
    print("\n  [4/4] Benchmarking Phi-3 (Ollama) agent...")
    topics = ["Copper sulfate", "Mitochondria", "Newton's laws",
              "Photosynthesis", "Hagia Sophia"]
    schema = {
        "type": "object",
        "properties": {
            "name":    {"type": "string"},
            "subject": {"type": "string"},
            "oneline": {"type": "string"},
            "facts":   {"type": "array"},
        },
        "required": ["name", "subject", "oneline", "facts"],
    }
    latencies, correct, total = [], 0, 0

    for i in range(min(n_runs, len(topics))):
        try:
            body = {
                "model": OLLAMA_MODEL, "stream": False, "format": schema,
                "options": {"temperature": 0.2, "num_predict": 512},
                "messages": [
                    {"role": "system",
                     "content": "Write educational flashcards as JSON."},
                    {"role": "user",
                     "content": f"Write a flashcard for: {topics[i]}"},
                ],
            }
            t0     = time.perf_counter()
            r      = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=120)
            dt     = (time.perf_counter() - t0) * 1000
            latencies.append(dt)
            parsed = json.loads(
                r.json().get("message", {}).get("content", "{}"))
            correct += int(all(k in parsed
                               for k in ["name", "subject", "oneline", "facts"]))
            total  += 1
        except Exception as e:
            print(f"    Phi-3 error: {e}")

    return _agent_result(f"Phi-3 ({OLLAMA_MODEL})", latencies, [], correct, total)

# ---------------------------------------------------------------------------
# Pipeline mode benchmarks (end-to-end)
# ---------------------------------------------------------------------------

def bench_single_mode(images, runs) -> dict:
    print("\n  [E2E 1/3] Single mode (CLIP)…")
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
    except ImportError:
        return _empty_pipeline("Single (CLIP)")

    prompts        = [f"a photo of a {v}"
                      for variants in CLIP_LABEL_MAP.values() for v in variants]
    prompt_classes = [cls for cls, variants in CLIP_LABEL_MAP.items()
                      for _ in variants]
    latencies, correct, total = [], 0, 0

    for class_name, p in images[:runs]:
        try:
            im   = Image.open(p).convert("RGB")
            w, h = im.size
            side = min(w, h)
            im   = im.crop(((w-side)//2, (h-side)//2,
                             (w+side)//2, (h+side)//2))
            im   = im.resize((224, 224), Image.LANCZOS)
            t0     = time.perf_counter()
            inputs = processor(text=prompts, images=im,
                               return_tensors="pt", padding=True)
            with torch.no_grad():
                probs = (model(**inputs)
                         .logits_per_image[0]
                         .softmax(dim=0)
                         .cpu().tolist())
            latencies.append((time.perf_counter() - t0) * 1000)
            correct += int(prompt_classes[probs.index(max(probs))] == class_name)
            total   += 1
        except Exception as e:
            print(f"    Single mode error: {e}")

    return _pipeline_result("Single (CLIP)", latencies, correct, total)


def bench_multi_mode(images, runs) -> dict:
    print("\n  [E2E 2/3] Multi mode (YOLO + CLIP)…")
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
    except ImportError:
        return _empty_pipeline("Multi (YOLO+CLIP)")

    prompts        = [f"a photo of a {v}"
                      for variants in CLIP_LABEL_MAP.values() for v in variants]
    prompt_classes = [cls for cls, variants in CLIP_LABEL_MAP.items()
                      for _ in variants]
    latencies, correct, total = [], 0, 0

    for class_name, p in images[:runs]:
        try:
            data_url = encode_image(p, max_side=720)
            t0 = time.perf_counter()
            r  = requests.post(f"{YOLO_URL}/detect",
                               json={"image": data_url,
                                     "conf": 0.25, "max_dets": 6},
                               timeout=30)
            dets = r.json().get("detections", [])
            if dets:
                raw     = base64.b64decode(dets[0]["crop"].split(",")[1])
                crop_im = (Image.open(io.BytesIO(raw))
                           .convert("RGB")
                           .resize((224, 224)))
            else:
                crop_im = Image.open(p).convert("RGB").resize((224, 224))

            inputs = processor(text=prompts, images=crop_im,
                               return_tensors="pt", padding=True)
            with torch.no_grad():
                probs = (model(**inputs)
                         .logits_per_image[0]
                         .softmax(dim=0)
                         .cpu().tolist())
            latencies.append((time.perf_counter() - t0) * 1000)
            correct += int(prompt_classes[probs.index(max(probs))] == class_name)
            total   += 1
        except Exception as e:
            print(f"    Multi mode error: {e}")

    return _pipeline_result("Multi (YOLO+CLIP)", latencies, correct, total)


def bench_ocr_mode(images, runs) -> dict:
    print("\n  [E2E 3/3] OCR mode (EasyOCR + Phi-3)…")
    schema = {
        "type": "object",
        "properties": {
            "name":    {"type": "string"},
            "subject": {"type": "string"},
            "oneline": {"type": "string"},
            "facts":   {"type": "array"},
        },
        "required": ["name", "subject", "oneline", "facts"],
    }
    latencies, correct, total = [], 0, 0

    for _, p in images[:runs]:
        try:
            data_url = encode_image(p, max_side=720)
            t0       = time.perf_counter()
            r        = requests.post(f"{OCR_URL}/ocr",
                                     json={"image": data_url, "conf": 0.5},
                                     timeout=60)
            raw_text = r.json().get("raw_text", "").strip()

            if not raw_text:
                latencies.append((time.perf_counter() - t0) * 1000)
                total += 1
                continue

            body = {
                "model": OLLAMA_MODEL, "stream": False, "format": schema,
                "options": {"temperature": 0.2, "num_predict": 512},
                "messages": [
                    {"role": "system",
                     "content": "Write educational flashcards as JSON."},
                    {"role": "user",
                     "content": f'Flashcard for: "{raw_text[:200]}"'},
                ],
            }
            r2 = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=120)
            latencies.append((time.perf_counter() - t0) * 1000)
            parsed  = json.loads(
                r2.json().get("message", {}).get("content", "{}"))
            correct += int(all(k in parsed
                               for k in ["name", "subject", "oneline", "facts"]))
            total   += 1
        except Exception as e:
            print(f"    OCR mode error: {e}")

    return _pipeline_result("OCR (EasyOCR+Phi-3)", latencies, correct, total)

# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_agent_csv(path: Path, rows: list):
    fields = ["agent", "images", "avg_ms", "p50_ms", "p95_ms",
              "fps", "accuracy", "avg_conf"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_pipeline_csv(path: Path, rows: list):
    fields = ["mode", "runs", "avg_ms", "p50_ms", "p95_ms",
              "fps", "accuracy", "realtime"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_markdown(path: Path, agent_rows: list,
                   pipeline_rows: list, meta: dict):
    lines = [
        "# AR Flashcards — End-to-End Pipeline Benchmark\n",
        f"- timestamp: `{meta['timestamp']}`",
        f"- object images: `{meta['object_images']}`",
        f"- ocr images: `{meta['ocr_images']}`",
        f"- warmup runs: `{meta['warmup']}`",
        f"- runs per agent: `{meta['runs']}`\n",
        "## Agent Benchmarks (Isolation)\n",
        "| Agent | Images | Avg ms | p50 ms | p95 ms | FPS |"
        " Accuracy | Avg Conf |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agent_rows:
        lines.append(
            f"| {r['agent']} | {r['images']} | {r['avg_ms']} |"
            f" {r['p50_ms']} | {r['p95_ms']} | {r['fps']} |"
            f" {r['accuracy']:.1%} | {r['avg_conf']:.2%} |"
        )
    lines += [
        "\n## Pipeline Mode Benchmarks (End-to-End)\n",
        "| Mode | Runs | Avg ms | p50 ms | p95 ms | FPS |"
        " Accuracy | Real-time? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pipeline_rows:
        lines.append(
            f"| {r['mode']} | {r['runs']} | {r['avg_ms']} |"
            f" {r['p50_ms']} | {r['p95_ms']} | {r['fps']} |"
            f" {r['accuracy']:.1%} | {r['realtime']} |"
        )
    lines += [
        "\n## Accuracy Notes\n",
        "- CLIP accuracy  = top-1 class match vs ground-truth label",
        "- YOLO accuracy  = ≥1 detection overlapping expected class",
        "- OCR accuracy   = non-empty text extracted from image",
        "- Phi-3 accuracy = response parses as valid flashcard JSON",
        "- Phi-3 avg_conf = N/A (LLMs do not expose a confidence score)",
        "- Real-time threshold: ≥ 15 FPS",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--object-dir", default=DEFAULT_OBJECT_DIR)
    ap.add_argument("--ocr-dir",    default=DEFAULT_OCR_DIR)
    ap.add_argument("--warmup",     type=int, default=3)
    ap.add_argument("--runs",       type=int, default=10)
    ap.add_argument("--phi3-runs",  type=int, default=5)
    ap.add_argument("--out-dir",    default=DEFAULT_OUT_DIR,
                    help="Output folder (default: OCR/results/)")
    ap.add_argument("--skip-clip",  action="store_true")
    ap.add_argument("--skip-phi3",  action="store_true")
    args = ap.parse_args()

    print("=" * 66)
    print("  AR Flashcards — End-to-End Pipeline Benchmark")
    print("=" * 66)

    yolo_up = server_ok(YOLO_URL)
    ocr_up  = server_ok(OCR_URL)
    phi3_up = ollama_ok() and not args.skip_phi3

    print(f"\n  YOLO server  ({YOLO_URL}) :"
          f" {'✓ online' if yolo_up else '✗ offline'}")
    print(f"  OCR  server  ({OCR_URL})  :"
          f" {'✓ online' if ocr_up  else '✗ offline'}")
    print(f"  Ollama/Phi-3 ({OLLAMA_URL}):"
          f" {'✓ online' if phi3_up else '✗ offline'}")

    obj_images = [(c, p) for c, p in gather_images(Path(args.object_dir))
                  if c in CLIP_LABEL_MAP]
    ocr_images = gather_images(Path(args.ocr_dir))

    print(f"\n  Object images : {len(obj_images)}")
    print(f"  OCR images    : {len(ocr_images)}")
    print(f"  Warmup        : {args.warmup}")
    print(f"  Runs          : {args.runs}")

    if not obj_images:
        print(f"\n  ⚠ No object images found in {args.object_dir}")
        print("    Run collect_dataset.py first, or set --object-dir")
    if not ocr_images:
        print(f"\n  ⚠ No OCR images found in {args.ocr_dir}")
        print("    Add printed-text images to OCR/test_images/")

    # Phase 1 — agent isolation
    print("\n" + "=" * 66)
    print("  PHASE 1 — Agent Benchmarks (Isolation)")
    print("=" * 66)

    agent_rows = []
    if yolo_up and obj_images:
        agent_rows.append(bench_yolo(obj_images, args.warmup, args.runs))
    else:
        print("\n  [1/4] YOLO — skipped")
    if ocr_up and ocr_images:
        agent_rows.append(bench_ocr(ocr_images, args.warmup, args.runs))
    else:
        print("\n  [2/4] EasyOCR — skipped")
    if not args.skip_clip and obj_images:
        agent_rows.append(bench_clip(obj_images, args.warmup, args.runs))
    else:
        print("\n  [3/4] CLIP — skipped")
    if phi3_up:
        agent_rows.append(bench_phi3(args.phi3_runs))
    else:
        print("\n  [4/4] Phi-3 — skipped")

    # Phase 2 — end-to-end pipeline
    print("\n" + "=" * 66)
    print("  PHASE 2 — Pipeline Mode Benchmarks (End-to-End)")
    print("=" * 66)

    pipeline_rows = []
    if not args.skip_clip and obj_images:
        pipeline_rows.append(bench_single_mode(obj_images, args.runs))
    if yolo_up and not args.skip_clip and obj_images:
        pipeline_rows.append(bench_multi_mode(obj_images, args.runs))
    if ocr_up and phi3_up and ocr_images:
        pipeline_rows.append(bench_ocr_mode(ocr_images, args.runs))

    # Terminal summary
    print("\n" + "=" * 66)
    print("  AGENT SUMMARY")
    print("=" * 66)
    print(f"  {'Agent':<30} {'Avg ms':>8} {'p95 ms':>8}"
          f" {'FPS':>6} {'Acc':>7} {'Conf':>7}")
    print("  " + "-" * 62)
    for r in agent_rows:
        print(f"  {r['agent']:<30} {r['avg_ms']:>8.1f} {r['p95_ms']:>8.1f}"
              f" {r['fps']:>6.1f} {r['accuracy']:>6.1%} {r['avg_conf']:>6.1%}")

    print("\n" + "=" * 66)
    print("  PIPELINE SUMMARY")
    print("=" * 66)
    print(f"  {'Mode':<25} {'Avg ms':>8} {'p95 ms':>8}"
          f" {'FPS':>6} {'Acc':>7} {'RT?':>5}")
    print("  " + "-" * 62)
    for r in pipeline_rows:
        print(f"  {r['mode']:<25} {r['avg_ms']:>8.1f} {r['p95_ms']:>8.1f}"
              f" {r['fps']:>6.1f} {r['accuracy']:>6.1%} {r['realtime']:>5}")

    # Write outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if agent_rows:
        p = out_dir / f"e2e_{stamp}_agents.csv"
        write_agent_csv(p, agent_rows)
        print(f"\n  Agent CSV    : {p}")

    if pipeline_rows:
        p = out_dir / f"e2e_{stamp}_pipeline.csv"
        write_pipeline_csv(p, pipeline_rows)
        print(f"  Pipeline CSV : {p}")

    if agent_rows or pipeline_rows:
        p = out_dir / f"e2e_{stamp}.md"
        write_markdown(p, agent_rows, pipeline_rows, {
            "timestamp":     stamp,
            "object_images": len(obj_images),
            "ocr_images":    len(ocr_images),
            "warmup":        args.warmup,
            "runs":          args.runs,
        })
        print(f"  Markdown     : {p}")

    print(f"\n  Run visualize_pipeline.py to generate the heatmap.")
    print("=" * 66)


if __name__ == "__main__":
    main()