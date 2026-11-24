"""
Energy per image for each OCR model, measured with macOS powermetrics (Apple silicon).

    # the pilot photos (data/pilot/manifest.csv, from the critical-error-benchmark branch)
    .venv/bin/python benchmarks/ocr_eval/energy.py --manifest data/pilot/manifest.csv \
        --types printed screen handwritten --out benchmarks/results/energy_pilot.json

Asks for your password once (powermetrics needs sudo) and keeps sudo alive
for the whole run. Before starting: plug in the charger, close other apps,
turn off Wi-Fi sync / backups, and don't touch the Mac while it runs
(about 1 hour for 73 phone photos x 2 models x 3 rounds).

Method, per model, per round:
  1. load + warm up the model (untimed, not measured)
  2. idle baseline: sample power for --idle seconds doing nothing     -> P_idle (W)
  3. workload: sample power while the model reads every image
       E_total = sum over samples of  P_i x dt_i        (each sample's own length)
       E_ocr   = E_total - P_idle x (sum of dt_i)       energy above idle
  energy per image (J) = E_ocr / images
Rounds alternate the models (A, B, A, B, ...) so heat and background
activity hit both equally; we report the median and range across rounds.

Power is powermetrics' "Combined Power (CPU + GPU + ANE)": the chip, not
the screen, RAM or the rest of the Mac.

Writes the --out JSON (plots.py draws a graph from every energy*.json).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "OCR"))

from PIL import Image, ImageOps                    # noqa: E402

# For testing without sudo: LENS_POWERMETRICS="python fake_powermetrics.py"
FAKE = os.environ.get("LENS_POWERMETRICS")
CMD = FAKE.split() if FAKE else ["sudo", "-n", "powermetrics"]

_BLOCK = re.compile(r"\*\*\* Sampled system activity .*?\(([\d.]+)ms elapsed\) \*\*\*(.*?)(?=\*\*\* Sampled|\Z)", re.S)
_COMBINED = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")


def parse_samples(text: str):
    """[(seconds, watts), ...] from powermetrics text output, one pair per complete sample."""
    out = []
    for elapsed_ms, body in _BLOCK.findall(text):
        m = _COMBINED.search(body)
        if m:
            out.append((float(elapsed_ms) / 1000, float(m[1]) / 1000))
    return out


def energy_above_idle(samples, idle_w: float):
    """(joules above idle, seconds sampled, total joules)."""
    secs = sum(dt for dt, _ in samples)
    total = sum(dt * w for dt, w in samples)
    return total - idle_w * secs, secs, total


def load_image(path: Path) -> Image.Image:
    """Upright RGB image (phone photos carry their rotation in EXIF)."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def manifest_photos(manifest: Path):
    """[(path, condition), ...] for rows of a pilot manifest.csv whose photo exists."""
    out = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = manifest.parent / row["path"] if row.get("path") else None
            if path and path.is_file():
                out.append((path, row["condition"]))
    return out


class PowerSampler:
    """Run powermetrics for the duration of a `with` block."""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms

    def __enter__(self):
        fd, name = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        self.out = Path(name)
        self.proc = subprocess.Popen(
            CMD + ["--samplers", "cpu_power,gpu_power,ane_power", "-i", str(self.interval_ms),
                   "-o", str(self.out)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(0.5)
        if self.proc.poll() is not None:
            raise RuntimeError("powermetrics didn't start: " + self.proc.stderr.read().decode().strip())
        return self

    def __exit__(self, *exc):
        time.sleep(self.interval_ms / 1000 * 1.5)        # let the last sample finish
        if FAKE:
            self.proc.send_signal(signal.SIGINT)
        else:
            subprocess.run(["sudo", "-n", "pkill", "-INT", "-x", "powermetrics"], check=False)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.samples = parse_samples(self.out.read_text(errors="ignore"))
        self.out.unlink(missing_ok=True)
        if not self.samples:
            raise RuntimeError("powermetrics produced no 'Combined Power' samples")
        self.mean_w = sum(dt * w for dt, w in self.samples) / sum(dt for dt, _ in self.samples)


def keep_sudo_alive(stop: threading.Event):
    while not stop.wait(60):
        subprocess.run(["sudo", "-n", "-v"], check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="manifest.csv listing the images (path and condition columns)")
    ap.add_argument("--models", nargs="+", default=["easyocr", "pix2text"])
    ap.add_argument("--types", nargs="+", help="conditions to use, e.g. printed screen handwritten (default: all)")
    ap.add_argument("--limit", type=int, help="use only the first N images (quick check)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--idle", type=float, default=20.0, help="seconds of idle baseline per measurement")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "energy.json")
    a = ap.parse_args()

    photos = manifest_photos(a.manifest.resolve())
    if a.types:
        photos = [p for p in photos if p[1] in a.types]
    photos = photos[:a.limit] if a.limit else photos
    if not photos:
        sys.exit("no images found")
    types = sorted({t for _, t in photos})

    stop = threading.Event()
    if not FAKE:
        if subprocess.run(["sudo", "-v"]).returncode != 0:
            sys.exit("powermetrics needs sudo")
        threading.Thread(target=keep_sudo_alive, args=(stop,), daemon=True).start()

    from engines import get_engine
    images = [load_image(path) for path, _ in photos]          # decode up front: not part of OCR energy
    engines = {}
    for m in a.models:
        engines[m] = get_engine(m)
        engines[m].load()
        engines[m].read(images[0])                         # warm-up
    print(f"{len(images)} images ({', '.join(types)}), models {a.models}, {a.repeats} rounds")

    runs = {m: [] for m in a.models}
    try:
        for r in range(1, a.repeats + 1):
            order = a.models if r % 2 else list(reversed(a.models))
            for m in order:
                with PowerSampler() as idle:
                    time.sleep(a.idle)
                t0 = time.perf_counter()
                with PowerSampler() as work:
                    for im in images:
                        engines[m].read(im)
                wall = time.perf_counter() - t0
                above, secs, total = energy_above_idle(work.samples, idle.mean_w)
                run = {"round": r, "idle_w": round(idle.mean_w, 3), "active_w": round(work.mean_w, 3),
                       "seconds": round(wall, 2), "sampled_seconds": round(secs, 2),
                       "joules_per_image": round(above / len(images), 4),
                       "total_joules_per_image": round(total / len(images), 4)}
                runs[m].append(run)
                print(f"round {r}  {m:<9} idle {idle.mean_w:5.2f} W  active {work.mean_w:5.2f} W  "
                      f"{wall:6.1f} s  ->  {run['joules_per_image']:.2f} J/image above idle")
    finally:
        stop.set()

    summary = {}
    for m, rs in runs.items():
        j = [x["joules_per_image"] for x in rs]
        summary[m] = {"joules_per_image_median": round(statistics.median(j), 4),
                      "joules_per_image_min": min(j), "joules_per_image_max": max(j),
                      "seconds_per_image_median": round(statistics.median(x["seconds"] for x in rs) / len(images), 3)}
        print(f"{m:<9} median {summary[m]['joules_per_image_median']:.2f} J/image "
              f"(range {min(j):.2f}-{max(j):.2f})")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "method": "powermetrics Combined Power (CPU+GPU+ANE), energy above idle baseline",
        "fake_sampler": bool(FAKE), "images": len(images), "types": types,
        "image_set": a.manifest.parent.name,
        "images_per_type": {t: sum(1 for _, x in photos if x == t) for t in types},
        "repeats": a.repeats, "idle_seconds": a.idle, "summary": summary, "runs": runs}, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
