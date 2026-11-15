"""
Energy per photo for each OCR model, from macOS powermetrics (Apple silicon).

    .venv/bin/python benchmarks/ocr_eval/energy.py --types printed screen

Asks for your password once (powermetrics needs sudo). For each model:
  1. load + warm up the model,
  2. sample power for --idle seconds while doing nothing   -> idle mW
  3. sample power while reading every photo                -> active mW, seconds
  energy per photo (J) = (active mW - idle mW) / 1000 * seconds / photos

"Combined Power (CPU + GPU + ANE)" is what's sampled. Close other apps and
keep the Mac on the charger for repeatable numbers; run it 3 times and
report the median. Writes benchmarks/results/energy.json.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "OCR"))

from dataset import find_photos, load_problems     # noqa: E402
from run_ocr import load_image                     # noqa: E402

COMBINED = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")


class PowerSampler:
    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms

    def __enter__(self):
        self.out = Path(tempfile.mkstemp(suffix=".txt")[1])
        self.proc = subprocess.Popen(
            ["sudo", "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power",
             "-i", str(self.interval_ms), "-o", str(self.out)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)                                  # let the first sample start
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        subprocess.run(["sudo", "pkill", "-INT", "-x", "powermetrics"], check=False)
        self.proc.wait(timeout=10)
        samples = [float(x) for x in COMBINED.findall(self.out.read_text(errors="ignore"))]
        self.out.unlink(missing_ok=True)
        if not samples:
            raise RuntimeError("powermetrics produced no 'Combined Power' samples")
        self.mean_mw = statistics.fmean(samples)
        self.n_samples = len(samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--models", nargs="+", default=["easyocr", "pix2text"])
    ap.add_argument("--types", nargs="+", default=["printed", "screen", "handwritten"])
    ap.add_argument("--idle", type=float, default=15.0, help="seconds of idle baseline")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "energy.json")
    a = ap.parse_args()

    if subprocess.run(["sudo", "-v"]).returncode != 0:
        sys.exit("powermetrics needs sudo")
    from engines import get_engine

    photos = [p for p in find_photos(a.dataset, load_problems(a.dataset)) if p.photo_type in a.types]
    images = [load_image(p.path) for p in photos]          # decode up front: not part of OCR energy
    if not images:
        sys.exit("no photos")

    results = {}
    for m in a.models:
        eng = get_engine(m)
        eng.load()
        eng.read(images[0])                                 # warm-up
        with PowerSampler() as idle:
            time.sleep(a.idle)
        with PowerSampler() as active:
            for im in images:
                eng.read(im)
        joules = max(0.0, active.mean_mw - idle.mean_mw) / 1000 * active.seconds / len(images)
        results[m] = {"photos": len(images), "idle_mw": round(idle.mean_mw, 1),
                      "active_mw": round(active.mean_mw, 1), "seconds": round(active.seconds, 2),
                      "joules_per_photo": round(joules, 3), "samples": active.n_samples}
        print(f"{m:<10} idle {idle.mean_mw:7.0f} mW  active {active.mean_mw:7.0f} mW  "
              f"{active.seconds:6.1f} s  ->  {joules:.2f} J per photo")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"types": a.types, "results": results}, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
