"""
Stand-in for `powermetrics`, for testing energy.py without sudo:

    LENS_POWERMETRICS=".venv/bin/python benchmarks/ocr_eval/fake_powermetrics.py" \\
        .venv/bin/python benchmarks/ocr_eval/energy.py --limit 2 --repeats 1 --idle 2

Writes powermetrics-format samples at a constant power (FAKE_MW, default 2000)
to the -o file every -i milliseconds until interrupted. With constant power,
energy above idle should come out near 0.
"""

import argparse
import os
import signal
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--samplers")
ap.add_argument("-i", type=int, default=100)
ap.add_argument("-o", required=True)
a = ap.parse_args()
mw = float(os.environ.get("FAKE_MW", "2000"))

running = True
signal.signal(signal.SIGINT, lambda *_: globals().update(running=False))
with open(a.o, "w") as f:
    last = time.perf_counter()
    while running:
        time.sleep(a.i / 1000)
        now = time.perf_counter()
        f.write(f"*** Sampled system activity (fake) ({(now - last) * 1000:.2f}ms elapsed) ***\n\n"
                f"CPU Power: {mw * 0.9:.0f} mW\nGPU Power: {mw * 0.1:.0f} mW\nANE Power: 0 mW\n"
                f"Combined Power (CPU + GPU + ANE): {mw:.0f} mW\n\n")
        f.flush()
        last = now
sys.exit(0)
