"""
Tests for benchmarks/power.py and the --energy option of benchmarks/run_ocr.py.
No administrator rights needed: a small script stands in for powermetrics
and a small local server stands in for the OCR service.

Run from the repo root:
  python3 -m unittest discover -s benchmarks -p "test_*.py" -v
"""

from __future__ import annotations

import csv
import io
import json
import shlex
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_ocr  # noqa: E402
from power import PowerMonitor, Sample, energy_between, parse_sample  # noqa: E402

SAMPLE = """*** Sampled system activity (Thu Nov 12 10:00:00 2026 -0500) (102.39ms elapsed) ***

**** Processor usage ****

E-Cluster HW active frequency: 1200 MHz
CPU Power: 1500 mW
GPU Power: 400 mW
ANE Power: 100 mW
Combined Power (CPU + GPU + ANE): 2000 mW
"""

# Stand-in for powermetrics: a steady 3000 mW reading every 50 ms.
FAKE_POWERMETRICS = """
import sys, time
while True:
    time.sleep(0.05)
    print("*** Sampled system activity (now) (50.00ms elapsed) ***")
    print("CPU Power: 2500 mW")
    print("GPU Power: 500 mW")
    print("Combined Power (CPU + GPU + ANE): 3000 mW", flush=True)
"""


class TestParsing(unittest.TestCase):
    def test_combined_line_used(self):
        elapsed, watts = parse_sample(SAMPLE)
        self.assertAlmostEqual(elapsed, 0.10239)
        self.assertAlmostEqual(watts, 2.0)

    def test_parts_added_when_no_combined_line(self):
        block = "\n".join(l for l in SAMPLE.splitlines() if not l.startswith("Combined"))
        self.assertAlmostEqual(parse_sample(block)[1], 2.0)

    def test_no_power_lines(self):
        self.assertIsNone(parse_sample("*** Sampled system activity (x) (100.0ms elapsed) ***\nnothing"))


class TestEnergyBetween(unittest.TestCase):
    def test_overlap_is_weighted(self):
        samples = [Sample(0.0, 1.0, 2.0), Sample(1.0, 2.0, 4.0), Sample(2.0, 3.0, 10.0)]
        joules, covered = energy_between(samples, 0.5, 2.0)
        self.assertAlmostEqual(joules, 2.0 * 0.5 + 4.0 * 1.0)
        self.assertAlmostEqual(covered, 1.5)

    def test_outside_window(self):
        self.assertEqual(energy_between([Sample(0.0, 1.0, 2.0)], 5.0, 6.0), (0.0, 0.0))


class TestMonitor(unittest.TestCase):
    def test_reads_fake_tool(self):
        m = PowerMonitor(50, [sys.executable, "-c", FAKE_POWERMETRICS])
        m.start()
        try:
            self.assertAlmostEqual(m.idle_watts(0.4), 3.0, places=3)
        finally:
            m.stop()

    def test_clear_error_when_tool_fails(self):
        m = PowerMonitor(50, [sys.executable, "-c", "import sys; sys.exit('no rights')"])
        with self.assertRaises(RuntimeError) as cm:
            m.start()
        self.assertIn("sudo -v", str(cm.exception))


class _FakeOcr(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"ok": True, "model": "fake", "version": "0"})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        time.sleep(0.3)
        self._send({"image_size": {"w": 1, "h": 1}, "inference_ms": 300.0, "raw_text": "2 + 2",
                    "lines": [{"text": "2 + 2", "confidence": 0.9,
                               "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}]})


class TestRunOcrEnergy(unittest.TestCase):
    def test_energy_columns_written(self):
        from PIL import Image
        server = HTTPServer(("127.0.0.1", 0), _FakeOcr)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            Image.new("RGB", (8, 8), "white").save(d / "a.jpg")
            (d / "fake_power.py").write_text(FAKE_POWERMETRICS, encoding="utf-8")
            (d / "manifest.csv").write_text(
                "image_id,problem_id,condition,path\ni1,p1,printed,a.jpg\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                run_ocr.main(["--manifest", str(d / "manifest.csv"),
                              "--endpoint", f"http://127.0.0.1:{server.server_port}",
                              "--out", str(d / "out.csv"), "--energy", "--idle-seconds", "0.3",
                              "--power-interval", "50",
                              "--power-cmd", f"{shlex.quote(sys.executable)} {shlex.quote(str(d / 'fake_power.py'))}"])
            with (d / "out.csv").open(encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            meta = json.loads((d / "out.meta.json").read_text())
        server.shutdown()
        server.server_close()
        # Steady 3 W for both idle and the request: about 0.9 J in total, about 0 J above idle.
        self.assertAlmostEqual(float(row["power_w"]), 3.0, places=1)
        self.assertAlmostEqual(float(row["energy_total_j"]), 0.9, delta=0.15)
        self.assertLess(float(row["energy_j"]), 0.05)
        self.assertAlmostEqual(meta["idle_w"], 3.0, places=2)


if __name__ == "__main__":
    unittest.main()
