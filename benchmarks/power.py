"""
Measure how much energy each OCR request uses on an Apple Silicon Mac.

macOS ships a tool called powermetrics that reports the chip's power draw
(CPU + GPU + neural engine) several times a second. It needs administrator
rights, so before a run with --energy type this once in the same terminal:

  sudo -v

How the number is made
----------------------
1. powermetrics prints a power reading (milliwatts) every `interval_ms`.
   Each reading is the average power over the time since the last one.
2. Before any OCR, the Mac is left alone for a few seconds. The average
   power over that time is the idle power: what the Mac uses doing nothing.
3. For one OCR request running from t0 to t1:
       energy = sum(power of each reading x how long it overlaps t0..t1)
       energy above idle = energy - idle power x (t1 - t0)
   Energy is in joules (1 joule = 1 watt for 1 second).

What it does and does not cover
-------------------------------
- Covers the chip only: CPU, GPU and neural engine. Not the screen, memory,
  storage or Wi-Fi.
- It is the whole chip, not one program, so close other apps first.
- Readings are 100 ms apart by default, so a request shorter than that is
  measured with only one or two readings. Averages over many images are
  more reliable than any single image's number.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass

DEFAULT_CMD = ["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-i", "{interval_ms}"]

_HEADER = re.compile(r"\*\*\* Sampled system activity .*\(([\d.]+)ms elapsed\)")
_POWER = re.compile(r"^(CPU|GPU|ANE|Combined) Power[^:]*:\s*([\d.]+)\s*mW", re.M)


@dataclass
class Sample:
    start: float   # seconds, time.perf_counter() clock
    end: float
    watts: float


def parse_sample(block: str) -> tuple[float, float] | None:
    """One powermetrics sample -> (elapsed seconds, watts), or None if it has no power lines."""
    head = _HEADER.search(block)
    found = {name: float(mw) for name, mw in _POWER.findall(block)}
    if not head or not found:
        return None
    mw = found["Combined"] if "Combined" in found else sum(found.get(k, 0.0) for k in ("CPU", "GPU", "ANE"))
    return float(head.group(1)) / 1000.0, mw / 1000.0


def energy_between(samples: list[Sample], t0: float, t1: float) -> tuple[float, float]:
    """(joules, seconds covered by readings) for the window t0..t1."""
    joules = covered = 0.0
    for s in samples:
        overlap = min(s.end, t1) - max(s.start, t0)
        if overlap > 0:
            joules += s.watts * overlap
            covered += overlap
    return joules, covered


class PowerMonitor:
    """Runs powermetrics in the background and keeps its readings."""

    def __init__(self, interval_ms: int = 100, cmd: list[str] | None = None):
        self.interval_ms = interval_ms
        self.cmd = [c.format(interval_ms=interval_ms) for c in (cmd or DEFAULT_CMD)]
        self.samples: list[Sample] = []
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True, bufsize=1)
        except FileNotFoundError as e:
            raise RuntimeError(f"cannot start power tool: {e}") from e
        threading.Thread(target=self._read, daemon=True).start()
        deadline = time.perf_counter() + 5
        while not self.samples and time.perf_counter() < deadline:
            if self._proc.poll() is not None:
                err = (self._proc.stderr.read() or "").strip()
                raise RuntimeError(
                    "powermetrics did not start"
                    + (f": {err}" if err else "")
                    + ". Run `sudo -v` in this terminal first, then try again.")
            time.sleep(0.05)
        if not self.samples:
            self.stop()
            raise RuntimeError("powermetrics started but gave no power readings in 5 s "
                               "(this script expects an Apple Silicon Mac).")

    def _read(self) -> None:
        block: list[str] = []
        for line in self._proc.stdout:
            if line.startswith("*** Sampled system activity") and block:
                self._add("".join(block))
                block = []
            block.append(line)
            # The combined line is the last power line of a sample: record it right away
            if line.startswith("Combined Power"):
                self._add("".join(block))
                block = []

    def _add(self, block: str) -> None:
        parsed = parse_sample(block)
        if parsed:
            elapsed, watts = parsed
            now = time.perf_counter()
            with self._lock:
                self.samples.append(Sample(now - elapsed, now, watts))

    def window(self, t0: float, t1: float, wait: float = 1.0) -> tuple[float, float]:
        """Energy for t0..t1, waiting briefly for the reading that covers t1 to arrive."""
        deadline = time.perf_counter() + wait
        while time.perf_counter() < deadline:
            with self._lock:
                if self.samples and self.samples[-1].end >= t1:
                    break
            time.sleep(0.01)
        with self._lock:
            return energy_between(self.samples, t0, t1)

    def idle_watts(self, seconds: float) -> float:
        """Average power while nothing is running, measured now."""
        t0 = time.perf_counter()
        time.sleep(seconds)
        joules, covered = self.window(t0, time.perf_counter())
        if covered <= 0:
            raise RuntimeError("no power readings during the idle measurement")
        return joules / covered

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._proc:
            for pipe in (self._proc.stdout, self._proc.stderr):
                if pipe:
                    pipe.close()
