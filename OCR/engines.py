"""
OCR engines behind one interface.

Each engine turns a PIL image into the same list of lines, so the OCR
service (ocr_serve.py) and the evaluation scripts can call any model the
same way and score them on identical images.

    from engines import get_engine, available_engines
    eng   = get_engine("pix2text")      # loads on first use
    lines = eng.read(pil_image)         # [{text, confidence, bbox, kind}, ...]

Line format (shared by every engine):
    text        str    what the model read. Math from Pix2Text is wrapped as
                       $...$ (inline) or $$...$$ (display) LaTeX.
    confidence  float  0-1. For Pix2Text, the lowest block score on the line,
                       so one shaky block flags the whole line.
    bbox        dict   {x1, y1, x2, y2} in image pixels
    kind        str    "text", "math", or "mixed"

Settings that affect results are recorded in `Engine.settings` and returned
with every response, so each saved transcript says exactly how it was made.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, List

import numpy as np
from PIL import Image

log = logging.getLogger("ocr.engines")


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _bbox(points) -> dict:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return {"x1": round(min(xs), 1), "y1": round(min(ys), 1),
            "x2": round(max(xs), 1), "y2": round(max(ys), 1)}


class Engine:
    name: str = ""
    package: str = ""
    settings: dict = {}

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()   # one inference at a time per model

    @classmethod
    def installed(cls) -> bool:
        return importlib.util.find_spec(cls.package) is not None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return _pkg_version(self.package)

    def info(self) -> dict:
        return {"name": self.name, "version": self.version,
                "settings": self.settings, "loaded": self.loaded}

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                log.info("Loading %s…", self.name)
                self._model = self._build()
                log.info("%s ready", self.name)

    def read(self, im: Image.Image, conf: float = 0.0) -> List[dict]:
        """Read an RGB image. Drops pieces with confidence below `conf`."""
        self.load()
        with self._lock:
            return self._read(im, conf)

    # Subclasses implement these two.
    def _build(self):
        raise NotImplementedError

    def _read(self, im: Image.Image, conf: float) -> List[dict]:
        raise NotImplementedError


class EasyOcrEngine(Engine):
    name = "easyocr"
    package = "easyocr"
    settings = {"languages": ["en"], "gpu": False}

    def _build(self):
        import easyocr
        return easyocr.Reader(self.settings["languages"],
                              gpu=self.settings["gpu"], verbose=False)

    def _read(self, im, conf):
        lines = []
        for points, text, score in self._model.readtext(np.array(im)):
            if score < conf:
                continue
            lines.append({"text": text.strip(),
                          "confidence": round(float(score), 4),
                          "bbox": _bbox(points),
                          "kind": "text"})
        return lines


class Pix2TextEngine(Engine):
    """Pix2Text text + formula OCR (math- and handwriting-aware).

    Runs on CPU: Pix2Text's formula detector crashes on Apple's MPS backend
    (torchvision NMS shader), and CPU keeps it comparable with EasyOCR.
    English only: with the default ('en', 'ch_sim') it misreads noise as
    Chinese and its English confidences drop to ~0.5.
    Spell checker off: we score raw OCR, separately from any cleanup.
    """
    name = "pix2text"
    package = "pix2text"
    settings = {"languages": ["en"], "device": "cpu",
                "spell_checker": False, "resized_shape": 768}

    def _build(self):
        from pix2text import TextFormulaOCR
        return TextFormulaOCR.from_config(
            total_configs={"languages": tuple(self.settings["languages"])},
            enable_spell_checker=self.settings["spell_checker"],
            device=self.settings["device"],
        )

    def _read(self, im, conf):
        blocks = self._model.recognize(
            im, return_text=False, resized_shape=self.settings["resized_shape"])

        # Pix2Text returns blocks (text pieces and formulas) tagged with a
        # line number. Group them into lines, left to right.
        by_line: Dict[int, list] = {}
        for b in blocks:
            score = float(b.get("score", 0.0))
            if score < conf or not b.get("text", "").strip():
                continue
            by_line.setdefault(int(b.get("line_number", 0)), []).append(b)

        lines = []
        for _, bs in sorted(by_line.items()):
            bs.sort(key=lambda b: float(np.min(np.asarray(b["position"])[:, 0])))
            parts, kinds = [], set()
            for b in bs:
                t = b["text"].strip()
                if b["type"] == "embedding":
                    parts.append(f"${t}$"); kinds.add("math")
                elif b["type"] == "isolated":
                    parts.append(f"$${t}$$"); kinds.add("math")
                else:
                    parts.append(t); kinds.add("text")
            points = np.concatenate([np.asarray(b["position"]) for b in bs])
            lines.append({
                "text": " ".join(parts),
                "confidence": round(min(float(b.get("score", 0.0)) for b in bs), 4),
                "bbox": _bbox(points),
                "kind": kinds.pop() if len(kinds) == 1 else "mixed",
            })
        return lines


ENGINES = {cls.name: cls for cls in (EasyOcrEngine, Pix2TextEngine)}
_instances: Dict[str, Engine] = {}
_instances_lock = threading.Lock()


def available_engines() -> List[str]:
    """Engines whose Python package is installed."""
    return [n for n, cls in ENGINES.items() if cls.installed()]


def get_engine(name: str) -> Engine:
    if name not in ENGINES:
        raise KeyError(f"unknown OCR model '{name}'; choose from {list(ENGINES)}")
    if not ENGINES[name].installed():
        raise RuntimeError(f"OCR model '{name}' is not installed (pip install {ENGINES[name].package})")
    with _instances_lock:
        if name not in _instances:
            _instances[name] = ENGINES[name]()
        return _instances[name]
