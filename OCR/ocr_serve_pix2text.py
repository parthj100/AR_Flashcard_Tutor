"""
Pix2Text OCR endpoint for the Lens study: the math- and handwriting-aware
model compared against EasyOCR (OCR/ocr_serve.py).

The app uses Pix2Text through ocr_serve.py. This separate service is kept
because the photo pilot (data/pilot/results/ocr_pix2text.csv) was run with
it, and its settings differ from engines.py (spell checker on, default image
size). benchmarks/run_ocr.py works with either:

  GET  /health  — model, version and settings
  POST /ocr     — { image, conf? } -> { image_size, inference_ms, raw_text, raw_output, lines }

Pix2Text finds formulas in the image and returns them as LaTeX, and reads
the rest as plain text. The response gives both:
  raw_output  Pix2Text's own format: text with formulas in $...$ (embedded)
              or $$...$$ (on their own line)
  raw_text    the same with every formula converted to plain text
              (OCR/latex_plain.py); this is what gets scored

Settings (environment variables), reported by /health so runs can be locked:
  LENS_P2T_PORT     default 8767
  LENS_P2T_DEVICE   default cpu (EasyOCR also runs on the CPU); or mps
  LENS_P2T_SPELL    default 1: keep Pix2Text's built-in English spell checker
  LENS_OCR_CONF     default 0.0: minimum box confidence to keep

Run:
  .venv/bin/python OCR/ocr_serve_pix2text.py

Note: Pix2Text 1.1.7 needs rapidocr 3.9.x (pinned in OCR/requirements.txt);
rapidocr 3.10 renamed a function its dependency cnstd imports.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latex_plain import latex_to_plain  # noqa: E402

# --- Config ---------------------------------------------------------------

PORT = int(os.environ.get("LENS_P2T_PORT", "8767"))
DEVICE = os.environ.get("LENS_P2T_DEVICE", "cpu")
SPELL = os.environ.get("LENS_P2T_SPELL", "1") not in ("0", "false", "no")
OCR_CONF = float(os.environ.get("LENS_OCR_CONF", "0.0"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ocr-pix2text")

# --- App ------------------------------------------------------------------

app = FastAPI(title="Lens OCR server (Pix2Text)", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Pix2Text ----------------------------------------------------------

log.info("Loading Pix2Text (device=%s, spell checker=%s)…", DEVICE, SPELL)
from pix2text import TextFormulaOCR  # noqa: E402
from pix2text.__version__ import __version__ as P2T_VERSION  # pix2text.__version__ is a module

reader = TextFormulaOCR.from_config(
    total_configs={"languages": ("en",)},
    enable_formula=True,
    enable_spell_checker=SPELL,
    device=DEVICE,
)
log.info("Pix2Text %s ready", P2T_VERSION)


# --- Schema ---------------------------------------------------------------

class OcrRequest(BaseModel):
    image: str = Field(..., description="data URL or raw base64 JPEG/PNG")
    conf: Optional[float] = Field(None, ge=0, le=1)


class OcrLine(BaseModel):
    text: str             # plain text (formulas converted)
    confidence: float
    bbox: dict
    kind: str             # text | embedding (formula inside a line) | isolated (formula on its own line)
    latex: str = ""       # Pix2Text's LaTeX, for formulas
    line_number: int = 0


class OcrResponse(BaseModel):
    image_size: dict
    inference_ms: float
    raw_text: str
    raw_output: str
    lines: List[OcrLine]


# --- Helpers --------------------------------------------------------------

def decode_b64_image(s: str) -> Image.Image:
    b64 = s.split(",", 1)[1] if s.startswith("data:") else s
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def join_lines(boxes: list[OcrLine], field: str) -> str:
    """Boxes on the same line joined by spaces, lines joined by newlines."""
    lines: dict[int, list[str]] = {}
    for b in boxes:
        lines.setdefault(b.line_number, []).append(getattr(b, field).strip())
    return "\n".join(" ".join(p for p in parts if p) for _, parts in sorted(lines.items()))


# --- Routes ---------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "ocr": f"pix2text {P2T_VERSION} · en", "model": "pix2text",
            "version": P2T_VERSION, "device": DEVICE, "spell_checker": SPELL,
            "port": PORT}


@app.post("/ocr", response_model=OcrResponse)
def ocr(req: OcrRequest):
    try:
        im = decode_b64_image(req.image)
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}")

    conf_thresh = req.conf if req.conf is not None else OCR_CONF

    t0 = time.perf_counter()
    results = reader.recognize(im, return_text=False)
    dt_ms = (time.perf_counter() - t0) * 1000

    boxes: list[OcrLine] = []
    raw_parts: list[OcrLine] = []
    for r in results:
        score = float(r.get("score", 0.0))
        if score < conf_thresh:
            continue
        kind = r.get("type", "text")
        text = str(r.get("text", ""))
        pos = r.get("position")
        xs = [float(p[0]) for p in pos] if pos is not None else [0.0]
        ys = [float(p[1]) for p in pos] if pos is not None else [0.0]
        is_formula = kind in ("embedding", "isolated")
        line = OcrLine(
            text=latex_to_plain(text) if is_formula else text.strip(),
            confidence=round(score, 4),
            bbox={"x1": round(min(xs), 1), "y1": round(min(ys), 1),
                  "x2": round(max(xs), 1), "y2": round(max(ys), 1)},
            kind=kind,
            latex=text if is_formula else "",
            line_number=int(r.get("line_number", 0) or 0),
        )
        boxes.append(line)
        delim = {"embedding": "$", "isolated": "$$"}.get(kind, "")
        raw_parts.append(line.model_copy(update={"text": f"{delim}{text.strip()}{delim}"}))

    return OcrResponse(
        image_size={"w": im.width, "h": im.height},
        inference_ms=dt_ms,
        raw_text=join_lines(boxes, "text"),
        raw_output=join_lines(raw_parts, "text"),
        lines=boxes,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
