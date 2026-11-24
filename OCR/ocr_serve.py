"""
OCR service for the Lens prototype.

Serves every installed OCR model behind one endpoint, so the app (and the
benchmarks) can read the same photo with EasyOCR or Pix2Text and compare.

Routes:
  GET  /health  — server status and each model's name, version, settings
  POST /ocr     — read an image: {image, model?, conf?}
                  model defaults to "easyocr"; see engines.py for the list

Run with:
    pip install -r OCR/requirements.txt
    python OCR/ocr_serve.py

Defaults to http://localhost:8766. Permits all origins so the prototype
served from http://localhost:5500 can call it from the browser.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from engines import ENGINES, available_engines, get_engine

# --- Config ---------------------------------------------------------------

OCR_CONF      = float(os.environ.get("LENS_OCR_CONF", "0.5"))
PORT          = int(os.environ.get("LENS_OCR_PORT", "8766"))
DEFAULT_MODEL = os.environ.get("LENS_OCR_MODEL", "easyocr")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ocr")

# --- App ------------------------------------------------------------------

app = FastAPI(title="Lens OCR server", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the default model now; load the others in the background so the
# first capture with them doesn't wait on model loading.
get_engine(DEFAULT_MODEL).load()


def _preload_others() -> None:
    for name in available_engines():
        if name != DEFAULT_MODEL:
            try:
                get_engine(name).load()
            except Exception as e:  # keep serving the models that did load
                log.warning("Could not load %s: %s", name, e)


threading.Thread(target=_preload_others, daemon=True).start()


# --- Schema ---------------------------------------------------------------

class OcrRequest(BaseModel):
    image: str             = Field(..., description="data URL or raw base64 JPEG/PNG")
    model: Optional[str]   = Field(None, description=f"one of {list(ENGINES)}")
    conf:  Optional[float] = Field(None, ge=0, le=1)


class OcrLine(BaseModel):
    text:       str
    confidence: float
    bbox:       dict
    kind:       str = "text"     # "text", "math", or "mixed"


class OcrResponse(BaseModel):
    model:          str
    model_version:  str
    model_settings: dict
    image_size:     dict
    inference_ms:   float
    raw_text:       str
    lines:          List[OcrLine]


# --- Helpers --------------------------------------------------------------

def decode_b64_image(s: str) -> Image.Image:
    if s.startswith("data:"):
        _, b64 = s.split(",", 1)
    else:
        b64 = s
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# --- Routes ---------------------------------------------------------------

@app.get("/health")
def health():
    models = {name: get_engine(name).info() for name in available_engines()}
    return {
        "ok": True,
        "ocr": " · ".join(models),       # kept for older clients
        "default_model": DEFAULT_MODEL,
        "models": models,
        "port": PORT,
    }


@app.post("/ocr", response_model=OcrResponse)
def ocr(req: OcrRequest):
    try:
        im = decode_b64_image(req.image)
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}")

    name = req.model or DEFAULT_MODEL
    try:
        engine = get_engine(name)
    except KeyError as e:
        raise HTTPException(400, e.args[0])
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    conf = req.conf if req.conf is not None else OCR_CONF
    engine.load()                       # no-op once loaded; kept out of the timing
    t0    = time.perf_counter()
    lines = engine.read(im, conf=conf)
    dt_ms = (time.perf_counter() - t0) * 1000

    return OcrResponse(
        model=engine.name,
        model_version=engine.version,
        model_settings=engine.settings,
        image_size={"w": im.width, "h": im.height},
        inference_ms=dt_ms,
        raw_text=" ".join(l["text"] for l in lines),
        lines=[OcrLine(**l) for l in lines],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
