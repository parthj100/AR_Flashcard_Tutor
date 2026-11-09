"""
YOLOv8 detection endpoint for the Lens prototype.

Exposes a single POST /detect route that accepts a base64-encoded image and
returns bounding boxes + class names + per-box base64 JPEG crops. The browser
pairs these with CLIP to identify what's in each box against our vocabulary,
then hands the selected object off to the existing flashcard flow.

Run with:
    python -m pip install fastapi uvicorn ultralytics pillow
    python YOLOv8-Detection/serve.py

Defaults to http://localhost:8765 and permits all origins so the prototype
(served from http://localhost:5500) can call it from the browser.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO

# --- Config ---------------------------------------------------------------

MODEL_PATH = os.environ.get("LENS_YOLO_MODEL", "yolov8n.pt")
CONF_THRESH = float(os.environ.get("LENS_YOLO_CONF", "0.25"))
IMG_SIZE = int(os.environ.get("LENS_YOLO_IMG_SIZE", "640"))
CROP_PADDING = float(os.environ.get("LENS_YOLO_PAD", "0.08"))  # box expansion %
MAX_DETS = int(os.environ.get("LENS_YOLO_MAX_DETS", "8"))

PORT = int(os.environ.get("LENS_YOLO_PORT", "8765"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("yolo")


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --- App + model ----------------------------------------------------------

app = FastAPI(title="Lens YOLO detect", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

DEVICE = pick_device()
log.info("Loading %s on %s", MODEL_PATH, DEVICE)
model = YOLO(MODEL_PATH)
model.to(DEVICE)
# Warm up so the first real request isn't glacial.
_warmup = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127))
model.predict(_warmup, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
log.info("YOLO ready")


# --- Schema ---------------------------------------------------------------


class DetectRequest(BaseModel):
    image: str = Field(..., description="data URL or raw base64 JPEG/PNG")
    conf: Optional[float] = Field(None, ge=0, le=1)
    max_dets: Optional[int] = Field(None, ge=1, le=50)


class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Detection(BaseModel):
    class_name: str
    confidence: float
    box: Box             # image-space pixels
    box_rel: Box         # 0..1 relative to image
    crop: str            # base64 JPEG crop (data: URL)


class DetectResponse(BaseModel):
    image_size: dict
    inference_ms: float
    device: str
    detections: List[Detection]


# --- Helpers --------------------------------------------------------------


def decode_b64_image(s: str) -> Image.Image:
    if s.startswith("data:"):
        # data:[mime];base64,<payload>
        _, b64 = s.split(",", 1)
    else:
        b64 = s
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def encode_jpeg_b64(im: Image.Image, quality: int = 82) -> str:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def padded_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int, pad: float):
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad
    py = bh * pad
    nx1 = max(0, int(x1 - px))
    ny1 = max(0, int(y1 - py))
    nx2 = min(w, int(x2 + px))
    ny2 = min(h, int(y2 + py))
    return nx1, ny1, nx2, ny2


# --- Routes ---------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_PATH,
        "device": DEVICE,
        "conf": CONF_THRESH,
        "img_size": IMG_SIZE,
    }


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    try:
        im = decode_b64_image(req.image)
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}")

    conf = req.conf if req.conf is not None else CONF_THRESH
    max_dets = req.max_dets if req.max_dets is not None else MAX_DETS

    t0 = time.perf_counter()
    results = model.predict(im, imgsz=IMG_SIZE, conf=conf, verbose=False, device=DEVICE)
    dt_ms = (time.perf_counter() - t0) * 1000

    if not results:
        return DetectResponse(
            image_size={"w": im.width, "h": im.height},
            inference_ms=dt_ms,
            device=DEVICE,
            detections=[],
        )

    r = results[0]
    names = r.names
    detections: list[Detection] = []

    if r.boxes is not None and len(r.boxes) > 0:
        # Sort by confidence desc, take top max_dets
        confs = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        order = confs.argsort()[::-1][:max_dets]

        for idx in order:
            x1, y1, x2, y2 = xyxy[idx].tolist()
            c = float(confs[idx])
            cls_id = int(cls_ids[idx])
            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]

            px1, py1, px2, py2 = padded_box(x1, y1, x2, y2, im.width, im.height, CROP_PADDING)
            crop = im.crop((px1, py1, px2, py2))
            # Keep crops reasonably small for network.
            crop.thumbnail((384, 384))
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=c,
                    box=Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    box_rel=Box(
                        x=x1 / im.width,
                        y=y1 / im.height,
                        w=(x2 - x1) / im.width,
                        h=(y2 - y1) / im.height,
                    ),
                    crop=encode_jpeg_b64(crop),
                )
            )

    return DetectResponse(
        image_size={"w": im.width, "h": im.height},
        inference_ms=dt_ms,
        device=DEVICE,
        detections=detections,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
