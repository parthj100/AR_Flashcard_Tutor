# YOLOv8 multi-object mode

Opt-in scan mode for the Lens prototype. YOLO runs on the capture frame and
returns bounding boxes + class names + cropped JPEGs for each object. The
browser then re-identifies each crop against the Lens vocabulary with CLIP
(same zero-shot setup as the default mode) and asks the user to pick one.
Picking an object falls into the existing authored / Phi-3 generated flashcard
flow, so generation still works for unknown topics.

Nothing about the default (CLIP-only, single-object) mode is changed. YOLO is
strictly an additional path reachable via the mode toggle in the scan view.

## Running

```bash
# one-time
python3 -m pip install -r YOLOv8-Detection/requirements.txt

# each run
python3 YOLOv8-Detection/serve.py
```

Defaults:

| env var              | default                | notes                                    |
|----------------------|------------------------|------------------------------------------|
| `LENS_YOLO_MODEL`    | `yolov8n.pt`           | Any Ultralytics checkpoint path.         |
| `LENS_YOLO_CONF`     | `0.25`                 | Min detection confidence.                |
| `LENS_YOLO_IMG_SIZE` | `640`                  | Input size passed to the model.          |
| `LENS_YOLO_PAD`      | `0.08`                 | Box expansion before cropping.           |
| `LENS_YOLO_MAX_DETS` | `8`                    | Cap on boxes returned per frame.         |
| `LENS_YOLO_PORT`     | `8765`                 | Server port.                             |

The server auto-picks MPS (Apple Silicon), CUDA, or CPU in that order. On an
M2 the baseline benchmark shows ~215 ms per 640 px frame, which is fine for
capture-time use (one shot) but too slow for a live loop. That's why the
live detection loop still uses CLIP in the browser.

## Endpoints

### `GET /health`
Returns model + device info.

### `POST /detect`
Body:
```json
{ "image": "data:image/jpeg;base64,...", "conf": 0.3, "max_dets": 6 }
```

Response:
```json
{
  "image_size": { "w": 960, "h": 540 },
  "inference_ms": 214.8,
  "device": "mps",
  "detections": [
    {
      "class_name": "laptop",
      "confidence": 0.81,
      "box":     { "x": 42, "y": 80, "w": 410, "h": 260 },
      "box_rel": { "x": 0.044, "y": 0.148, "w": 0.427, "h": 0.481 },
      "crop": "data:image/jpeg;base64,..."
    }
  ]
}
```

## Why YOLO plus CLIP (not YOLO alone)

- YOLO's pretrained classes (COCO, 80 classes) only overlap with a sliver of
  the Lens vocabulary. Using YOLO for classification alone would regress
  recognition coverage.
- YOLO is great at localization. Running CLIP on each crop reuses the existing
  vocabulary index and keeps zero-shot behaviour for new `extendedVocab`
  entries — no retraining required when you add a topic.
- If you do fine-tune YOLO on Lens-specific classes later, the server contract
  does not need to change; swap `LENS_YOLO_MODEL` to the new checkpoint.

## Troubleshooting

| Symptom                       | Fix                                                                         |
|-------------------------------|------------------------------------------------------------------------------|
| `ModuleNotFoundError: torch`  | `pip install -r requirements.txt` (pulls the correct torch via ultralytics).|
| CORS error in the browser     | Make sure you hit `http://127.0.0.1:8765` from a page on `http://localhost:5500`. CORS is already `*`. |
| First detect call takes 8-15s | Model warmup — one-shot cost per server process. Subsequent calls are ~200 ms. |
| Empty detections list         | Lower `LENS_YOLO_CONF` or point the camera at a COCO-ish object (laptop, book, cup). |
