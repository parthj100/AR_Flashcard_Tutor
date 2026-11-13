# Lens prototype — running the OCR pipeline

The scan view reads a photo of a problem with OCR and hands the text to a local LLM:

- **EasyOCR** or **Pix2Text** reads the captured frame. Both run in one local FastAPI service ([`OCR/ocr_serve.py`](../OCR/ocr_serve.py), port 8766); a switch in the scan view picks which one.
- **Ollama + Phi-4-mini** writes the card from the transcript. Free, runs locally.

YOLO (detection) and CLIP (recognition) were removed in Fall 2026. Their code is in
[`archive/spring-2026-multi-agent/`](../archive/spring-2026-multi-agent/).

The rest of the UI works from the cards in `data.js`, so the prototype is usable even with the services down.

---

## One-time setup

### 1. Python environment for the OCR service

From the repo root:

Use Python 3.12. Pix2Text needs `onnxruntime`, which has no build for Python 3.15.

```bash
brew install python@3.12            # if you don't have it
python3.12 -m venv .venv
.venv/bin/pip install -r OCR/requirements.txt
```

The first start downloads the EasyOCR and Pix2Text model files. Pix2Text loads in the background after
the service starts (about 10 s).

In VS Code, pick this interpreter with **Python: Select Interpreter → `.venv`** so it stops flagging the packages as missing.

### 2. Install Ollama and pull the model

macOS / Windows: download from [ollama.com](https://ollama.com/download).

```bash
ollama pull phi4-mini
```

To try another model, change `DEFAULT_MODEL` in `prototype/lib/llm.js`.

### 3. Start Ollama with CORS enabled

The browser calls `http://localhost:11434` from `http://localhost:5500`, and Ollama blocks cross-origin requests by default:

```bash
OLLAMA_ORIGINS='*' ollama serve
```

If Ollama is already running as a background app, quit it first or the setting has no effect.

---

## Running the prototype

Each in its own terminal, from the repo root:

```bash
OLLAMA_ORIGINS='*' ollama serve                     # LLM      :11434
.venv/bin/python OCR/ocr_serve.py                   # OCR      :8766
python3 -m http.server 5500 --directory prototype   # web app  :5500
```

The prototype must be served over http(s), not opened as `file://`, or the browser blocks the camera.
Open <http://localhost:5500> and grant camera permission.

---

## What you should see

1. Click **New scan** (top of the sidebar) or press ⌘S.
2. The top-right pills show **Camera**, **EasyOCR · ready** (or **Pix2Text · ready**) and **phi4-mini · ready**.
3. Pick the OCR model with the **EasyOCR / Pix2Text** switch in the top bar. The choice is remembered.
4. Frame the problem inside the corners and press **space** (or click the white button).
5. The right-hand panel lists each line the model read with its confidence. Math from Pix2Text shows as LaTeX
   in a monospace font. Then the "Writing flashcard" overlay appears while Phi-4-mini writes the card (~20 s).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Camera permission denied" | Grant camera permission in the browser's site settings and reload. |
| "Camera requires HTTPS or http://localhost" | You opened `file://…/index.html`. Use the static server above. |
| "OCR offline" pill | `OCR/ocr_serve.py` isn't running, or isn't on port 8766. Check `curl http://127.0.0.1:8766/health`. |
| "Pix2Text · loading…" pill | It's still loading in the background. The pill turns green on its own. |
| "Pix2Text · not installed" pill | The venv is missing Pix2Text. Rebuild it on Python 3.12 and run `pip install -r OCR/requirements.txt`. |
| "LLM offline" pill | Ollama isn't reachable. Check `curl http://localhost:11434/api/tags`. |
| "LLM · pull phi4-mini" pill | Ollama is up but the model isn't installed. Run `ollama pull phi4-mini`. |
| Capture fails with a CORS error | Ollama was started without `OLLAMA_ORIGINS='*'`. Quit it and start it again with the variable. |
