# Lens prototype — running the OCR pipeline

The scan view reads a photo of a problem with OCR and hands the text to a local LLM:

- **EasyOCR** in a local FastAPI service ([`OCR/ocr_serve.py`](../OCR/ocr_serve.py), port 8766) reads the captured frame.
- **Ollama + Phi-4-mini** writes the card from the transcript. Free, runs locally.

YOLO (detection) and CLIP (recognition) were removed in Fall 2026. Their code is in
[`archive/spring-2026-multi-agent/`](../archive/spring-2026-multi-agent/).

The rest of the UI works from the cards in `data.js`, so the prototype is usable even with the services down.

---

## One-time setup

### 1. Python environment for the OCR service

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r OCR/requirements.txt
```

The first OCR request downloads the EasyOCR models (~64 MB).

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
2. The top-right pills show **Camera**, **OCR · ready** and **phi4-mini · ready**.
3. Frame the problem inside the corners and press **space** (or click the white button).
4. The right-hand panel lists each line EasyOCR read with its confidence, then the "Writing flashcard" overlay
   appears while Phi-4-mini writes the card (~5–20 s).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Camera permission denied" | Grant camera permission in the browser's site settings and reload. |
| "Camera requires HTTPS or http://localhost" | You opened `file://…/index.html`. Use the static server above. |
| "OCR offline" pill | `OCR/ocr_serve.py` isn't running, or isn't on port 8766. Check `curl http://127.0.0.1:8766/health`. |
| "LLM offline" pill | Ollama isn't reachable. Check `curl http://localhost:11434/api/tags`. |
| "LLM · pull phi4-mini" pill | Ollama is up but the model isn't installed. Run `ollama pull phi4-mini`. |
| Capture fails with a CORS error | Ollama was started without `OLLAMA_ORIGINS='*'`. Quit it and start it again with the variable. |
