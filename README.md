# Lens — Measuring Critical OCR Errors in Photographed Math Problems

Lens is a study app: a student photographs a math problem (printed, handwritten or on a screen),
OCR reads it, and a local LLM works from the transcript.

**Fall 2026 research question:** how accurately and efficiently can OCR read a math problem from a photo?
We compare EasyOCR with one math- and handwriting-aware OCR model on the same images, and measure
character and word error rate, *critical errors* (a changed number, decimal point, sign or operator,
fraction or exponent, or unit or currency symbol), latency, and energy per image.

Last semester's prototype ran four agents (YOLOv8n, CLIP, EasyOCR, Phi-3-mini). YOLO and CLIP have been
removed; that code, its benchmarks and its results are kept in
[`archive/spring-2026-multi-agent/`](archive/spring-2026-multi-agent/) for reference.

---

## What's in this repo

| Path | What it is |
|---|---|
| [`prototype/`](prototype/) | Vanilla-JS web app (Lens UI): dashboard, decks, scan and flashcard views. The scan view captures a frame, sends it to the OCR service and shows the lines it read. See [`prototype/SETUP.md`](prototype/SETUP.md). |
| [`OCR/`](OCR/) | EasyOCR FastAPI service ([`ocr_serve.py`](OCR/ocr_serve.py), port 8766), last semester's OCR speed benchmark ([`benchmark.py`](OCR/benchmark.py)), and five test images. |
| [`benchmarks/`](benchmarks/) | Shared service clients ([`agents.py`](benchmarks/agents.py)) and the LLM benchmark ([`benchmark_llm.py`](benchmarks/benchmark_llm.py)). |
| [`archive/spring-2026-multi-agent/`](archive/spring-2026-multi-agent/) | YOLO and CLIP code, multi-agent benchmarks, results, figures and methodology docs from Spring 2026. |
| [`Research Updates/`](Research%20Updates/) | Progress-update PDFs. |
| [`ALGORITHMS.md`](ALGORITHMS.md), [`CARDS.md`](CARDS.md) | Spring 2026 docs on the learning framing and the flashcard format. |

---

## Pipeline

```
 Photo ──► Preprocessing ──► OCR ──────────────► Flagging ──► Transcript ──► Saved record ··► Phi-4-mini ··► Verifier / tutor card
 printed    straighten,      EasyOCR vs. one     low-conf.    beside the     photo, raw OCR,   (later)        (later)
 handwritten contrast, crop, math/handwriting    words, odd   photo; student corrections,
 on a screen line order      model, same images  numbers      fixes errors   model version
```

Today the app runs **Photo → EasyOCR → lines shown in the panel → Ollama writes a card**. Preprocessing,
the second OCR model, flagging, the editable transcript and the saved record are being built this semester.

| Layer | Implementation |
|---|---|
| OCR | **EasyOCR** (English), FastAPI service on port 8766 |
| LLM | **Phi-4-mini** via Ollama (`phi4-mini`), JSON output constrained to a schema ([`prototype/lib/llm.js`](prototype/lib/llm.js)) |
| Frontend | Vanilla HTML / CSS / JS, hash-routed, no build step |
| Tooling | Python 3.10+ |

---

## Getting started

```bash
python3 -m venv .venv
.venv/bin/pip install -r OCR/requirements.txt
ollama pull phi4-mini
```

Then, each in its own terminal:

```bash
OLLAMA_ORIGINS='*' ollama serve                     # LLM      :11434
.venv/bin/python OCR/ocr_serve.py                   # OCR      :8766
python3 -m http.server 5500 --directory prototype   # web app  :5500
```

Open <http://localhost:5500> and click **New scan**. Details and troubleshooting are in
[`prototype/SETUP.md`](prototype/SETUP.md).

---

## Team

| Name | Role |
|---|---|
| **Alexis Juarez Gomez** | *(fill in Fall 2026 role)* |
| **Parthkumar Joshi** | *(fill in Fall 2026 role)* |

---

## References

- JaidedAI. *EasyOCR*. [GitHub](https://github.com/JaidedAI/EasyOCR)
- Microsoft. *Phi-4-mini*. [Hugging Face](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- Ollama. *Local LLM runtime*. [ollama.com](https://ollama.com/)
- Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems* (GSM8K). arXiv:2110.14168.
- Yuan, Y., et al. (2022). *Syntax-Aware Network for Handwritten Mathematical Expression Recognition* (HME100K). CVPR.
