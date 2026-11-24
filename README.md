# Lens

Lens is a study app. A student takes a photo of a math problem, OCR reads the text, and a local
language model works from that text.

**Our question this semester:** how well can OCR read a math problem from a photo?

We compare two OCR tools on the same images: **EasyOCR** (general) and **Pix2Text** (built for math).
Our main measure is **critical errors**: misreads that change the math, such as a wrong number, decimal
point, sign, fraction, power or unit.

## What's new in this branch

- **Both tools in one service.** The app can switch between EasyOCR and Pix2Text.
- **Typeset benchmark.** 30 problems, each as a sharp and a blurred image (60 images).
- **Photo pilot.** 73 real photos: 30 printed, 30 on a screen, 13 handwritten.
- **Automatic scoring.** Counts critical errors by type.
- **Reading order.** Puts EasyOCR's words back in the right order.
- **Energy script.** Measures energy per image. Not run yet.
- **Hand-label sheets.** For checking the scoring by hand. Not filled in yet.
- **Licenses and related work.** See [LICENSES.md](LICENSES.md) and [RELATED_WORK.md](RELATED_WORK.md).

## Results so far

Share of images read with no critical errors:

| Images | EasyOCR | Pix2Text |
|---|---|---|
| Typeset, sharp (30) | 43% | 83% |
| Typeset, blurred (30) | 20% | 70% |
| Real photos (73) | 47% | 56% |

Neither tool read any of the 13 handwritten photos without a critical error.

The typeset and real-photo rows use different problems and scoring scripts, so don't compare them with
each other. The real-photo EasyOCR number uses the reading-order fix (45% without it).

## Run the app

You need Python 3.12 and [Ollama](https://ollama.com/).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r OCR/requirements.txt
ollama pull phi4-mini
```

Then run each of these in its own terminal and open <http://localhost:5500>:

```bash
OLLAMA_ORIGINS='*' ollama serve
.venv/bin/python OCR/ocr_serve.py
python3 -m http.server 5500 --directory prototype
```

Help: [prototype/SETUP.md](prototype/SETUP.md)

## Run the benchmarks

- **Typeset images:** [benchmarks/ocr_eval/README.md](benchmarks/ocr_eval/README.md)
- **Real photos:** [benchmarks/README.md](benchmarks/README.md)

## Run the tests

```bash
.venv/bin/python -m unittest discover -s benchmarks -p "test_*.py"
.venv/bin/python -m unittest discover -s benchmarks/ocr_eval -p "test_*.py"
.venv/bin/python -m unittest discover -s OCR -p "test_*.py"
```

## What's in each folder

| Folder | What it is |
|---|---|
| [prototype/](prototype/) | The web app |
| [OCR/](OCR/) | The OCR service |
| [benchmarks/](benchmarks/) | Scripts that run and score the OCR tools |
| [dataset/](dataset/) | Problems and images for the typeset benchmark |
| [data/pilot/](data/pilot/) | Problems, photos and results for the photo pilot |
| [Research Updates/](Research%20Updates/) | Progress reports |
| [archive/](archive/) | Spring 2026 work (YOLO, CLIP) |

[ALGORITHMS.md](ALGORITHMS.md) and [CARDS.md](CARDS.md) are also from Spring 2026.

## Team

| Name | Role |
|---|---|
| **Alexis Juarez Gomez** | *(fill in Fall 2026 role)* |
| **Parthkumar Joshi** | *(fill in Fall 2026 role)* |

## References

- JaidedAI. *EasyOCR*. [GitHub](https://github.com/JaidedAI/EasyOCR)
- Breezedeus. *Pix2Text*. [GitHub](https://github.com/breezedeus/Pix2Text)
- Microsoft. *Phi-4-mini*. [Hugging Face](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- Ollama. *Local LLM runtime*. [ollama.com](https://ollama.com/)
- Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems* (GSM8K). arXiv:2110.14168.
- Hendrycks, D., et al. (2021). *Measuring Mathematical Problem Solving With the MATH Dataset*. arXiv:2103.03874.
