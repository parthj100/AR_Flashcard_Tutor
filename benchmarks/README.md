# Photo pilot: running and scoring

This page covers the photo pilot. For the typeset-image benchmark, see
[`ocr_eval/README.md`](ocr_eval/README.md).

With the OCR service running, [`benchmarks/run_ocr.py`](run_ocr.py) sends every photo in a
manifest to one OCR tool, one at a time like the app, and writes its transcripts.
[`benchmarks/score_ocr.py`](score_ocr.py) then compares them with the checked reference
transcripts and reports CER, WER and critical errors by rubric category (number, decimal, sign or
operator, fraction or exponent, unit or currency).

```bash
python3 benchmarks/run_ocr.py --manifest manifest.csv --model easyocr --out ocr_easyocr.csv
python3 benchmarks/run_ocr.py --manifest manifest.csv --model pix2text --out ocr_pix2text.csv
python3 benchmarks/score_ocr.py --refs references.csv --preds ocr_easyocr.csv ocr_pix2text.csv
```

- `manifest.csv`: `image_id, problem_id, condition, path` (path relative to the manifest), plus optional
  `split` and metadata columns (device, lighting, angle, writer), which are carried through.
- `references.csv`: `problem_id, reference`, written in the style of
  [`benchmarks/REFERENCE_STYLE.md`](REFERENCE_STYLE.md). Pix2Text's LaTeX is converted to the
  same style before scoring, and its original output is kept in the `raw_output` column.

The runner sends a confidence threshold of 0, so neither model drops the words it is unsure of, and it
reports the model-loading warm-up separately. Each output row records the model, version, time per image
and any error.

- `ocr_*.csv` (the runner's output, or your own): `image_id, problem_id, model, condition, hypothesis`, where condition is
  `printed`, `handwritten`, `screen` or `synthetic`. Optional: `split` (`dev`/`test`), `energy_j`,
  `latency_ms`, plus any metadata columns (device, lighting, angle, writer), which are carried through.

Rows with `split=test` are skipped unless you pass `--include-test`, so the held-out set stays untouched
until the final run. Synthetic images are always reported separately from real photos. When `energy_j`
is present, the summary includes joules per image and joules per critical-free transcript. Each image's
row lists its critical errors (for example `number: '12' -> '512'`) so they can be checked against hand labels.

**Reading order.** EasyOCR sometimes returns a word after the rest of its line. The OCR service puts
EasyOCR's text boxes back into reading order ([`OCR/line_order.py`](../OCR/line_order.py)) and labels it
`easyocr+order` in results; the untouched order is kept in `raw_output`. Start it with `LENS_OCR_ORDER=0` to turn
this off.

**Energy.** On an Apple Silicon Mac, add `--energy` to `run_ocr.py` to record joules per image. Type
`sudo -v` in the same terminal first and close other apps:

```bash
sudo -v
.venv/bin/python benchmarks/run_ocr.py --manifest manifest.csv --model pix2text --out ocr_pix2text.csv --energy
```

It reads the chip's power (CPU + GPU + neural engine) from macOS's `powermetrics`, measures idle power
for 10 seconds, and writes `energy_j` (joules above idle) for each image. How the number is made and
what it leaves out are at the top of [`benchmarks/power.py`](power.py).

**Checking the scorer against people.** The critical-error rules are documented at the top of
[`benchmarks/scoring.py`](scoring.py). They are an automatic first pass.
[`benchmarks/label_agreement.py`](label_agreement.py) makes blank labeling sheets for two
people from a sample of transcripts, then reports how often they and the scorer agree (percent
agreement and Cohen's kappa):

```bash
.venv/bin/python benchmarks/label_agreement.py make --refs references.csv --preds ocr_easyocr.csv ocr_pix2text.csv --out-dir labels
.venv/bin/python benchmarks/label_agreement.py score --dir labels
```

Tests:

```bash
.venv/bin/python -m unittest discover -s benchmarks -p "test_*.py"
.venv/bin/python -m unittest discover -s benchmarks/ocr_eval -p "test_*.py"
.venv/bin/python -m unittest discover -s OCR -p "test_*.py"
```
