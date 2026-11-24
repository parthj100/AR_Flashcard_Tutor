# OCR energy benchmark

Measures how much energy EasyOCR and Pix2Text use to read one image, on an Apple silicon Mac.
Accuracy (CER, WER, critical errors) is scored by Parth's pipeline on the
`critical-error-benchmark` branch (`benchmarks/scoring.py`, `benchmarks/score_ocr.py`), using the
pilot photos in `data/pilot/`.

| File | What it does |
|---|---|
| `energy.py` | Runs both models over every image in a `manifest.csv` while `powermetrics` records chip power; writes `benchmarks/results/energy*.json` |
| `plots.py` | Draws `benchmarks/figures/energy*.png` from those results |
| `fake_powermetrics.py` | Stand-in power meter for testing without sudo |
| `test_energy.py` | Tests for the parsing and energy arithmetic |

## Running

```bash
.venv/bin/python benchmarks/ocr_eval/energy.py --manifest data/pilot/manifest.csv \
    --types printed screen handwritten --out benchmarks/results/energy_pilot.json
.venv/bin/python benchmarks/ocr_eval/plots.py
```

It asks for your password once (`powermetrics` needs sudo). Before running: plug in the charger,
close other apps, and leave the Mac alone. The pilot photos take about an hour (73 photos ×
2 models × 3 rounds); EasyOCR takes about 14 s per 2048×1536 photo.

Tests: `cd benchmarks/ocr_eval && ../../.venv/bin/python -m unittest test_energy`

## Method

For each model, in each round:

1. Load and warm up the model (not measured).
2. Record an idle baseline for 20 s: mean power `P_idle`.
3. Record power while the model reads every image. Energy is each power sample times its own
   length, summed; energy above idle subtracts `P_idle` × the time sampled.
4. Energy per image = energy above idle ÷ number of images.

Rounds alternate which model goes first, so heat and background activity affect both equally.
We report the median and range across 3 rounds. Power is `powermetrics`' "Combined Power
(CPU + GPU + ANE)": the chip only, not the screen, memory or the rest of the Mac.

Both models run on the CPU, with one small exception: on a Mac, onnxruntime runs Pix2Text's
text-region detector through Apple's CoreML. Forcing it onto the CPU changed Pix2Text's speed by
about 1% (0.624 vs 0.632 s/image), so it doesn't explain the gap between the models. Either way,
the energy figure counts the CPU, GPU and Neural Engine together.

## Results (Apple M2 MacBook Air, 8 GB)

| Image set | Model | Energy per image (median, range) | Time per image |
|---|---|---|---|
| **Pilot photos** (73: 30 printed, 30 screen, 13 handwritten) — `energy_pilot.json` | EasyOCR | 70.1 J (65.0–71.1) | 13.9 s |
| | Pix2Text | 13.5 J (13.0–15.2) | 0.97 s |
| Rendered images (60, first run; image set since removed) — `energy.json` | EasyOCR | 30.6 J (27.7–31.2) | 6.3 s |
| | Pix2Text | 11.8 J (11.5–12.0) | 0.76 s |

EasyOCR reads photos at full size, so its energy grows with image size. Pix2Text resizes
every image to 768 px first, so its energy barely changes between the two sets.
