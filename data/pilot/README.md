# Pilot capture kit: 30 problems × 4 conditions

| | |
|---|---|
| Problems | 20 GSM8K (money, decimals, percents, fractions, plain) + 10 MATH (fractions, exponents, roots, π, inequalities). All MIT; see [`../../LICENSES.md`](../../LICENSES.md) |
| Split | All development set. Nothing here is the held-out test set |
| Conditions | printed, handwritten, screen (photos you take) + synthetic (already generated in `synthetic/`) |
| Machine for all runs | This Apple M5 MacBook |

## Files

| File | What it is |
|---|---|
| `problems.jsonl` | The 30 problems with source dataset, row, and the text as shown |
| `references.csv` | The correct transcript of each problem, in [`REFERENCE_STYLE.md`](../../benchmarks/REFERENCE_STYLE.md) style. This is what OCR is scored against |
| `sheets.pdf` | One problem per page: print it for the printed condition, copy from it for handwritten |
| `screen.html` | One problem at a time on a monitor for the screen condition (open in a browser; ← → or space) |
| `synthetic/` | 30 degraded renders and the settings used for each (`params.csv`) |
| `manifest.csv` | One row per photo. Synthetic rows are done; photo rows are filled in by the import script |
| `photos/` | Your imported photos (created by the import script) |

Rebuild everything with `data/tools/select_pilot.py` then `data/tools/build_kit.py`. The selection is
fixed (seed 2026), so you get the same 30 problems.

## Taking the photos (about 1.5–2 hours in total)

**Same for every condition**
- Use one phone for everything, and say which in the import command.
- Photograph in order, **shot 1 to 30**. The import script matches photos to problems by the time they
  were taken, so order is what matters, not filenames.
- Keep each problem's text fully in frame, holding the phone roughly straight above (or straight on for
  the screen). Normal room light; don't fix bad shots with editing apps.
- **Keep problem IDs out of the photo.** OCR would read their digits as extra numbers.
- Delete any retakes so there are exactly 30 photos per condition (or import with `--shots` for a part).

**Printed:** print `sheets.pdf` (skip the cover page) and photograph each page above the dashed line.

**Handwritten:** copy each problem onto its own blank sheet exactly as printed: same words, numbers and
symbols, any line breaks. Write the shot number and ID **on the back**. Split the sheets between you
(for example shots 1–15 and 16–30) so "writer" varies. If you mis-copy something, leave it, and put the
shot number and what differs in the manifest's `notes` column. The reference gets corrected to match the
paper, so OCR isn't blamed for a copying slip.

**Screen:** open `screen.html` on a laptop or monitor, full screen, and photograph each problem. Keep the
bottom bar (with the ID) out of the shot.

## Importing

AirDrop each condition's photos into its own folder, then (from the repo root):

```bash
.venv/bin/python data/tools/import_photos.py --condition printed --from ~/Downloads/printed --device "iPhone 15" --lighting "room light" --angle overhead
```

Add `--writer "Parth" --shots 1-15` for one person's handwritten sheets, and `--dry-run` to check the
matching first. Each import rotates photos upright, **removes location and all other metadata**, saves
them to `photos/`, fills in the manifest, and makes `photos/contact_<condition>.jpg` so you can check the
order at a glance.

## Running and scoring

Start the OCR service (see the main README), then:

```bash
python3 benchmarks/run_ocr.py --manifest data/pilot/manifest.csv --model easyocr --out data/pilot/results/ocr_easyocr_order.csv
python3 benchmarks/run_ocr.py --manifest data/pilot/manifest.csv --model pix2text --out data/pilot/results/ocr_pix2text.csv
python3 benchmarks/score_ocr.py --refs data/pilot/references.csv --preds data/pilot/results/ocr_easyocr_order.csv data/pilot/results/ocr_pix2text.csv --out-dir data/pilot/results
```

Rows without a photo yet are skipped, so this works with just the synthetic set.
