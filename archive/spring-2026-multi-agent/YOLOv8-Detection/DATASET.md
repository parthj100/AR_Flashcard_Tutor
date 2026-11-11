# School Objects Dataset — specification

The dataset feeds (a) YOLOv8 fine-tuning and (b) the comparative detection
benchmark reported in Research Update 5. It is produced in two stages.

## Stage 1 — scraping

`collect_dataset.py` scrapes Bing image results via `icrawler`, groups the
results by class under `dataset/images/raw/<class>/`, and removes exact
duplicates by MD5 hash of the raw bytes. Each class has between two and four
query strings chosen to diversify framing, focus, and context.

```bash
python YOLOv8-Detection/collect_dataset.py                      # all classes
python YOLOv8-Detection/collect_dataset.py --expanded-only      # Update 5 additions only
python YOLOv8-Detection/collect_dataset.py --classes microscope,calculator --per-class 150
```

### Classes

| # | class                   | introduced in | query count |
|---|-------------------------|---------------|-------------|
| 0 | `textbook`              | Update 4      | 3           |
| 1 | `whiteboard`            | Update 4      | 3           |
| 2 | `desk_chair`            | Update 4      | 3           |
| 3 | `chemistry_flask`       | Update 4      | 3           |
| 4 | `laptop_computer`       | Update 4      | 3           |
| 5 | `ruler_pencil_pen`      | Update 4      | 3           |
| 6 | `microscope`            | Update 5      | 3           |
| 7 | `calculator`            | Update 5      | 3           |
| 8 | `backpack`              | Update 5      | 3           |
| 9 | `periodic_table_poster` | Update 5      | 3           |
| 10 | `globe_model`          | Update 5      | 3           |
| 11 | `safety_goggles`       | Update 5      | 3           |

Target per-class volume: 100 images, yielding roughly 1,200 images across the
full expanded corpus. The Update 4 benchmark was run on the first six classes
only, 591 images after dedupe.

## Stage 2 — annotation and export

Bounding boxes are drawn in Roboflow. The target export format is Ultralytics
YOLO, which writes one `.txt` per image containing lines of the form:

```
<class_id> <cx> <cy> <w> <h>
```

with every coordinate normalized to the unit square. Class IDs match the
indices in `dataset/data.yaml`. The split is 70/20/10 train/val/test,
stratified by class to avoid the long tail pulling mAP toward the dominant
classes. `data.yaml` must be kept in sync with the Roboflow export; when
Roboflow produces its own `data.yaml`, prefer that one and discard the file
committed here.

## Format summary

- Images: `.jpg` or `.webp`, arbitrary resolution, minimum 100x100.
- Labels: Ultralytics YOLO text format (one line per box, normalized).
- Splits: `images/{train,val,test}/`, `labels/{train,val,test}/`.
- Classes file: `data.yaml` (this repo) or `data.yaml` from Roboflow.

## Licensing

The scraped images are used for research under fair-use assumptions and are
not redistributed. `collect_dataset.py` is committed so the corpus can be
regenerated deterministically up to Bing's ranking non-determinism. The
annotated label `.txt` files we produce are original work and can be
redistributed under the project's license.

## Benchmark reproducibility

- Hardware: Apple M2, MPS backend.
- `benchmark.py` flags: `imgsz=640`, `conf=0.25`, `warmup=5`.
- Dataset at benchmark time: Update 4 baseline used classes 0 through 5 only,
  pre-annotation raw images. Update 5 comparative benchmark runs on the same
  six classes for comparability, plus the expanded 12-class split for the
  fine-tuned-checkpoint evaluation.

## Implicit corpus: quiz-grade logs

A secondary "dataset" accumulates from prototype usage: every quiz session
produces a `(card_id, correct, total, timestamp)` tuple recorded in
`RT.quizSessions`, and every generation records `(topic, flashcard_json,
generated=true)`. These together form the `(s, a, r)` triples referenced in
the algorithmic-framing section of the paper (imitation learning for
authored expert cards, offline RL for the generated-plus-feedback subset).
This corpus is currently session-local; persisting it to `localStorage`
or a small SQLite file is on the Update 5 next-steps list.
