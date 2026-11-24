"""
Check the automatic critical-error scorer against people.

The scorer in scoring.py decides by rule whether an OCR transcript changed a
number, decimal point, sign/operator, fraction/exponent or unit/currency.
Before relying on it, two people label a sample by hand and we measure how
often the scorer and the people agree.

Step 1: make the labeling sheets (the scorer's answers are kept in a
separate key file so they cannot influence the labelers):

  python benchmarks/label_agreement.py make --refs data/pilot/references.csv \\
      --preds data/pilot/results/ocr_easyocr_order.csv data/pilot/results/ocr_pix2text.csv \\
      --out-dir data/pilot/labels --n 40

  This writes  labels_A.csv, labels_B.csv   one per person, same items, blank answers
               key.csv                      the scorer's answers. Do not open before labeling
               LABELING.md                  the instructions

Step 2: each person fills in their own sheet without looking at the other.

Step 3: score the agreement:

  python benchmarks/label_agreement.py score --dir data/pilot/labels

Agreement is reported two ways:
  - percent agreement: share of items where both gave the same answer
  - Cohen's kappa: agreement after removing what chance alone would give.
    1 is perfect, 0 is chance. Above about 0.8 is usually called strong.

The sample is picked with a fixed seed, spread evenly over model x condition,
and half of it (when possible) from transcripts the scorer flagged, so both
answers are well represented.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import CATEGORIES, critical_errors  # noqa: E402
from score_ocr import load_rows  # noqa: E402

SHEET_COLS = ["item", "reference", "ocr_output", "critical_error", *CATEGORIES, "comment"]
YES, NO = "y", "n"

INSTRUCTIONS = """# Labeling critical errors by hand

You have a sheet (`labels_A.csv` or `labels_B.csv`). Open it in Excel, Numbers or Google Sheets.
Each row shows the correct text (`reference`) and what the OCR tool read (`ocr_output`).

**Do not open `key.csv`, and do not compare answers with the other labeler until both are done.**

For each row:

1. `critical_error`: type `y` if the OCR output changed, dropped or added anything that would change the
   math. Otherwise type `n`.
2. If `y`, put `y` under each kind of error you see (leave the others blank):

| Column | Means | Example (correct -> OCR) |
|---|---|---|
| `number` | a number is different, missing or extra | 18 -> 13 |
| `decimal` | same digits, but the decimal point moved or vanished | 3.50 -> 350 |
| `sign_operator` | a minus sign or an operator (+ - x / = < > root) changed or vanished | 8 + 3 -> 8 - 3 |
| `fraction_exponent` | a fraction or power lost its shape | x^2 -> x2, 1/2 -> 12 |
| `unit_currency` | a $, %, degree sign or unit (cm, kg...) changed or vanished | $40 -> 40 |

**Not critical** (answer `n` if this is all that is wrong): spelling mistakes in words, capital letters,
punctuation, spacing, line breaks, "1,000" vs "1000", and different ways of writing the same thing
(`x^2` and `x²`, `3 × 4` and `3 * 4`).

Use `comment` for anything you were unsure about. Save as CSV with the same file name when done.
"""


def kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    """(percent agreement, Cohen's kappa) for two lists of y/n answers."""
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    agree = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = a.count(YES) / n, b.count(YES) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    return agree, (1.0 if agree == 1 else float("nan")) if chance == 1 else (agree - chance) / (1 - chance)


def cmd_make(args) -> int:
    refs = {r["problem_id"]: r["reference"] for r in load_rows(args.refs)}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for path in args.preds:
        for r in load_rows(path):
            if r.get("split", "").strip().lower() == "test" or r["problem_id"] not in refs:
                continue
            cats = sorted({e.category for e in critical_errors(refs[r["problem_id"]], r["hypothesis"])})
            groups[(r["model"], r["condition"])].append({**r, "reference": refs[r["problem_id"]], "cats": cats})
    if not groups:
        sys.exit("error: no transcripts to sample")

    rng = random.Random(args.seed)
    per_group = max(1, args.n // len(groups))
    picked: list[dict] = []
    for key in sorted(groups):
        flagged = [r for r in groups[key] if r["cats"]]
        clean = [r for r in groups[key] if not r["cats"]]
        rng.shuffle(flagged)
        rng.shuffle(clean)
        take_flagged = min(len(flagged), (per_group + 1) // 2)
        take_clean = min(len(clean), per_group - take_flagged)
        take_flagged = min(len(flagged), per_group - take_clean)
        picked += flagged[:take_flagged] + clean[:take_clean]
    rng.shuffle(picked)  # labelers cannot guess model or condition from the order

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("labels_A.csv", "labels_B.csv"):
        if (args.out_dir / name).exists() and not args.force:
            sys.exit(f"error: {args.out_dir / name} exists (labels may be filled in). Use --force to overwrite.")
    for name in ("labels_A.csv", "labels_B.csv"):
        with (args.out_dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SHEET_COLS)
            w.writeheader()
            for i, r in enumerate(picked, 1):
                w.writerow({"item": i, "reference": r["reference"], "ocr_output": r["hypothesis"]})
    with (args.out_dir / "key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "image_id", "model", "condition", "critical_error", *CATEGORIES])
        w.writeheader()
        for i, r in enumerate(picked, 1):
            w.writerow({"item": i, "image_id": r["image_id"], "model": r["model"], "condition": r["condition"],
                        "critical_error": YES if r["cats"] else NO,
                        **{c: (YES if c in r["cats"] else NO) for c in CATEGORIES}})
    (args.out_dir / "LABELING.md").write_text(INSTRUCTIONS, encoding="utf-8")
    flagged = sum(1 for r in picked if r["cats"])
    print(f"Wrote {len(picked)} items to {args.out_dir} ({flagged} flagged by the scorer, "
          f"{len(picked) - flagged} not), from {len(groups)} model x condition groups.")
    return 0


def _answers(path: Path) -> dict[str, dict[str, str]]:
    out = {}
    for r in load_rows(path):
        crit = (r.get("critical_error") or "").strip().lower()[:1]
        if crit not in (YES, NO):
            continue  # not labeled yet
        out[str(r["item"])] = {
            "critical_error": crit,
            **{c: (YES if (r.get(c) or "").strip().lower()[:1] == YES else NO) for c in CATEGORIES},
        }
    return out


def cmd_score(args) -> int:
    sets = {"scorer": _answers(args.dir / "key.csv")}
    for who in ("A", "B"):
        p = args.dir / f"labels_{who}.csv"
        if p.exists():
            got = _answers(p)
            if got:
                sets[f"labeler {who}"] = got
            print(f"labeler {who}: {len(got)} of {len(sets['scorer'])} items labeled")
    if len(sets) < 2:
        sys.exit("error: no filled-in labels yet. Type y or n in the critical_error column of each sheet.")

    names = list(sets)
    lines = ["# Agreement between the automatic scorer and hand labels", "",
             "| Pair | Items | What | Agree | Kappa |", "|---|---|---|---|---|"]
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            items = sorted(set(sets[x]) & set(sets[y]), key=int)
            for field in ("critical_error", *CATEGORIES):
                agree, k = kappa([sets[x][it][field] for it in items], [sets[y][it][field] for it in items])
                what = "any critical error" if field == "critical_error" else field
                lines.append(f"| {x} vs {y} | {len(items)} | {what} | {agree:.0%} | {k:.2f} |")
    for who in names[1:]:
        diff = [it for it in sorted(sets[who], key=int)
                if sets[who][it]["critical_error"] != sets["scorer"][it]["critical_error"]]
        lines += ["", f"Items where {who} and the scorer disagree on \"any critical error\": "
                  + (", ".join(diff) if diff else "none")]
    text = "\n".join(lines) + "\n"
    (args.dir / "agreement.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hand-label check for the critical-error scorer.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="write blank labeling sheets and the scorer's key")
    mk.add_argument("--refs", type=Path, required=True)
    mk.add_argument("--preds", type=Path, nargs="+", required=True)
    mk.add_argument("--out-dir", type=Path, required=True)
    mk.add_argument("--n", type=int, default=40, help="about how many items (default 40)")
    mk.add_argument("--seed", type=int, default=2026)
    mk.add_argument("--force", action="store_true", help="overwrite existing sheets")
    mk.set_defaults(fn=cmd_make)
    sc = sub.add_parser("score", help="compare filled-in sheets with the scorer")
    sc.add_argument("--dir", type=Path, required=True)
    sc.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
