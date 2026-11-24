"""
Score OCR transcripts against reference transcripts.

Inputs
------
--refs   CSV or JSONL, one row per problem:
           problem_id, reference
--preds  CSV or JSONL, one row per OCR run on one image:
           image_id, problem_id, model, condition, hypothesis
         condition is printed | handwritten | screen | synthetic.
         Optional columns used when present:
           split      dev | test   (test rows are skipped unless --include-test)
           energy_j   joules for this image (idle power already subtracted)
           latency_ms
         Any other columns (device, lighting, angle, writer, ...) are copied
         to the per-image output so you can slice by them later.

Outputs (in --out-dir, default benchmarks/results/)
-------
  scores_<timestamp>.csv          one row per image: CER, WER, critical errors
  scores_<timestamp>_summary.csv  one row per model x condition
  scores_<timestamp>_summary.md   the same, readable

Synthetic images are never pooled with real photos. Each summary has a
"real (all)" row per model that pools printed, handwritten and screen only.

Run:
  python benchmarks/score_ocr.py --refs data/references.csv --preds data/ocr_easyocr.csv
  python benchmarks/score_ocr.py --refs data/references.csv --preds data/ocr_easyocr.csv data/ocr_pix2text.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import CATEGORIES, score  # noqa: E402

REAL_CONDITIONS = ("printed", "handwritten", "screen")
CONDITIONS = REAL_CONDITIONS + ("synthetic",)
REAL_ALL = "real (all)"
REQUIRED_PRED_COLS = ("image_id", "problem_id", "model", "condition", "hypothesis")


# --- Loading --------------------------------------------------------------------

def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_refs(path: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    for row in load_rows(path):
        pid = str(row["problem_id"]).strip()
        if pid in refs:
            sys.exit(f"error: problem_id {pid!r} appears twice in {path}")
        refs[pid] = row["reference"]
    return refs


def load_preds(path: Path, refs: dict[str, str], include_test: bool) -> list[dict]:
    rows = load_rows(path)
    if rows:
        missing = [c for c in REQUIRED_PRED_COLS if c not in rows[0]]
        if missing:
            sys.exit(f"error: {path} is missing column(s): {', '.join(missing)}")

    out, skipped_test, unknown_ids, bad_conditions = [], 0, set(), set()
    for row in rows:
        row = {k: ("" if v is None else v) for k, v in row.items()}
        row["problem_id"] = str(row["problem_id"]).strip()
        row["condition"] = str(row["condition"]).strip().lower()
        if str(row.get("split", "")).strip().lower() == "test" and not include_test:
            skipped_test += 1
            continue
        if row["problem_id"] not in refs:
            unknown_ids.add(row["problem_id"])
            continue
        if row["condition"] not in CONDITIONS:
            bad_conditions.add(row["condition"])
            continue
        out.append(row)

    if unknown_ids:
        sys.exit(f"error: no reference for problem_id(s): {', '.join(sorted(unknown_ids))}")
    if bad_conditions:
        sys.exit(f"error: unknown condition(s) {sorted(bad_conditions)}; "
                 f"expected one of {', '.join(CONDITIONS)}")
    if skipped_test:
        print(f"Skipped {skipped_test} held-out test row(s). "
              f"Pass --include-test only for the final, locked run.")
    return out


# --- Scoring ----------------------------------------------------------------------

def score_rows(preds: list[dict], refs: dict[str, str]) -> list[dict]:
    scored = []
    for row in preds:
        s = score(refs[row["problem_id"]], row["hypothesis"])
        extra = {k: v for k, v in row.items() if k != "hypothesis"}
        scored.append({
            **extra,
            "cer": round(s.cer, 4),
            "wer": round(s.wer, 4),
            "cer_edits": s.cer_edits,
            "ref_chars": s.ref_chars,
            "wer_edits": s.wer_edits,
            "ref_words": s.ref_words,
            "critical_errors": s.critical_count,
            "critical_free": int(s.critical_free),
            **{f"crit_{c}": n for c, n in s.category_counts().items()},
            "critical_detail": "; ".join(e.describe() for e in s.errors),
            "hypothesis": row["hypothesis"],
        })
    return scored


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ref_chars = sum(r["ref_chars"] for r in rows)
    ref_words = sum(r["ref_words"] for r in rows)
    critical_free = sum(r["critical_free"] for r in rows)
    out = {
        "n": n,
        "cer_corpus": round(sum(r["cer_edits"] for r in rows) / ref_chars, 4) if ref_chars else 0.0,
        "cer_mean": round(mean(r["cer"] for r in rows), 4),
        "cer_median": round(median(r["cer"] for r in rows), 4),
        "wer_corpus": round(sum(r["wer_edits"] for r in rows) / ref_words, 4) if ref_words else 0.0,
        "wer_mean": round(mean(r["wer"] for r in rows), 4),
        "critical_errors": sum(r["critical_errors"] for r in rows),
        "critical_free_rate": round(critical_free / n, 4),
        **{f"crit_{c}": sum(r[f"crit_{c}"] for r in rows) for c in CATEGORIES},
    }

    latencies = [x for x in (_float(r.get("latency_ms")) for r in rows) if x is not None]
    if latencies:
        out["latency_ms_mean"] = round(mean(latencies), 1)

    energies = [_float(r.get("energy_j")) for r in rows]
    if energies and all(e is not None for e in energies):
        total = sum(energies)
        out["energy_j_per_image"] = round(total / n, 3)
        # Joules per transcript with no critical errors (Update #1, section 4).
        out["energy_j_per_critical_free"] = round(total / critical_free, 3) if critical_free else ""
    return out


def group_summaries(scored: list[dict]) -> list[dict]:
    has_split = any(r.get("split") for r in scored)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in scored:
        split = r.get("split", "") if has_split else ""
        groups[(split, r["model"], r["condition"])].append(r)
        if r["condition"] in REAL_CONDITIONS:
            groups[(split, r["model"], REAL_ALL)].append(r)

    order = {c: i for i, c in enumerate(CONDITIONS + (REAL_ALL,))}
    summaries = []
    for (split, model, condition) in sorted(groups, key=lambda k: (k[0], k[1], order[k[2]])):
        head = {"split": split} if has_split else {}
        summaries.append({**head, "model": model, "condition": condition,
                          **summarize(groups[(split, model, condition)])})
    return summaries


# --- Output -------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_md(path: Path, summaries: list[dict], scored: list[dict], args) -> None:
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    has_split = summaries and "split" in summaries[0]
    has_energy = any("energy_j_per_image" in s for s in summaries)

    cols = (["Split"] if has_split else []) + [
        "Model", "Condition", "n", "CER", "WER", "Critical errors", "Critical-free",
        "Number", "Decimal", "Sign/op", "Frac/exp", "Unit/$"]
    if has_energy:
        cols += ["J/image", "J/critical-free"]

    lines = [
        "# OCR transcript scores", "",
        f"- References: `{args.refs}`",
        f"- Predictions: {', '.join(f'`{p}`' for p in args.preds)}",
        f"- Images scored: {len(scored)}",
        f"- Held-out test rows included: {'yes' if args.include_test else 'no'}",
        "",
        "CER and WER are corpus-level (total edits / total reference length). "
        "Critical-free is the share of transcripts with no critical errors. "
        "Synthetic images are reported on their own and never pooled into \"real (all)\".",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for s in summaries:
        cells = ([s["split"]] if has_split else []) + [
            s["model"], s["condition"], str(s["n"]),
            pct(s["cer_corpus"]), pct(s["wer_corpus"]), str(s["critical_errors"]),
            pct(s["critical_free_rate"]),
            *(str(s[f"crit_{c}"]) for c in CATEGORIES)]
        if has_energy:
            cells += [str(s.get("energy_j_per_image", "")), str(s.get("energy_j_per_critical_free", ""))]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--refs", type=Path, required=True, help="reference transcripts (CSV or JSONL)")
    ap.add_argument("--preds", type=Path, nargs="+", required=True,
                    help="OCR outputs (CSV or JSONL); pass one file per model to compare them")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--include-test", action="store_true",
                    help="also score rows with split=test (final locked run only)")
    args = ap.parse_args(argv)

    refs = load_refs(args.refs)
    preds = [row for p in args.preds for row in load_preds(p, refs, args.include_test)]
    if not preds:
        sys.exit("error: nothing to score")
    failed = sum(1 for r in preds if str(r.get("error", "")).strip())
    if failed:
        print(f"Note: {failed} row(s) have an OCR error; they are scored as the (empty) "
              f"text the service returned.")

    scored = score_rows(preds, refs)
    summaries = group_summaries(scored)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"scores_{time.strftime('%Y%m%d-%H%M%S')}"
    write_csv(stem.with_suffix(".csv"), scored)
    write_csv(Path(f"{stem}_summary.csv"), summaries)
    write_md(Path(f"{stem}_summary.md"), summaries, scored, args)

    print(f"Scored {len(scored)} image(s).")
    for s in summaries:
        label = " / ".join(x for x in (s.get("split", ""), s["model"], s["condition"]) if x)
        print(f"  {label:40s} n={s['n']:<4} CER={s['cer_corpus']:.3f} "
              f"WER={s['wer_corpus']:.3f} critical-free={s['critical_free_rate']:.0%}")
    print(f"Wrote {stem}.csv, {stem.name}_summary.csv and {stem.name}_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
