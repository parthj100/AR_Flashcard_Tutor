"""
Step 2: score every reading against its reference.

    .venv/bin/python benchmarks/ocr_eval/score.py

Writes to benchmarks/results/:
    scored.csv       one row per (photo, model): CER, WER, critical errors by category
    summary.json     everything the graphs need (read by plots.py)
    hand_check.csv   a random sample to check by hand; fill in manual_critical,
                     then run again to print how often the script agrees with you

Real photos (printed, screen, handwritten) are the headline; rendered images
(typeset by render_pages.py, not AI-generated) are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from dataset import PHOTO_TYPES, REAL_TYPES, load_problems
from metrics import CATEGORIES, score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOW_CER = 0.05          # H4: "readings with less than 5% of characters wrong"
TYPE_LABEL = {"printed": "Printed", "screen": "Screen", "handwritten": "Handwritten",
              "rendered": "Rendered (sharp)", "rendered_blur": "Rendered (blurred)"}


def load_readings(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(rows, models):
    real = [r for r in rows if r["photo_type"] in REAL_TYPES]
    # Pooled stats (H4, error types, scatter) use every photo; "pool_label" names them in subtitles.
    pool, pool_label = (real, "real photos") if real else (rows, "rendered images")

    def rate(sel):
        return {m: [sum(r["error_free"] for r in sel if r["model"] == m),
                    sum(1 for r in sel if r["model"] == m)] for m in models}

    printed_screen = [r for r in real if r["photo_type"] in ("printed", "screen")]
    kind_sel, kind_label = ((printed_screen, "printed + screen") if printed_screen
                            else (rows, "all images"))
    groups = [("All real photos", real)]
    groups += [(TYPE_LABEL[t], [r for r in rows if r["photo_type"] == t]) for t in PHOTO_TYPES]
    groups += [(f"Math problems\n({kind_label})", [r for r in kind_sel if r["source"] == "math"]),
               (f"Word problems\n({kind_label})", [r for r in kind_sel if r["source"] == "gsm8k"])]
    error_free = {label: rate(sel) for label, sel in groups if sel}

    def mean_by_type(key):
        out = {}
        for t in PHOTO_TYPES:
            sel = [r for r in rows if r["photo_type"] == t]
            if sel:
                out[TYPE_LABEL[t]] = {m: statistics.fmean(r[key] for r in sel if r["model"] == m)
                                      for m in models if any(r["model"] == m for r in sel)}
        return out

    low = [r for r in pool if r["cer"] < LOW_CER]
    cats = {m: {c: sum(r[f"crit_{c}"] for r in pool if r["model"] == m) for c in CATEGORIES}
            for m in models}
    return {
        "source": "benchmarks/ocr_eval/score.py",
        "models": models,
        "n_real_photos": len({r["photo_id"] for r in real}),
        "n_problems": len({r["problem_id"] for r in rows}),
        "pool_label": pool_label,
        "error_free_subtitle": (f"Higher is better. {len({r['problem_id'] for r in rows})} problems, "
                                f"{len({r['photo_id'] for r in rows})} images; a critical error changes "
                                "a number or math symbol."),
        "error_free": error_free,
        "cer": mean_by_type("cer"),
        "wer": mean_by_type("wer"),
        "time_s": mean_by_type("seconds"),
        "hidden": {"cer_threshold": LOW_CER, "low_cer": len(low),
                   "with_critical": sum(1 for r in low if r["critical"] > 0)},
        "categories": cats,
        "points": [{"model": r["model"], "cer": r["cer"], "critical": r["critical"],
                    "photo_type": r["photo_type"]} for r in pool],
    }


def hand_check(rows, path: Path, n: int, seed: int):
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            done = [r for r in csv.DictReader(f) if r.get("manual_critical", "").strip()]
        if done:
            exact = sum(int(r["manual_critical"]) == int(r["auto_critical"]) for r in done)
            ef = sum((int(r["manual_critical"]) == 0) == (int(r["auto_critical"]) == 0) for r in done)
            print(f"\nHand check: {len(done)} rows filled in. Script agrees with you on the exact count "
                  f"{exact}/{len(done)}, on error-free yes/no {ef}/{len(done)}.")
            return
        print(f"\n{path.name} exists but has no manual_critical filled in; leaving it as is.")
        return
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo_id", "model", "reference", "ocr_text", "auto_missing", "auto_extra",
                    "auto_critical", "manual_critical", "notes"])
        for r in sample:
            w.writerow([r["photo_id"], r["model"], r["ref_norm"], r["hyp_norm"],
                        " ".join(r["missing"]), " ".join(r["extra"]), r["critical"], "", ""])
    print(f"\nWrote {len(sample)} rows to check by hand: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--results", type=Path, default=ROOT / "benchmarks" / "results")
    ap.add_argument("--sample", type=int, default=30, help="rows in hand_check.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    problems = load_problems(args.dataset)
    readings = load_readings(args.results / "readings.jsonl")
    models = list(dict.fromkeys(r["model"] for r in readings))

    rows = []
    for rd in readings:
        p = problems[rd["problem_id"]]
        s = score(p.reference, rd["plain_text"], p.format)
        rows.append({
            "photo_id": rd["photo_id"], "problem_id": rd["problem_id"], "photo_type": rd["photo_type"],
            "source": p.source, "model": rd["model"], "seconds": rd["seconds"],
            "cer": s["cer"], "wer": s["wer"], "critical": s["critical"], "error_free": s["error_free"],
            **{f"crit_{c}": s["by_category"][c] for c in CATEGORIES},
            "missing": s["missing"], "extra": s["extra"], "ref_norm": s["ref_norm"], "hyp_norm": s["hyp_norm"],
        })

    with (args.results / "scored.csv").open("w", newline="", encoding="utf-8") as f:
        cols = [k for k in rows[0] if k not in ("ref_norm", "hyp_norm")]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "cer": f"{r['cer']:.4f}", "wer": f"{r['wer']:.4f}",
                        "missing": " ".join(r["missing"]), "extra": " ".join(r["extra"])})

    summary = summarize(rows, models)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Terminal table
    print(f"{'group':<36}" + "".join(f"{m:>18}" for m in models))
    for label, by_m in summary["error_free"].items():
        cells = "".join(f"{k}/{n} ({k / n:4.0%})".rjust(18) if n else " " * 18 for k, n in by_m.values())
        print(f"{'error-free  ' + label.replace(chr(10), ' '):<36}{cells}")
    for key, name, fmt in (("cer", "CER", "{:.1%}"), ("time_s", "time", "{:.2f} s")):
        for t, by_m in summary[key].items():
            print(f"{name + '  ' + t:<36}" + "".join(fmt.format(by_m[m]).rjust(18) if m in by_m else " " * 18
                                                    for m in models))
    h = summary["hidden"]
    print(f"\nH4: {h['with_critical']} of {h['low_cer']} readings with CER < {h['cer_threshold']:.0%} "
          f"still had a critical error.")

    hand_check([r for r in rows if r["photo_type"] in REAL_TYPES] or rows,
               args.results / "hand_check.csv", args.sample, args.seed)


if __name__ == "__main__":
    main()
