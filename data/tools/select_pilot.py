"""
Select the pilot problems: 20 from GSM8K and 10 notation-heavy ones from MATH.

Selection is deterministic (fixed seed), so running this again on the same
source files gives the same problems. Each problem keeps its source row so
it can be traced back to the original dataset.

GSM8K (MIT): test split, 25-60 words, spread over the quantity types our
rubric cares about:
  money ($) 5 · decimals 4 · percent 4 · fractions 3 · plain numbers 4
MATH (MIT): test split, Levels 1-3, Prealgebra / Algebra / Intermediate
Algebra / Number Theory, short (<= 200 characters), no diagrams, spread over
the special characters our professor flagged:
  fractions 3 · exponents 2 · square roots 2 · pi 1 · inequalities 2

Outputs (in --out, default data/pilot/):
  problems.jsonl   one problem per line: id, source, row, bucket, license,
                   text (as shown on the sheet; LaTeX kept for MATH),
                   reference (plain text in benchmarks/REFERENCE_STYLE.md style)
  references.csv   problem_id, reference  (what score_ocr.py reads)

Source files (download once; both MIT):
  GSM8K: https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
  MATH:  https://huggingface.co/api/datasets/EleutherAI/hendrycks_math/parquet/<subject>/test/0.parquet

Run:
  .venv/bin/python data/tools/select_pilot.py --gsm8k test.jsonl --math-dir <folder with math_<subject>.parquet>
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OCR"))
from latex_plain import latex_to_plain  # noqa: E402

SEED = 2026

GSM_BUCKETS = [  # (name, count, test)
    ("money", 5, lambda q: "$" in q),
    ("decimal", 4, lambda q: re.search(r"\d\.\d", q) is not None),
    ("percent", 4, lambda q: "%" in q),
    ("fraction", 3, lambda q: re.search(r"\d/\d", q) is not None),
    ("plain", 4, lambda q: not re.search(r"[$%]|\d\.\d|\d/\d", q)),
]

MATH_SUBJECTS = {"prealgebra": "pre", "algebra": "alg", "intermediate_algebra": "ialg",
                 "number_theory": "nt"}
MATH_BUCKETS = [
    ("fraction", 3, lambda t: r"\frac" in t),
    ("exponent", 2, lambda t: re.search(r"\^\{?[\d\w]", t) is not None and r"\frac" not in t),
    ("sqrt", 2, lambda t: r"\sqrt" in t),
    ("pi", 1, lambda t: r"\pi" in t),
    # \le, \geq ... as whole commands (not the "le" in \left), or a literal < or >
    ("inequality", 2, lambda t: re.search(r"\\(?:le|leq|ge|geq|lt|gt|neq)(?![a-z])|[<>]", t) is not None),
]

# Math delimiters used in MATH problems: $$..$$, \[..\], \(..\), $..$ (but not \$).
_MATH_SPLIT = re.compile(r"(\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)|(?<!\\)\$.+?(?<!\\)\$)", re.S)


def math_to_reference(text: str) -> str | None:
    """MATH problem text -> plain reference, or None if it can't be cleanly converted."""
    parts = []
    for i, chunk in enumerate(_MATH_SPLIT.split(text)):
        if i % 2 == 1:  # a formula
            inner = re.sub(r"^(\$\$|\\\[|\\\(|\$)|(\$\$|\\\]|\\\)|\$)$", "", chunk)
            plain = latex_to_plain(inner)
            if "\\" in plain or re.search(r"frac|sqrt|cdot|left|right|begin", plain):
                return None
            display = chunk.startswith(("$$", r"\["))
            parts.append(f" {plain} " if display else plain)  # display math stands apart: "= 0 for x"
        else:
            prose = chunk.replace(r"\$", "$").replace(r"\%", "%").replace("~", " ")
            if "\\" in prose or "{" in prose:
                return None  # \emph, \textbf, ... outside math: skip the problem
            parts.append(prose)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def pick(candidates: list[dict], buckets, key: str, rng: random.Random) -> list[dict]:
    rng.shuffle(candidates)
    chosen, used = [], set()
    for name, count, test in buckets:
        n = 0
        for c in candidates:
            if n == count:
                break
            if c["id"] in used or not test(c[key]):
                continue
            chosen.append({**c, "bucket": name})
            used.add(c["id"])
            n += 1
        if n < count:
            sys.exit(f"error: only {n} candidates for bucket {name!r}")
    return chosen


def load_gsm8k(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for row, line in enumerate(f):
            q = re.sub(r"\s+", " ", json.loads(line)["question"]).strip()
            if 25 <= len(q.split()) <= 60:
                out.append({"id": f"gsm-{row:04d}", "source": "GSM8K", "split": "test",
                            "row": row, "license": "MIT", "text": q, "reference": q})
    return out


def load_math(folder: Path) -> list[dict]:
    import polars as pl  # only needed for MATH's parquet files

    out = []
    for subject, short in MATH_SUBJECTS.items():
        df = pl.read_parquet(folder / f"math_{subject}.parquet")
        for row, rec in enumerate(df.iter_rows(named=True)):
            t = rec["problem"].strip()
            if rec["level"] not in ("Level 1", "Level 2", "Level 3"):
                continue
            if len(t) > 200 or "[asy]" in t or "\\begin" in t or "\n\n" in t:
                continue
            ref = math_to_reference(t)
            if not ref:
                continue
            out.append({"id": f"math-{short}-{row:04d}", "source": "MATH", "split": "test",
                        "subject": subject, "level": rec["level"], "row": row,
                        "license": "MIT", "text": t, "reference": ref})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Select the pilot problems.")
    ap.add_argument("--gsm8k", type=Path, required=True, help="GSM8K test.jsonl")
    ap.add_argument("--math-dir", type=Path, required=True, help="folder with math_<subject>.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "pilot")
    args = ap.parse_args(argv)

    rng = random.Random(SEED)
    gsm = pick(load_gsm8k(args.gsm8k), GSM_BUCKETS, "text", rng)
    math = pick(load_math(args.math_dir), MATH_BUCKETS, "text", rng)
    problems = gsm + math

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "problems.jsonl").open("w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps({**p, "pilot_split": "dev"}, ensure_ascii=False) + "\n")
    with (args.out / "references.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["problem_id", "reference"])
        for p in problems:
            w.writerow([p["id"], p["reference"]])

    print(f"Selected {len(gsm)} GSM8K + {len(math)} MATH problems -> {args.out}")
    for p in problems:
        print(f"  {p['id']:16s} {p['bucket']:10s} {p['reference'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
