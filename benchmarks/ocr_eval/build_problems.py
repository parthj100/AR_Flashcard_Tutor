"""
Pick the test problems and write dataset/problems.csv (reproducible: fixed seed).

    .venv/bin/python benchmarks/ocr_eval/build_problems.py --gsm8k 20 --math 10

Downloads the public test sets on first run (into dataset/sources/):
    GSM8K test  (MIT, (c) 2021 OpenAI)        github.com/openai/grade-school-math
    MATH-500    (MIT, (c) 2023 OpenAI; problems from MATH, MIT, (c) 2021 Dan Hendrycks)
                500-problem MATH test split from github.com/openai/prm800k
See dataset/LICENSES.md.

MATH problems are kept only if they use the symbols the study is about
(fractions, powers, square roots, π), have no diagram, and are short enough
to fit on one page. GSM8K problems are kept as-is.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    # Original MIT-licensed source (Git LFS file, so the media URL). The Hugging Face mirror
    # HuggingFaceH4/MATH-500 has identical content but lists no license, so we don't use it.
    "math500_test.jsonl": "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/math_splits/test.jsonl",
}
MATH_SYMBOLS = re.compile(r"\\[dt]?frac|\\sqrt|\^|\\pi")
MATH_EXCLUDE = re.compile(r"\[asy\]|\\begin|\\\\|\\(?:overline|underline|boxed|hline|mbox|qquad)")


def fetch(dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in SOURCES.items():
        p = dest / name
        if not p.exists():
            print("downloading", url)
            urllib.request.urlretrieve(url, p)
        out[name] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--gsm8k", type=int, default=20)
    ap.add_argument("--math", type=int, default=10)
    ap.add_argument("--max-math-len", type=int, default=260)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    src = fetch(a.dataset / "sources")
    rnd = random.Random(a.seed)

    gsm = rnd.sample(range(len(src["gsm8k_test.jsonl"])), a.gsm8k)
    rows = [{"problem_id": f"gsm8k_{i:04d}", "source": "gsm8k", "format": "plain",
             "reference": src["gsm8k_test.jsonl"][i]["question"].strip()} for i in sorted(gsm)]

    pool = [(i, r) for i, r in enumerate(src["math500_test.jsonl"])
            if MATH_SYMBOLS.search(r["problem"]) and not MATH_EXCLUDE.search(r["problem"])
            and len(r["problem"]) <= a.max_math_len]
    rnd.shuffle(pool)
    picked = [{"problem_id": f"math_{i:03d}", "source": "math", "format": "latex",
               "reference": " ".join(r["problem"].split())} for i, r in pool[:a.math]]
    rows += sorted(picked, key=lambda r: r["problem_id"])

    out = a.dataset / "problems.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["problem_id", "source", "format", "reference"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} problems ({a.gsm8k} GSM8K, {len(picked)} MATH) -> {out}")


if __name__ == "__main__":
    main()
