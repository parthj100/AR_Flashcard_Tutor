"""
LLM agent benchmark.

Drives the local Ollama server (http://localhost:11434) with a fixed list of
topics, asks for a flashcard each, and reports:

  - time-to-first-token (TTFT) and total generation latency
  - tokens / second (eval_count / eval_duration_ns from Ollama)
  - schema validity rate (does each response parse under FLASHCARD_SCHEMA?)
  - mean response length

Goal: produce a standardized number for the LLM agent so the benchmark
suite covers the LLM as well as OCR. Outputs match the shape of
the OCR benchmark (timestamped CSV + MD + JSON in
benchmarks/results/).

Run:
  ollama serve                                      # in another terminal
  ollama pull phi4-mini                             # one-time
  python benchmarks/benchmark_llm.py                # default settings

Schema lifted verbatim from prototype/lib/llm.js so we test the same
contract the browser uses.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path
from statistics import mean

DEFAULT_HOST  = "http://localhost:11434"
DEFAULT_MODEL = "phi4-mini"
DEFAULT_TOPICS_FILE = "benchmarks/topics.txt"
DEFAULT_TOPICS = [
    "Copper sulfate", "Mitochondria", "Hagia Sophia", "Maple leaf",
    "Periodic table", "DNA double helix", "Stegosaurus", "Trumpet",
    "Photosynthesis", "Pythagorean theorem", "Eiffel tower",
    "Volcano", "Black hole", "Andromeda galaxy", "Magnetism",
    "Acid base reaction", "Bridge truss", "Solar system",
    "Cellular respiration", "Mendel's laws",
]

SYSTEM_PROMPT = (
    "You write concise educational flashcards for a science/humanities study "
    "app called Lens. For a given topic, output a single JSON object "
    "following the exact schema provided. Be factually accurate, neutral, "
    "and specific. No marketing fluff. Write in a calm, textbook tone. Do "
    "NOT include any commentary, code fences, or text outside the JSON "
    "object."
)

# Mirrors FLASHCARD_SCHEMA in prototype/lib/llm.js
FLASHCARD_SCHEMA = {
    "type": "object",
    "required": ["name", "subject", "formula", "mass", "oneline", "facts"],
    "properties": {
        "name":    {"type": "string"},
        "subject": {"type": "string"},
        "formula": {"type": "string"},
        "mass":    {"type": "string"},
        "oneline": {"type": "string"},
        "facts": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "required": ["num", "label", "body"],
                "properties": {
                    "num":   {"type": "string"},
                    "label": {"type": "string"},
                    "body":  {"type": "string"},
                },
            },
        },
    },
}


def percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def schema_valid(obj) -> tuple[bool, list[str]]:
    """Return (ok, list_of_problems). Mirrors normalizeFlashcard in llm.js."""
    problems = []
    if not isinstance(obj, dict):
        return False, ["response is not a JSON object"]
    for field in ["name", "subject", "formula", "mass", "oneline", "facts"]:
        if field not in obj:
            problems.append(f"missing field: {field}")
    facts = obj.get("facts")
    if not isinstance(facts, list):
        problems.append("facts is not an array")
    elif len(facts) != 4:
        problems.append(f"facts has {len(facts)} entries, want 4")
    else:
        for i, f in enumerate(facts):
            if not isinstance(f, dict):
                problems.append(f"facts[{i}] is not an object")
                continue
            for sub in ("num", "label", "body"):
                if sub not in f:
                    problems.append(f"facts[{i}] missing {sub}")
    return (not problems), problems


def check_ollama(host: str, model: str) -> dict:
    with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as r:
        data = json.loads(r.read().decode("utf-8"))
    names = [m.get("name", "") for m in data.get("models", [])]
    has = any(n == model or n.startswith(model.split(":")[0] + ":") for n in names)
    return {"models": names, "hasConfiguredModel": has}


def generate(host: str, model: str, topic: str, timeout: float = 120.0) -> dict:
    user_prompt = (
        f"Topic: {topic}\n\n"
        "Write a flashcard for this topic as JSON matching the provided schema.\n"
        "- 'name' must be a clean human-readable title.\n"
        "- 'subject' is 2-3 short ALL-CAPS tokens separated by ' · '.\n"
        "- 'formula' is the most canonical identifier.\n"
        "- 'mass' is a short quantitative note.\n"
        "- 'oneline' is exactly 1-2 sentences, ~25-40 words.\n"
        "- 'facts' is exactly 4 items with num '01'-'04', short ALL-CAPS label, one-sentence body."
    )
    body = {
        "model":  model,
        "stream": False,
        "format": FLASHCARD_SCHEMA,
        "options": {"temperature": 0.2, "num_predict": 512},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    wall_ms = (time.perf_counter() - t0) * 1000

    content = (payload.get("message") or {}).get("content", "").strip()
    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # salvage: small models sometimes wrap JSON in prose
        m_start = content.find("{")
        m_end = content.rfind("}")
        if m_start >= 0 and m_end > m_start:
            try:
                parsed = json.loads(content[m_start:m_end + 1])
            except json.JSONDecodeError:
                parsed = None

    ok, problems = (False, ["non-JSON response"]) if parsed is None else schema_valid(parsed)

    # Ollama returns timing in nanoseconds
    total_ns      = payload.get("total_duration", 0)
    load_ns       = payload.get("load_duration", 0)
    prompt_eval_ns = payload.get("prompt_eval_duration", 0)
    eval_ns       = payload.get("eval_duration", 0)
    eval_count    = payload.get("eval_count", 0)
    tps = (eval_count / (eval_ns / 1e9)) if eval_ns > 0 else 0.0

    return {
        "topic":          topic,
        "wall_ms":        wall_ms,
        "ollama_total_ms": total_ns / 1e6,
        "load_ms":         load_ns / 1e6,
        "prompt_eval_ms":  prompt_eval_ns / 1e6,
        "eval_ms":         eval_ns / 1e6,
        "tokens":          eval_count,
        "tokens_per_sec":  round(tps, 2),
        "chars":           len(content),
        "schema_valid":    ok,
        "problems":        problems,
        "raw":             content[:300],
    }


def run(args):
    print("=" * 62)
    print("  LLM agent benchmark — Ollama / Phi-4-mini")
    print("=" * 62)

    try:
        info = check_ollama(args.host, args.model)
    except (urllib.error.URLError, ConnectionError) as e:
        print(f"\n  ERROR: Ollama unreachable at {args.host}: {e}")
        print(f"  Start it with: OLLAMA_ORIGINS='*' ollama serve")
        return 1

    print(f"\n  Host    : {args.host}")
    print(f"  Model   : {args.model}")
    print(f"  Models  : {info['models']}")
    if not info["hasConfiguredModel"]:
        print(f"  WARN  : configured model '{args.model}' not in tags list")

    topics_file = Path(args.topics)
    if topics_file.is_file():
        topics = [ln.strip() for ln in topics_file.read_text().splitlines() if ln.strip()]
    else:
        topics = DEFAULT_TOPICS
    if args.limit:
        topics = topics[: args.limit]
    print(f"  Topics  : {len(topics)} (source: {topics_file if topics_file.is_file() else 'built-in'})")

    # Warmup (one full request, since cold-start dwarfs everything)
    print("\n  Warming up (one request)...", end=" ", flush=True)
    try:
        generate(args.host, args.model, topics[0])
    except Exception as e:
        print(f"FAIL: {e}")
        return 1
    print("done.")

    print("\n  Generating...")
    rows = []
    valid = 0
    t_wall = time.perf_counter()
    for i, topic in enumerate(topics, 1):
        try:
            r = generate(args.host, args.model, topic)
        except Exception as e:
            print(f"    [{i}/{len(topics)}] FAIL  topic={topic} :: {e}")
            continue
        rows.append(r)
        valid += 1 if r["schema_valid"] else 0
        status = "ok " if r["schema_valid"] else "BAD"
        print(f"    [{i}/{len(topics)}] {status}  topic={topic:<24}  {r['wall_ms']:.0f} ms  "
              f"{r['tokens']} tok  {r['tokens_per_sec']:.1f} tok/s")
    total_wall_s = time.perf_counter() - t_wall

    if not rows:
        print("\n  ERROR: no successful generations")
        return 1

    lats   = [r["wall_ms"] for r in rows]
    tps    = [r["tokens_per_sec"] for r in rows if r["tokens_per_sec"] > 0]
    tokens = [r["tokens"] for r in rows]
    schema_rate = valid / len(rows)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = out_dir / f"llm_benchmark_{ts}"

    # CSV (one row per topic)
    keys = ["topic", "wall_ms", "ollama_total_ms", "load_ms", "prompt_eval_ms",
            "eval_ms", "tokens", "tokens_per_sec", "chars", "schema_valid"]
    with stem.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    summary = {
        "model":         args.model,
        "topics":        len(rows),
        "wall_s":        round(total_wall_s, 2),
        "mean_ms":       round(mean(lats), 2),
        "p50_ms":        round(percentile(lats, 50), 2),
        "p95_ms":        round(percentile(lats, 95), 2),
        "mean_tokens":   round(mean(tokens), 1),
        "mean_tokens_per_sec": round(mean(tps), 2) if tps else 0.0,
        "schema_valid_rate":   round(schema_rate, 4),
    }

    with stem.with_suffix(".json").open("w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2, default=str)

    with stem.with_suffix(".md").open("w") as f:
        f.write(f"# LLM agent benchmark\n\n")
        f.write(f"- host    : `{args.host}`\n")
        f.write(f"- model   : `{args.model}`\n")
        f.write(f"- topics  : `{len(rows)}` (source: `{topics_file if topics_file.is_file() else 'built-in'}`)\n")
        f.write(f"- wall_s  : `{total_wall_s:.2f}`\n")
        f.write(f"- timestamp: `{ts}`\n\n")
        f.write("## Summary\n\n")
        f.write("| metric | value |\n|---|---:|\n")
        for k, v in summary.items():
            if k == "schema_valid_rate":
                f.write(f"| schema-valid rate | {v:.2%} |\n")
            else:
                f.write(f"| {k} | {v} |\n")
        f.write("\n## Per-topic\n\n")
        f.write("| topic | wall ms | tokens | tok/s | schema |\n|---|---:|---:|---:|---:|\n")
        for r in rows:
            check = "✓" if r["schema_valid"] else "✗"
            f.write(f"| `{r['topic']}` | {r['wall_ms']:.0f} | {r['tokens']} "
                    f"| {r['tokens_per_sec']:.1f} | {check} |\n")

    print("\n" + "=" * 62)
    print(f"  Done. {len(rows)} topics in {total_wall_s:.2f} s")
    print(f"  Mean wall latency : {summary['mean_ms']} ms (p95 {summary['p95_ms']} ms)")
    print(f"  Mean throughput   : {summary['mean_tokens_per_sec']} tok/s")
    print(f"  Schema valid rate : {schema_rate:.2%}")
    print(f"  Results           : {stem.with_suffix('.md')}")
    print("=" * 62)
    return 0


def main():
    p = argparse.ArgumentParser(description="LLM agent benchmark")
    p.add_argument("--host",   default=DEFAULT_HOST)
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--topics", default=DEFAULT_TOPICS_FILE)
    p.add_argument("--limit",  type=int, default=0, help="if >0, cap topics processed")
    p.add_argument("--results-dir", default="benchmarks/results")
    args = p.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
