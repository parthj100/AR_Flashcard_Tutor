"""
Run one OCR service over every photo in a manifest and write the file that
benchmarks/score_ocr.py reads.

Manifest (CSV): one row per photo
  image_id, problem_id, condition, path
  + optional split, device, lighting, angle, writer, ... (copied through)
  path is relative to the manifest's folder.

Images are sent one at a time, the way the app sends them. Before timing,
one warm-up request loads the model; its time is reported separately and not
counted in any row.

Output (CSV): the manifest columns plus
  model, model_version, hypothesis (plain text, scored),
  raw_output (model's own format, e.g. LaTeX), inference_ms, latency_ms,
  n_lines, mean_conf, error

Run with the OCR service up (OCR/ocr_serve.py serves both tools):
  python benchmarks/run_ocr.py --manifest data/manifest.csv --model easyocr --out data/ocr_easyocr.csv
  python benchmarks/run_ocr.py --manifest data/manifest.csv --model pix2text --out data/ocr_pix2text.csv
  python benchmarks/score_ocr.py --refs data/references.csv --preds data/ocr_easyocr.csv data/ocr_pix2text.csv

--conf defaults to 0 so the model's low-confidence words are kept. Dropping
them would turn uncertain reads into missing text and skew the comparison.
Like score_ocr.py, rows with split=test are skipped unless --include-test.

Energy (Apple Silicon Macs): add --energy to also record joules per image.
Type `sudo -v` in the same terminal first; close other apps. This adds
  energy_j        joules above idle (what score_ocr.py reports)
  energy_total_j  joules including idle
  power_w         average chip power during the request
See benchmarks/power.py for how the number is made and what it leaves out.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents import ocr_extract, ocr_health  # noqa: E402
from power import PowerMonitor  # noqa: E402

REQUIRED = ("image_id", "problem_id", "condition", "path")


def pick_model(health: dict, wanted: str | None) -> tuple[str | None, str, str]:
    """(model to request, name to record, version) from a service's /health reply.

    ocr_serve.py runs several models and lists them under "models".
    ocr_serve_pix2text.py runs one and names it under "model"; it takes no
    model in the request, so the first value is None.
    """
    models = health.get("models")
    if models:
        name = wanted or health.get("default_model")
        if name not in models:
            raise ValueError(f"the service has no model '{name}'; it has: {', '.join(models)}")
        info = models[name]
        return name, info.get("label") or name, info.get("version", "")
    name = health.get("model") or health.get("ocr", "unknown")
    if wanted and wanted != name:
        raise ValueError(f"the service runs '{name}', not '{wanted}'")
    return None, name, health.get("version", "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run an OCR service over a photo manifest.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8766")
    ap.add_argument("--model", help="which OCR tool to use, e.g. easyocr or pix2text "
                                    "(default: the service's own default)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.0,
                    help="confidence threshold sent to the service (default 0: keep everything)")
    ap.add_argument("--include-test", action="store_true",
                    help="also run rows with split=test (final locked run only)")
    ap.add_argument("--condition", nargs="+", choices=("printed", "handwritten", "screen", "synthetic"),
                    help="only run these conditions (default: every row that has a photo)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--energy", action="store_true",
                    help="also measure energy per image with powermetrics (Apple Silicon; "
                         "run `sudo -v` first). Adds energy_j: joules above idle")
    ap.add_argument("--idle-seconds", type=float, default=10.0,
                    help="how long to measure idle power before the run (default 10)")
    ap.add_argument("--power-interval", type=int, default=100, help="ms between power readings")
    ap.add_argument("--power-cmd", help=argparse.SUPPRESS)  # for tests
    args = ap.parse_args(argv)

    health = ocr_health(args.endpoint)
    if not health.get("ok"):
        sys.exit(f"error: OCR service at {args.endpoint} is not reachable: {health.get('error')}")
    try:
        request_model, model, version = pick_model(health, args.model)
    except ValueError as e:
        sys.exit(f"error: {e}")
    print(f"Model: {model} {version} at {args.endpoint}")

    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("error: manifest is empty")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        sys.exit(f"error: manifest is missing column(s): {', '.join(missing)}")

    held_out = [r for r in rows if r.get("split", "").strip().lower() == "test"]
    if held_out and not args.include_test:
        rows = [r for r in rows if r not in held_out]
        print(f"Skipping {len(held_out)} held-out test photo(s). "
              f"Pass --include-test only for the final, locked run.")

    if args.condition:
        rows = [r for r in rows if r["condition"].strip().lower() in args.condition]
    no_photo = [r for r in rows if not r["path"].strip()]
    if no_photo:
        rows = [r for r in rows if r["path"].strip()]
        print(f"Skipping {len(no_photo)} row(s) with no photo yet.")
    if not rows:
        sys.exit("error: no photos to run")

    base = args.manifest.parent
    paths = []
    for r in rows:
        p = (base / r["path"]).resolve()
        if not p.is_file():
            sys.exit(f"error: image not found for {r['image_id']}: {p}")
        paths.append(p)

    warm = ocr_extract(paths[0], conf=args.conf, model=request_model, endpoint=args.endpoint, timeout=args.timeout)
    print(f"Warm-up (not counted): {warm.get('wall_ms', 0):.0f} ms")

    monitor = idle_w = None
    if args.energy:
        monitor = PowerMonitor(args.power_interval, shlex.split(args.power_cmd) if args.power_cmd else None)
        try:
            monitor.start()
            print(f"Measuring idle power for {args.idle_seconds:.0f} s: leave the Mac alone...")
            idle_w = monitor.idle_watts(args.idle_seconds)
        except RuntimeError as e:
            monitor.stop()
            sys.exit(f"error: {e}")
        print(f"Idle power: {idle_w:.2f} W")

    out_rows = []
    for i, (r, p) in enumerate(zip(rows, paths), 1):
        t0 = time.perf_counter()
        res = ocr_extract(p, conf=args.conf, model=request_model, endpoint=args.endpoint, timeout=args.timeout)
        t1 = time.perf_counter()
        energy = {}
        if monitor:
            joules, covered = monitor.window(t0, t1)
            if covered >= 0.5 * (t1 - t0):  # enough readings to trust
                joules *= (t1 - t0) / covered  # fill any short gap at the average power
                energy = {"energy_j": round(max(joules - idle_w * (t1 - t0), 0.0), 3),
                          "energy_total_j": round(joules, 3),
                          "power_w": round(joules / (t1 - t0), 2)}
            else:
                energy = {"energy_j": "", "energy_total_j": "", "power_w": ""}
        out_rows.append({
            **r,
            "model": model,
            "model_version": version,
            "hypothesis": res.get("plain_text", "") or res.get("raw_text", ""),
            "raw_output": res.get("raw_output", "") or res.get("raw_text", ""),
            "inference_ms": round(res.get("inference_ms", 0.0), 1),
            "latency_ms": round(res.get("wall_ms", 0.0), 1),
            "n_lines": res.get("n_lines", 0),
            "mean_conf": round(res.get("mean_conf", 0.0), 4),
            "error": "" if res.get("ok") else res.get("error", "failed"),
            **energy,
        })
        status = "ok" if res.get("ok") else f"ERROR {res.get('error')}"
        joule_txt = f"  {energy['energy_j']} J" if energy.get("energy_j", "") != "" else ""
        print(f"  [{i}/{len(rows)}] {r['image_id']:20s} {res.get('wall_ms', 0):7.0f} ms{joule_txt}  {status}")
    if monitor:
        monitor.stop()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in out_rows:
        fields += [k for k in r if k not in fields]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    meta = {"model": model, "version": version, "endpoint": args.endpoint, "conf": args.conf,
            "images": len(out_rows), "errors": sum(1 for r in out_rows if r["error"]),
            "warmup_ms": round(warm.get("wall_ms", 0.0), 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if monitor:
        meta.update({"idle_w": round(idle_w, 3), "power_interval_ms": args.power_interval,
                     "energy_scope": "chip only (CPU + GPU + neural engine), joules above idle"})
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(out_rows)} row(s) to {args.out} ({meta['errors']} error(s))")
    return 1 if meta["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
