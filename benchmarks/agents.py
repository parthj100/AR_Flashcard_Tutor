"""
Shared client helpers for the agent benchmarks.

Every benchmark in this directory hits the same two local services:
  - OCR  sidecar      (default http://127.0.0.1:8766)
  - Ollama LLM server (default http://127.0.0.1:11434)

Each helper returns a small dict; if the service is unreachable the call
returns {"ok": False, "error": "..."} rather than raising. This lets the
pipeline / rewards scripts run partially when a server is down — the
unavailable agent shows up as "skipped" instead of crashing the whole run.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

DEFAULT_OCR    = "http://127.0.0.1:8766"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL = "phi4-mini"

# Mirrors prototype/lib/llm.js : FLASHCARD_SCHEMA
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
            "type": "array", "minItems": 4, "maxItems": 4,
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

LLM_SYSTEM_PROMPT = (
    "You write concise educational flashcards for a science/humanities study "
    "app called Lens. For a given topic, output a single JSON object "
    "following the exact schema provided. Be factually accurate, neutral, "
    "and specific. No marketing fluff. Write in a calm, textbook tone. Do "
    "NOT include any commentary, code fences, or text outside the JSON "
    "object."
)


# ---------- low-level HTTP -------------------------------------------------

def _post_json(url: str, body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _img_to_data_url(img: Path) -> str:
    raw = img.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    suffix = img.suffix.lower().lstrip(".") or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{b64}"


# ---------- agent calls ----------------------------------------------------

def ocr_health(endpoint: str = DEFAULT_OCR) -> dict:
    try:
        return {"ok": True, **_get_json(f"{endpoint.rstrip('/')}/health")}
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        return {"ok": False, "error": str(e)}


def ocr_extract(img: Path, *, conf: float = 0.5, model: str | None = None,
                endpoint: str = DEFAULT_OCR, timeout: float = 120.0) -> dict:
    """model: "easyocr" (server default) or "pix2text"."""
    body = {"image": _img_to_data_url(img), "conf": conf}
    if model:
        body["model"] = model
    t0 = time.perf_counter()
    try:
        payload = _post_json(f"{endpoint.rstrip('/')}/ocr", body, timeout=timeout)
    except Exception as e:
        return {"ok": False, "error": str(e), "wall_ms": (time.perf_counter() - t0) * 1000}
    wall_ms = (time.perf_counter() - t0) * 1000
    lines = payload.get("lines", [])
    confs = [ln["confidence"] for ln in lines]
    return {
        "ok": True,
        "model": payload.get("model"),
        "model_version": payload.get("model_version"),
        "wall_ms": wall_ms,
        "inference_ms": payload.get("inference_ms", 0.0),
        "image_size": payload.get("image_size", {}),
        "n_lines": len(lines),
        "mean_conf": (sum(confs) / len(confs)) if confs else 0.0,
        "raw_text": payload.get("raw_text", ""),
        "lines": lines,
    }


def ollama_health(host: str = DEFAULT_OLLAMA, model: str = DEFAULT_LLM_MODEL) -> dict:
    try:
        data = _get_json(f"{host}/api/tags")
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        return {"ok": False, "error": str(e)}
    names = [m.get("name", "") for m in data.get("models", [])]
    has = any(n == model or n.startswith(model.split(":")[0] + ":") for n in names)
    return {"ok": True, "models": names, "hasConfiguredModel": has}


def schema_valid(obj) -> tuple[bool, list[str]]:
    """Validate against FLASHCARD_SCHEMA (light-weight; no external dep)."""
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


def llm_generate(topic: str, *, host: str = DEFAULT_OLLAMA,
                 model: str = DEFAULT_LLM_MODEL, hint: str = "",
                 timeout: float = 180.0) -> dict:
    user_prompt = (
        f"Topic: {topic}" + (f"\nVisual hint: {hint}" if hint else "") + "\n\n"
        "Write a flashcard for this topic as JSON matching the provided schema."
    )
    body = {
        "model": model,
        "stream": False,
        "format": FLASHCARD_SCHEMA,
        "options": {"temperature": 0.2, "num_predict": 512},
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    }
    t0 = time.perf_counter()
    try:
        payload = _post_json(f"{host}/api/chat", body, timeout=timeout)
    except Exception as e:
        return {"ok": False, "error": str(e), "wall_ms": (time.perf_counter() - t0) * 1000}
    wall_ms = (time.perf_counter() - t0) * 1000

    content = (payload.get("message") or {}).get("content", "").strip()
    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        s, e = content.find("{"), content.rfind("}")
        if s >= 0 and e > s:
            try:
                parsed = json.loads(content[s:e + 1])
            except json.JSONDecodeError:
                pass

    ok, problems = (False, ["non-JSON response"]) if parsed is None else schema_valid(parsed)
    eval_count = payload.get("eval_count", 0)
    eval_ns    = payload.get("eval_duration", 0)
    tps = (eval_count / (eval_ns / 1e9)) if eval_ns > 0 else 0.0

    return {
        "ok": True,
        "topic": topic,
        "wall_ms": wall_ms,
        "ollama_total_ms": payload.get("total_duration", 0) / 1e6,
        "load_ms":         payload.get("load_duration", 0) / 1e6,
        "eval_ms":         eval_ns / 1e6,
        "tokens":          eval_count,
        "tokens_per_sec":  round(tps, 2),
        "schema_valid":    ok,
        "problems":        problems,
        "card":            parsed,
        "raw":             content[:300],
    }
