"""
Convert the LaTeX that Pix2Text produces for formulas into the plain-text
style used by the reference transcripts, so the two OCR models are scored on
what they read, not on how they write it down.

  \\frac{1}{2}          -> 1/2
  \\frac{a+b}{c}        -> (a + b)/c
  x^{2}                -> x^2
  x^{n+1}              -> x^(n + 1)
  90^{\\circ}           -> 90°
  \\sqrt{x}             -> √(x)         (\\sqrt[3]{x} -> ∛(x))
  10^{-3}              -> 10^-3
  3 . 7 5              -> 3.75          (spaces inside math carry no meaning)
  \\times \\div \\cdot     -> × ÷ ·
  \\leq \\geq \\neq \\pm    -> ≤ ≥ ≠ ±
  \\$ \\%                -> $ %
  \\text{ cm}           -> cm  (text inside math keeps its spaces)

Inside a formula all spaces are dropped, then a single space is put around
binary operators (= + - × ÷ < > ...), which is how references are written:
"x^2 - 4 = 21", "-5 + 3", "1/2".

Reference transcripts should follow the same style; see benchmarks/REFERENCE_STYLE.md.
"""

from __future__ import annotations

import re

SYMBOLS = {
    "times": "×", "div": "÷", "cdot": "·", "pm": "±", "mp": "∓",
    "le": "≤", "leq": "≤", "leqslant": "≤", "ge": "≥", "geq": "≥", "geqslant": "≥",
    "neq": "≠", "ne": "≠", "approx": "≈", "sim": "~",
    "lt": "<", "gt": ">",
    "circ": "°", "degree": "°", "infty": "∞", "percent": "%",
    "pi": "π", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "Delta": "Δ", "lambda": "λ", "mu": "μ", "sigma": "σ", "Sigma": "Σ", "omega": "ω",
    "Omega": "Ω", "phi": "φ", "rho": "ρ", "epsilon": "ε", "varepsilon": "ε",
    "rightarrow": "→", "to": "→", "leftarrow": "←", "Rightarrow": "⇒",
    "angle": "∠", "triangle": "△", "perp": "⊥", "parallel": "∥",
    "ldots": "...", "cdots": "...", "dots": "...",
    "dollar": "$",
    "S": "§",  # Pix2Text reads printed "$" as \S in the pilot; keep it a symbol, not a letter
}

# Commands whose argument is kept as-is (styling only).
PASSTHROUGH = {"text", "mathrm", "mathbf", "mathit", "textbf", "textit", "operatorname",
               "mbox", "boldsymbol", "mathsf", "textrm", "mathtt", "overline", "underline",
               "hat", "bar", "vec", "widehat", "tilde"}
# Commands whose text content should keep its spaces.
TEXT_COMMANDS = {"text", "mbox", "textbf", "textit", "textrm"}
# Sizing/spacing commands that produce nothing.
DROP = {"left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr",
        "displaystyle", "textstyle", "limits", "nolimits", "quad", "qquad", "hfill",
        "mathbb", "mathcal",  # mathbb/mathcal keep their argument through bare-group handling
        # old-style font and size switches that take no argument: {\bf a} -> a
        "bf", "it", "rm", "sf", "tt", "cal", "scriptstyle", "scriptscriptstyle",
        "tiny", "small", "normalsize", "large", "Large", "huge", "Huge"}

_KEEP_SPACE = "\u00a0"  # protects spaces inside \text{...}

BINARY_OPS = "=+×÷·±∓<>≤≥≠≈→⇒"


def _read_group(s: str, i: int) -> tuple[str, int]:
    """Read one argument starting at s[i]: {...}, \\command or a single char."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        depth, j = 1, i + 1
        while j < len(s) and depth:
            if s[j] == "\\":
                j += 2
                continue
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            j += 1
        return s[i + 1:j - 1] if depth == 0 else s[i + 1:], j
    if s[i] == "\\":
        m = re.match(r"\\([A-Za-z]+|.)", s[i:])
        return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def _read_optional(s: str, i: int) -> tuple[str | None, int]:
    j = i
    while j < len(s) and s[j] == " ":
        j += 1
    if j < len(s) and s[j] == "[":
        k = s.find("]", j)
        if k != -1:
            return s[j + 1:k], k + 1
    return None, i


def _is_atom(x: str) -> bool:
    """True when x needs no parentheses: a number, a letter run or one symbol."""
    x = x.strip()
    return bool(re.fullmatch(r"-?\d+(?:[.,]\d+)*|-?[A-Za-zα-ωΑ-Ω]+|.", x))


def _wrap(x: str) -> str:
    return x if _is_atom(x) else f"({x})"


def _convert(s: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if not m:  # escaped symbol: \% \$ \, \; \{ ...
                nxt = s[i + 1] if i + 1 < len(s) else ""
                out.append({",": " ", ";": " ", ":": " ", "!": "", " ": " ", "\\": " "}.get(nxt, nxt))
                i += 2
                continue
            name = m.group(1)
            i += len(m.group(0))
            if name in ("frac", "dfrac", "tfrac", "cfrac"):
                a, i = _read_group(s, i)
                b, i = _read_group(s, i)
                out.append(f"{_wrap(_convert(a))}/{_wrap(_convert(b))}")
            elif name == "sqrt":
                n, i = _read_optional(s, i)
                a, i = _read_group(s, i)
                root = {None: "√", "2": "√", "3": "∛", "4": "∜"}.get(n.strip() if n else None)
                out.append(f"{root or _convert(n) + '√'}({_convert(a)})")
            elif name in PASSTHROUGH:
                a, i = _read_group(s, i)
                inner = _convert(a)
                if name in TEXT_COMMANDS:
                    inner = inner.replace(" ", _KEEP_SPACE)
                out.append(inner)
            elif name in DROP:
                pass
            elif name in SYMBOLS:
                out.append(SYMBOLS[name])
            else:
                out.append(name)  # unknown command: keep its name rather than lose it
        elif c in "^_":
            a, i = _read_group(s, i + 1)
            inner = _convert(a).strip()
            if c == "^" and inner == "°":
                out.append("°")
            elif inner:
                out.append(c + _wrap(inner))
        elif c == "{":
            a, i = _read_group(s, i)
            out.append(_convert(a))
        elif c == "}":
            i += 1
        elif c == "~":
            out.append(" ")
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _space_operators(s: str) -> str:
    out: list[str] = []
    for k, c in enumerate(s):
        prev = out[-1].strip() if out else ""
        operand_before = bool(prev) and (prev[-1].isalnum() or prev[-1] in ")]°%.")
        if c in BINARY_OPS:
            if c in "+±" and not operand_before:
                out.append(c)  # unary: "+5", "±3"
            else:
                out.append(f" {c} ")
        elif c == "-":
            out.append(" - " if operand_before else "-")
        elif c == "," and k + 1 < len(s) and not s[k + 1].isdigit():
            out.append(", ")
        else:
            out.append(c)
    return re.sub(r" {2,}", " ", "".join(out)).strip()


def latex_to_plain(latex: str) -> str:
    """Convert one formula (without $ delimiters) to plain text."""
    s = (latex or "").strip()
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2]
    elif s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1]
    for env in ("aligned", "align*", "align", "array", "gathered", "cases",
                "matrix", "pmatrix", "bmatrix", "vmatrix", "Bmatrix", "smallmatrix", "split"):
        s = re.sub(r"\\(begin|end)\{" + re.escape(env) + r"\}(\{[^}]*\})?", " ", s)
    s = s.replace("&", " ").replace("\\\\", _KEEP_SPACE)  # row breaks keep rows apart
    converted = _convert(s)
    converted = re.sub(r"[ \t\r\n]+", "", converted)  # spaces inside math carry no meaning (\s would also eat _KEEP_SPACE)
    converted = _space_operators(converted)
    return converted.replace(_KEEP_SPACE, " ")
