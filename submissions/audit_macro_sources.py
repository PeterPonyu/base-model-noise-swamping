#!/usr/bin/env python3
"""
audit_macro_sources.py -- machine audit of quoted numbers in four manuscripts.

For each submissions/<paper>/macros.tex:
  * parse every \\newcommand{\\X}{value} plus its trailing "% ..." provenance comment
  * resolve the named source artifact (JSON under edit-harness/results/ or the paper dir)
  * resolve the field path inside that artifact when the comment states one
  * recompute + compare AT THE MACRO'S DISPLAYED PRECISION
  * classify MATCH / MISMATCH / CANNOT-VERIFY (with a reason)

READ-ONLY with respect to every manuscript. Writes nothing.

Usage:
    python3 submissions/audit_macro_sources.py                 # summary to stdout
    python3 submissions/audit_macro_sources.py --json out.json # machine-readable dump
    python3 submissions/audit_macro_sources.py --macro NAME    # single-macro trace
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "edit-harness", "results")
SUBMISSIONS = os.path.join(ROOT, "submissions")

PAPERS = [
    ("ieee", "B6 geometry/damage law (IEEE TETCI, under review)"),
    ("d2-neurocomputing", "D2 merging federation"),
    ("paper-b-neurocomputing", "Paper B quantisation survival"),
    ("frame-a-eswa", "Frame-A cost-aware maintenance (QUARANTINED)"),
]

# Macros the task pre-declares as expected-mismatch (Phi refix H1 pending).
# Tagged in the report, never counted as defects.
EXPECTED_MISMATCH = {"magPhi", "magPhiSD", "magPhiPeek", "phiSigned"}

# Papers whose stale numbers are reported but never failed.
QUARANTINED = {"frame-a-eswa"}

# Comment markers that self-declare a value as stale / superseded / pre-refix.
STALE_MARKERS = (
    "STALE", "stale", "pre-refix", "PRE-REFIX", "pre-fix", "PRE-FIX",
    "superseded", "SUPERSEDED", "RETRACTED", "retracted", "was ",
    "pending", "PENDING", "TODO", "DEAD", "never quote",
)


# ---------------------------------------------------------------- data classes
@dataclass
class Finding:
    paper: str
    macro: str
    raw_value: str
    quoted: float | None
    recomputed: float | None
    status: str                      # MATCH | MISMATCH | CANNOT-VERIFY
    reason: str = ""                 # populated for CANNOT-VERIFY / MISMATCH
    artifact: str = ""               # resolved file path (relative to ROOT)
    field_path: str = ""             # resolved field path inside the artifact
    comment: str = ""                # the provenance comment, trimmed
    line: int = 0
    expected: bool = False           # pre-declared expected mismatch
    stale_tagged: bool = False       # comment self-declares staleness
    method: str = ""                 # which resolver produced `recomputed`


@dataclass
class MacroDef:
    name: str
    raw_value: str
    comment: str
    line: int
    block_comment: str = ""          # nearest preceding "% source:" block


# ------------------------------------------------------------- value scrubbing
_LATEX_STRIP = [
    (re.compile(r"\\ensuremath\s*"), ""),
    (re.compile(r"\\text\s*\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\,|\\;|\\!|\\ "), ""),
    (re.compile(r"\{,\}"), ""),          # 11{,}427 -> 11427
    (re.compile(r"\\%"), ""),
    (re.compile(r"\\\$"), ""),
    (re.compile(r"[$]"), ""),
    (re.compile(r"\\pm"), " +- "),
    (re.compile(r"[\u2212\u2013\u2014]"), "-"),   # unicode minus / en / em dash
]

_SCI = re.compile(
    r"^([+-]?\d*\.?\d+)\s*\\times\s*10\^\{?\s*([+-]?\d+)\s*\}?$"
)


def strip_latex(s: str) -> str:
    out = s
    for pat, rep in _LATEX_STRIP:
        out = pat.sub(rep, out)
    return out.strip()


def parse_scalar(raw: str) -> float | None:
    """Best-effort single-number extraction from a macro body. None if not scalar."""
    s = strip_latex(raw)
    if not s:
        return None
    m = _SCI.match(s)
    if m:
        try:
            return float(m.group(1)) * (10.0 ** int(m.group(2)))
        except ValueError:
            return None
    # plain number, optionally leading + / -
    m = re.fullmatch(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", s)
    if m:
        try:
            return float(s)
        except ValueError:
            return None
    return None


def displayed_decimals(raw: str) -> int:
    """Number of decimal digits the macro actually shows (drives comparison precision)."""
    s = strip_latex(raw)
    m = _SCI.match(s)
    if m:
        frac = m.group(1).split(".")
        return len(frac[1]) if len(frac) > 1 else 0
    m = re.search(r"\.(\d+)", s)
    return len(m.group(1)) if m else 0


def sig_round(x: float, decimals: int) -> float:
    """Round-half-away-from-zero at `decimals`, matching how humans typeset."""
    if x is None:
        return None
    factor = 10.0 ** decimals
    scaled = x * factor
    # nudge away from binary-representation ties (2.675 -> 2.68, not 2.67)
    eps = 1e-9 * (1 if scaled >= 0 else -1)
    return math.floor(abs(scaled) + 0.5 + abs(eps)) / factor * (1 if x >= 0 else -1)


# --------------------------------------------------------------- macros parser
# \newcommand{\name}{body}  with brace-balanced body, then optional % comment.
_NEWCMD = re.compile(r"\\newcommand\s*\{\s*\\([A-Za-z@]+)\s*\}\s*\{")


def _balanced_body(text: str, start: int) -> tuple[str, int]:
    """Read a brace-balanced group starting at `start` (which points just past '{')."""
    depth = 1
    i = start
    while i < len(text) and depth:
        c = text[i]
        if c == "\\":            # skip escaped char
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    return text[start:i], i


def parse_macros(path: str) -> list[MacroDef]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    lines = text.split("\n")

    # map char offset -> 1-based line number
    offsets, acc = [], 0
    for ln in lines:
        offsets.append(acc)
        acc += len(ln) + 1

    def line_of(pos: int) -> int:
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    out: list[MacroDef] = []
    for m in _NEWCMD.finditer(text):
        # skip definitions inside a commented-out line
        lno = line_of(m.start())
        line_text = lines[lno - 1]
        col = m.start() - offsets[lno - 1]
        if "%" in line_text[:col]:
            continue
        body, end = _balanced_body(text, m.end())
        # trailing same-line comment
        tail = text[end:]
        nl = tail.find("\n")
        tail_line = tail if nl < 0 else tail[:nl]
        cm = re.search(r"%\s*(.*)$", tail_line)
        comment = cm.group(1).strip() if cm else ""
        # continuation: subsequent pure-comment lines that are indented notes
        out.append(MacroDef(m.group(1), body, comment, lno))

    # Attach the nearest preceding "% source:" block to each macro.
    # A block only ABSORBS following comment lines while they are CONTIGUOUS with
    # it; any intervening \newcommand or rule line ends absorption. Without this,
    # an unrelated later comment ("% per-edit rhos") leaks into the block and
    # mis-attributes a field (observed false positive on \gResidNorm).
    block = ""        # the "% source: X.json" statement in scope
    section = ""      # the "% --- <statistic description> ---" header in scope
    absorbing = False
    idx = 0
    for lno, line_text in enumerate(lines, start=1):
        stripped = line_text.strip()
        is_macro_line = idx < len(out) and out[idx].line == lno
        if stripped.startswith("%") and not is_macro_line:
            body = stripped.lstrip("%").strip()
            core = body.strip("-= ").strip()
            if not core:
                absorbing = False
            elif re.search(r"(?i)\bsources?\b\s*:", body):
                # a header line may carry the source inline: "--- X, 3-seed. source: f.json ---"
                if re.match(r"^[-=]{3,}", body):
                    section = core
                block, absorbing = core, True
            elif re.match(r"^[-=]{3,}", body):
                section, block, absorbing = core, "", False   # new section resets source
            elif absorbing:
                block += " " + core
            else:
                section = (section + " " + core).strip()
        elif is_macro_line:
            absorbing = False              # macros stop absorption; scope still applies
        while idx < len(out) and out[idx].line == lno:
            out[idx].block_comment = (block + " || " + section).strip(" |")
            idx += 1
    return out


# ------------------------------------------------------- artifact file resolver
_JSON_CACHE: dict[str, Any] = {}
_FILE_INDEX: dict[str, list[str]] | None = None


def _build_index() -> dict[str, list[str]]:
    """basename -> [absolute paths], over results/ and submissions/ JSON artifacts."""
    global _FILE_INDEX
    if _FILE_INDEX is not None:
        return _FILE_INDEX
    idx: dict[str, list[str]] = {}
    for base in (RESULTS, SUBMISSIONS):
        for dirpath, dirnames, filenames in os.walk(base):
            # never resolve into quarantined / invalidated artifact trees
            dirnames[:] = [d for d in dirnames if not d.startswith("_invalid")
                           and d not in {"node_modules", ".git"}]
            for fn in filenames:
                if fn.endswith(".json"):
                    idx.setdefault(fn, []).append(os.path.join(dirpath, fn))
    _FILE_INDEX = idx
    return idx


def load_json(path: str) -> Any:
    if path not in _JSON_CACHE:
        with open(path, "r", encoding="utf-8") as fh:
            _JSON_CACHE[path] = json.load(fh)
    return _JSON_CACHE[path]


def invalidated_twin(name: str) -> str | None:
    """Detect an artifact that exists ONLY as a quarantined/invalidated rename."""
    base = os.path.basename(name)
    for dirpath, dirnames, filenames in os.walk(RESULTS):
        dirnames[:] = [d for d in dirnames if d not in {"node_modules", ".git"}]
        for fn in filenames:
            if fn.startswith(base + "."):
                tail = fn[len(base) + 1:]
                if "INVALID" in tail.upper() or "STALE" in tail.upper() \
                        or "PARTIAL" in tail.upper() or "KILLED" in tail.upper():
                    return os.path.relpath(os.path.join(dirpath, fn), ROOT)
    return None


def resolve_artifact(name: str, paper_dir: str) -> str | None:
    """Resolve a JSON filename / relative path mentioned in a comment to an abs path."""
    name = name.strip().strip("`'\"(),;")
    if not name.endswith(".json"):
        return None
    # explicit relative path forms
    cands = [
        os.path.join(ROOT, name),
        os.path.join(RESULTS, name),
        os.path.join(ROOT, "edit-harness", name),
        os.path.join(paper_dir, name),
        name if os.path.isabs(name) else None,
    ]
    for c in cands:
        if c and os.path.isfile(c) and "_invalid" not in c:
            return c
    hits = _build_index().get(os.path.basename(name), [])
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # prefer results/ top level, then shortest path (least nested)
        hits = sorted(hits, key=lambda p: (os.path.dirname(p) != RESULTS, len(p)))
        return hits[0]
    return None


# --------------------------------------------------------------- field lookups
def dig(obj: Any, parts: Iterable[str]) -> tuple[bool, Any]:
    """Walk a dotted path. Tolerates list-of-dicts (matches on any scalar field)."""
    cur = obj
    for p in parts:
        p = p.strip().strip("[]")
        if p == "":
            continue
        if isinstance(cur, dict):
            if p in cur:
                cur = cur[p]
                continue
            # numeric-vs-string key tolerance ("12" vs 12)
            alt = None
            for k in cur:
                if str(k) == p:
                    alt = k
                    break
            if alt is not None:
                cur = cur[alt]
                continue
            return False, None
        if isinstance(cur, list):
            if re.fullmatch(r"\d+", p):
                i = int(p)
                if 0 <= i < len(cur):
                    cur = cur[i]
                    continue
                return False, None
            # "L12" selector over a list of records -> the record with layer == 12
            lm = re.fullmatch(r"[Ll](\d{1,3})", p)
            if lm:
                want = int(lm.group(1))
                matches = [r for r in cur
                           if isinstance(r, dict) and r.get("layer") == want]
                if len(matches) == 1:
                    cur = matches[0]
                    continue
                return False, None
            # list of records: select the one whose value matches p exactly
            matches = [r for r in cur
                       if isinstance(r, dict) and any(str(v) == p for v in r.values())]
            if len(matches) == 1:
                cur = matches[0]
                continue
            return False, None
        return False, None
    return True, cur


def subsequence_lookup(doc: Any, parts: list[str],
                       max_depth: int = 9) -> list[tuple[str, float]]:
    """Find numeric leaves reachable by matching `parts` as an ordered subsequence.

    Handles two real provenance idioms:
      * an intermediate container is omitted   (profile.L14.x  vs rome_depth_profile.L14.x)
      * the final segment is abbreviated       (....ratio  vs  ratio_rome_over_alpha)
    Returns [(concrete_path, value)]. The caller requires a UNIQUE value.
    """
    if not parts:
        return []
    out: list[tuple[str, float]] = []

    def key_matches(key: str, want: str) -> bool:
        k, w = str(key), want
        if k == w:
            return True
        kl, wl = k.lower(), w.lower()
        if kl == wl:
            return True
        # "L14" selector against key "L14"/"14"; abbreviation of a longer key
        if re.fullmatch(r"[Ll]\d{1,3}", w) and kl in {wl, wl[1:]}:
            return True
        return wl != "" and (kl.startswith(wl + "_") or kl.endswith("_" + wl))

    def walk(node: Any, remaining: list[str], trail: str, depth: int) -> None:
        if depth > max_depth or len(out) > 40:
            return
        if not remaining:
            v = numeric(node)
            if v is not None:
                out.append((trail, v))
            return
        want = remaining[0]
        if isinstance(node, dict):
            for k, v in node.items():
                if key_matches(k, want):
                    walk(v, remaining[1:], f"{trail}.{k}" if trail else str(k), depth + 1)
                else:
                    walk(v, remaining, f"{trail}.{k}" if trail else str(k), depth + 1)
        elif isinstance(node, list):
            lm = re.fullmatch(r"[Ll](\d{1,3})", want)
            for i, v in enumerate(node):
                if lm and isinstance(v, dict) and v.get("layer") == int(lm.group(1)):
                    walk(v, remaining[1:], f"{trail}[{i}]", depth + 1)
                else:
                    walk(v, remaining, f"{trail}[{i}]", depth + 1)

    walk(doc, parts, "", 0)
    return out


def find_leaf(obj: Any, leaf: str, max_depth: int = 8) -> list[Any]:
    """Collect every value stored under key `leaf` anywhere in the tree."""
    found: list[Any] = []

    def walk(node: Any, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k) == leaf:
                    found.append(v)
                walk(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                walk(v, depth + 1)

    walk(obj, 0)
    return found


def numeric(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


# ------------------------------------------------------- comment shape parsing
# "FILE.json::path.to.field = 0.5945"
RE_DBLCOLON = re.compile(
    r"([A-Za-z0-9_.\-/]+\.json)\s*::\s*([A-Za-z0-9_.\[\]()\-]+?)\s*(?:=\s*([+-]?[\d.eE+\-]+))?\s*(?:$|[,;)\s])"
)
# "layers.12.within_probe_spearman = 0.5675"  (path with an = value, no file)
RE_PATH_EQ = re.compile(
    r"(?<![:\w])([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_\[\]]+){1,6})\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
)
# "field_name = 0.4015"  (bare leaf key with a value)
RE_LEAF_EQ = re.compile(
    r"(?<![:.\w])([a-z][a-z0-9_]{3,})\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
)
# arithmetic: "1 - 1.773/3.092", "65868 + 26946"
RE_ARITH = re.compile(r"^[\s\d.+\-*/()]+$")
# a JSON filename anywhere in the comment (allows brace-set globs: G1_L{8,10}_x.json)
RE_JSONFILE = re.compile(r"([A-Za-z0-9_.\-/]*(?:\{[^}]*\})?[A-Za-z0-9_.\-/]*\.json)")


def expand_braces(name: str) -> list[str]:
    """G1_L{8,10,12,14}_analysis.json -> the four concrete filenames."""
    m = re.search(r"\{([^}]*)\}", name)
    if not m:
        return [name]
    out: list[str] = []
    for alt in m.group(1).split(","):
        out.extend(expand_braces(name[:m.start()] + alt.strip() + name[m.end():]))
    return out


# Layer hints in a macro name, so \gWithinLtwelve picks G1_L12_analysis.json
_LAYER_WORDS = {
    "eight": 8, "ten": 10, "twelve": 12, "fourteen": 14, "sixteen": 16,
    "eighteen": 18, "twentyone": 21, "twentyfour": 24, "twentyeight": 28,
    "thirty": 30, "thirtysix": 36, "thirtythree": 33, "twenty": 20,
}


def layer_hint(macro: str, comment: str) -> int | None:
    """Extract the layer a macro refers to, from its name or its own comment."""
    m = re.search(r"(?:^|[^A-Za-z])L(\d{1,2})(?![\d])", comment)
    if m:
        return int(m.group(1))
    low = macro.lower()
    for word, n in sorted(_LAYER_WORDS.items(), key=lambda kv: -len(kv[0])):
        if "l" + word in low:
            return n
    m = re.search(r"l(\d{1,2})(?![\d])", low)
    return int(m.group(1)) if m else None


# A slash-separated run of 2+ numbers is a PER-SEED LIST, not a division.
RE_SEEDLIST = re.compile(
    r"(?<![\d./])(\d*\.?\d+(?:\s*/\s*\d*\.?\d+){1,5})(?![\d.]*\s*/\s*\d*\.?\d+\s*/)"
)


def find_seedlist(text: str) -> list[float] | None:
    """Extract a per-seed value list like '0.9997/0.9993/0.9997' or '0.403/0.406/0.382'."""
    best: list[float] | None = None
    for m in re.finditer(r"(?<![\w.\-])(\d*\.\d+(?:\s*/\s*\d*\.\d+){1,5})(?![\w.])", text):
        parts = [p.strip() for p in m.group(1).split("/")]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        # require >=2 entries and a consistent decimal style (a real seed list)
        if len(vals) >= 2 and (best is None or len(vals) > len(best)):
            best = vals
    return best


def find_arith(text: str) -> str | None:
    """Extract a *deliberate* arithmetic expression, e.g. '1 - 1.773/3.092'.

    Guarded hard: model names ("Llama-3.2-1B"), dates ("20260714"), layer tags
    ("L2-13") and version strings all look like arithmetic to a naive scanner and
    produced bogus reasons in an earlier revision. Require whitespace around the
    operator, which is what a human-written computation actually looks like.
    """
    for m in re.finditer(
            r"(?<![\w.\-/])(\(?\d[\d.]*\)?(?:\s+[-+*/]\s+\(?\d[\d.]*\)?)+)(?![\w.\-])",
            text):
        expr = m.group(1)
        if RE_ARITH.fullmatch(expr) and re.search(r"\s[-+*/]\s", expr):
            return expr
    return None


def eval_arith(expr: str) -> float | None:
    if not RE_ARITH.fullmatch(expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))   # noqa: S307 - digits/ops only
    except Exception:
        return None


# --------------------------------------------- canonical headline-field mapping
# A macro whose comment names a statistic but no field path can still be verified
# IF the artifact stores that statistic under exactly one canonical field. Each
# rule below requires BOTH a statistic phrase (in the macro's own comment or its
# source block) AND that the resulting leaf be unique in the artifact. Anything
# outside these rules stays CANNOT-VERIFY -- the auditor never guesses.
CANONICAL_LEAVES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    # (leaf field, required phrases in context, disqualifying phrases)
    ("within_probe_mean_across_seeds",
     ("within-probe",), ("per-edit", "norm-growth", "norm_growth",
                         "residual-norm", "norm growth")),
    ("within_probe_std_across_seeds",
     ("within-probe",), ("per-edit",)),
    ("per_edit_mean_across_seeds",
     ("per-edit",), ("within-probe",)),
]


def canonical_leaf(md: MacroDef, ctx: str) -> str | None:
    """Map a macro to a single canonical artifact field, or None (=> CANNOT-VERIFY).

    Deliberately strict. The statistic phrase must come from the macro's OWN
    comment or its source block, AND the macro name must not signal a different
    quantity (residual norm, floor, count, ...). Anything ambiguous returns None.
    """
    low = ctx.lower()
    name_low = md.name.lower()

    # Macro names that denote something other than the block's headline statistic.
    if any(tok in name_low for tok in
           ("residnorm", "permfloor", "meanc", "nseed", "npair", "count",
            "ratio", "pct", "thresh", "arxiv", "nfact")):
        return None

    # The macro NAME is authoritative for mean-vs-dispersion. A comment may
    # mention its partner's SD in passing ("0.395  % L8 (SD 0.011)"), so the
    # comment must never promote a mean macro to the std field.
    is_sd = bool(re.search(r"(SD|Std|Sigma)$", md.name))
    for leaf, need, block in CANONICAL_LEAVES:
        if leaf.endswith("std_across_seeds") and not is_sd:
            continue
        if leaf.endswith("mean_across_seeds") and is_sd:
            continue
        if all(n.lower() in low for n in need) and \
                not any(b.lower() in low for b in block):
            return leaf
    return None


# ------------------------------------------------------------------ the auditor
def audit_macro(md: MacroDef, paper: str, paper_dir: str) -> Finding:
    ctx = (md.comment + "  ||  " + md.block_comment).strip()
    quoted = parse_scalar(md.raw_value)
    dec = displayed_decimals(md.raw_value)

    f = Finding(
        paper=paper, macro=md.name, raw_value=md.raw_value, quoted=quoted,
        recomputed=None, status="CANNOT-VERIFY", comment=md.comment[:220],
        line=md.line,
        expected=md.name in EXPECTED_MISMATCH,
        stale_tagged=any(mk in ctx for mk in STALE_MARKERS),
    )

    if quoted is None:
        f.reason = "non-scalar macro body (string/list/range/CI -- not a single number)"
        return f

    # ---------- shape 1: FILE.json::field.path [= value]
    for m in RE_DBLCOLON.finditer(ctx):
        art = resolve_artifact(m.group(1), paper_dir)
        if art is None:
            f.artifact = m.group(1)
            f.field_path = m.group(2)
            twin = invalidated_twin(m.group(1))
            f.reason = (f"artifact not found on disk: {m.group(1)}"
                        + (f" -- exists ONLY as quarantined {twin}" if twin else ""))
            if twin:
                f.stale_tagged = True
            return f
        path = m.group(2)
        doc = load_json(art)
        f.artifact = os.path.relpath(art, ROOT)
        parts = [p for p in re.split(r"[.\[\]]+", path) if p]
        ok, val = dig(doc, parts)
        if ok and numeric(val) is not None:
            f.field_path = path
            f.recomputed = numeric(val)
            f.method = "json::path"
            return classify(f, dec)
        # tolerate a path that omits an intermediate container, or abbreviates a
        # leaf ("...ratio" for "ratio_rome_over_alpha"): walk the declared
        # segments as an ordered subsequence and require a UNIQUE landing site.
        cands = subsequence_lookup(doc, parts)
        uniq = sorted({round(v, 8) for _, v in cands})
        if len(cands) >= 1 and len(uniq) == 1:
            f.field_path = f"{path} -> {cands[0][0]}"
            f.recomputed = cands[0][1]
            f.method = "json path (subsequence, unique)"
            return classify(f, dec)
        f.field_path = path
        if len(uniq) > 1:
            f.reason = (f"declared path did not resolve; {len(uniq)} distinct "
                        f"candidate values under a relaxed match -- ambiguous")
        else:
            f.reason = "declared path did not resolve in artifact"
        return f

    # ---------- shape 2: artifact named in the comment (possibly brace-expanded)
    named = RE_JSONFILE.findall(ctx)
    # expand {8,10,12,14} sets, then prefer the file matching this macro's layer
    cands: list[str] = []
    for n in named:
        cands.extend(expand_braces(n))
    hint = layer_hint(md.name, md.comment)
    if hint is not None:
        pref = [c for c in cands if re.search(rf"[_L]{hint}(?![\d])", c)]
        cands = pref + [c for c in cands if c not in pref]

    art = None
    unresolved: list[str] = []
    for cand in cands:
        got = resolve_artifact(cand, paper_dir)
        if got:
            art = got
            break
        unresolved.append(cand)

    if art is None and unresolved:
        twin = invalidated_twin(unresolved[0])
        if twin:
            f.artifact = unresolved[0]
            f.stale_tagged = True
            f.reason = (f"declared artifact {unresolved[0]} exists ONLY as "
                        f"quarantined/invalidated {twin}")
            return f

    if art:
        f.artifact = os.path.relpath(art, ROOT)
        doc = load_json(art)
        for m in RE_PATH_EQ.finditer(ctx):
            path = m.group(1)
            if path.endswith(".json") or ".json" in path:
                continue
            ok, val = dig(doc, re.split(r"[.\[\]]+", path))
            if ok and numeric(val) is not None:
                f.field_path = path
                f.recomputed = numeric(val)
                f.method = "path=value in comment"
                return classify(f, dec)
        # 2b: no path stated, but the artifact carries a canonical headline field.
        #     Only used when the macro's semantic role is unambiguous from its
        #     block comment (which names the statistic) -- see CANONICAL_LEAVES.
        canon = canonical_leaf(md, ctx)
        # Layer-agreement guard: if the macro states a layer and the resolved
        # artifact filename states a DIFFERENT layer, refuse. Without this a
        # brace-set source block silently pairs an L12 macro to the L8 file
        # (observed false positive on \rewriteRef).
        if canon and hint is not None:
            fm = re.search(r"[_L](\d{1,2})(?![\d])", os.path.basename(art))
            if fm and int(fm.group(1)) != hint:
                f.field_path = canon
                f.reason = (f"macro refers to L{hint} but the only resolvable "
                            f"artifact is {os.path.basename(art)} (L{fm.group(1)}) "
                            f"-- provenance comment does not name this macro's artifact")
                return f
        if canon:
            hits = [numeric(v) for v in find_leaf(doc, canon)]
            hits = [h for h in hits if h is not None]
            uniq = sorted(set(round(h, 6) for h in hits))
            if len(uniq) == 1:
                f.field_path = canon
                f.recomputed = hits[0]
                f.method = f"canonical headline field {canon!r}"
                return classify(f, dec)

        for m in RE_LEAF_EQ.finditer(ctx):
            leaf = m.group(1)
            hits = [numeric(v) for v in find_leaf(doc, leaf)]
            hits = [h for h in hits if h is not None]
            uniq = sorted(set(round(h, 6) for h in hits))
            if len(uniq) == 1:
                f.field_path = leaf
                f.recomputed = hits[0]
                f.method = "leaf=value in comment (unique)"
                return classify(f, dec)
            # ambiguous leaf: accept only if exactly one candidate matches at precision
            near = [h for h in hits if sig_round(h, dec) == sig_round(quoted, dec)]
            if len(near) >= 1 and len(hits) > 1:
                f.field_path = f"{leaf} (ambiguous: {len(hits)} occurrences)"
                f.reason = (f"leaf {leaf!r} occurs {len(hits)} times in artifact; "
                            f"no unambiguous field path -- not mechanically verifiable")
                return f
        f.reason = ("artifact resolved but no parseable field path in the comment "
                    "(prose-only provenance)")
        # fall through to arithmetic below in case the comment states one
    # ---------- shape 2d: Frame-A per-cell glob (means over seed cell files)
    fa = framea_resolve(md, ctx)
    if fa is not None:
        f.recomputed, f.field_path = fa[0], fa[1]
        f.artifact = os.path.relpath(FRAMEA_CELLS, ROOT)
        f.method = "frame-a per-cell glob mean"
        return classify(f, dec)

    # ---------- shape 2c: the comment lists the per-seed values it averages
    # e.g. "\cbcRhoLeight}{0.397}  % (0.403/0.406/0.382)" -> mean must reproduce.
    seeds = find_seedlist(md.comment)
    if seeds and len(seeds) >= 2:
        mean = sum(seeds) / len(seeds)
        if sig_round(mean, dec) == sig_round(quoted, dec):
            f.recomputed = mean
            f.field_path = f"mean of per-seed list in comment {seeds}"
            f.method = "per-seed mean stated in comment"
            f.status = "MATCH"
            return f
        # a stated list that does not average to the macro is a real discrepancy
        if not f.artifact:
            f.recomputed = mean
            f.field_path = f"per-seed list {seeds}"
            f.method = "per-seed mean stated in comment"
            f.status = "MISMATCH"
            f.reason = (f"comment lists per-seed {seeds} (mean {mean:.4f}) but the "
                        f"macro quotes {f.raw_value}")
            return f

    # ---------- shape 3: derived arithmetic stated in the comment
    expr = find_arith(ctx)
    if expr is not None:
        val = eval_arith(expr)
        if val is not None:
            got = sig_round(val, dec)
            if got == sig_round(quoted, dec):
                f.recomputed = val
                f.method = f"arithmetic in comment: {expr}"
                f.field_path = expr
                return classify(f, dec)
            # arithmetic present but does not reproduce -> report, do not silently fail
            f.reason = (f"derived arithmetic ({expr} = {val:.6g}) does not reproduce "
                        f"the quoted value at {dec}dp -- inputs likely stated elsewhere")
            if not f.artifact:
                f.artifact = ""
            return f

    # ---------- shape 4: nothing resolvable
    if not f.reason:
        if not named:
            f.reason = "no artifact named in the provenance comment"
        else:
            f.reason = (f"named artifact(s) {named[:2]} not found on disk "
                        f"(or not JSON-resolvable)")
    return f


def classify(f: Finding, dec: int) -> Finding:
    q = sig_round(f.quoted, dec)
    r = sig_round(f.recomputed, dec)
    if q == r:
        f.status = "MATCH"
        return f
    # allow the neighbouring-digit case that pure float rounding can produce
    if abs(f.quoted - f.recomputed) <= 1.0001 * (10.0 ** -dec) / 2:
        f.status = "MATCH"
        f.reason = "matches within half of the last displayed digit"
        return f
    f.status = "MISMATCH"
    f.reason = (f"quoted {f.quoted} vs artifact {f.recomputed:.6g} "
                f"(rounded to {dec}dp: {q} vs {r})")
    return f


# -------------------------------------------- Frame-A per-cell glob aggregation
# Frame-A macros are means over per-(policy, seed) cell files, and the provenance
# comment names the glob:
#   % SOURCE: .../cells/cell_llama-3.2-1b_real_MIX_A_<policy>_s{0,1,2}.json
# The macro name encodes which policy and which quantity: \qEdit -> always_edit
# quality.Q ; \cRag -> always_rag cost.total_gpu_s. Resolve by expanding the glob.
FRAMEA_CELLS = os.path.join(RESULTS, "frame_a", "cells")

FRAMEA_POLICY = {           # macro-name suffix -> policy arm in the cell filename
    "both": "both", "grace": "always_grace", "edit": "always_edit",
    "rag": "always_rag", "costonly": "cost_only", "damageonly": "damage_only",
    "oracle": "oracle", "random": "random", "ft": "always_ft",
}
FRAMEA_QUANTITY = {         # macro-name prefix -> (field path, aggregator)
    "q": ("quality.Q", "mean"),
    "c": ("cost.total_gpu_s", "mean"),
}


def framea_resolve(md: MacroDef, ctx: str) -> tuple[float, str] | None:
    """Recompute a Frame-A macro as the across-seed mean of its per-cell files."""
    if "cells/cell_" not in ctx and "cell_llama" not in ctx:
        return None
    if not os.path.isdir(FRAMEA_CELLS):
        return None
    m = re.fullmatch(r"([qc])([A-Z][A-Za-z]*?)([ABC])?", md.name)
    if not m:
        return None
    quant, arm, mix = m.group(1), m.group(2).lower(), m.group(3) or "A"
    if quant not in FRAMEA_QUANTITY or arm not in FRAMEA_POLICY:
        return None
    fpath, _agg = FRAMEA_QUANTITY[quant]
    policy = FRAMEA_POLICY[arm]
    pat = re.compile(
        rf"^cell_.*_MIX_{mix}_{re.escape(policy)}_s\d+\.json$")
    vals: list[float] = []
    used: list[str] = []
    for fn in sorted(os.listdir(FRAMEA_CELLS)):
        if pat.match(fn):
            ok, v = dig(load_json(os.path.join(FRAMEA_CELLS, fn)), fpath.split("."))
            nv = numeric(v) if ok else None
            if nv is not None:
                vals.append(nv)
                used.append(fn)
    # The glob declares s{0,1,2}: require the full seed set on disk, else the
    # recomputation would silently average a DIFFERENT n than the paper did.
    want_seeds = re.search(r"s\{([^}]*)\}", ctx)
    n_want = len([s for s in want_seeds.group(1).split(",")]) if want_seeds else 3
    if not vals or len(vals) != n_want:
        return None
    return (sum(vals) / len(vals),
            f"mean({fpath}) over {len(vals)} cells MIX_{mix}/{policy}")


# ------------------------------------------------- generator-regeneration check
# Some macro files are AUTO-GENERATED from an artifact by a named script. For
# those, the authoritative check is not comment parsing but regeneration: run the
# generator to a temp path and diff macro-by-macro. Every macro then becomes a
# genuine MATCH/MISMATCH instead of CANNOT-VERIFY.
GENERATORS: dict[str, tuple[str, list[str]]] = {
    # paper dir -> (script relative to edit-harness/, extra argv)
    "paper-b-neurocomputing": ("experiments/quant_survival_macros.py", []),
}


def regenerate(paper: str) -> dict[str, str] | None:
    """Run the paper's macro generator and return {macro_name: value}, or None."""
    if paper not in GENERATORS:
        return None
    script, extra = GENERATORS[paper]
    harness = os.path.join(ROOT, "edit-harness")
    spath = os.path.join(harness, script)
    if not os.path.isfile(spath):
        return None
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "regen_macros.tex")
        try:
            proc = subprocess.run(
                [sys.executable, spath, "--out_path", out, *extra],
                cwd=harness, capture_output=True, text=True, timeout=900,
            )
        except Exception:
            return None
        if proc.returncode != 0 or not os.path.isfile(out):
            return None
        return {m.name: m.raw_value for m in parse_macros(out)}


def apply_regeneration(findings: list[Finding], paper: str,
                       regen: dict[str, str]) -> None:
    """Overwrite each finding for `paper` with the regeneration verdict."""
    for f in findings:
        if f.paper != paper:
            continue
        if f.macro not in regen:
            f.status = "CANNOT-VERIFY"
            f.reason = ("macro absent from generator output -- hand-added to an "
                        "auto-generated file (generator is the declared source)")
            f.method = "regeneration"
            continue
        want = regen[f.macro]
        f.method = "regeneration (generator re-run from artifact)"
        f.field_path = "generator output"
        f.artifact = GENERATORS[paper][0]
        if strip_latex(want) == strip_latex(f.raw_value):
            f.status, f.reason = "MATCH", ""
            f.recomputed = parse_scalar(want)
            continue
        rq, rr = parse_scalar(f.raw_value), parse_scalar(want)
        f.recomputed = rr
        if rq is not None and rr is not None:
            classify(f, displayed_decimals(f.raw_value))
            if f.status == "MISMATCH":
                f.reason = (f"committed {f.raw_value!r} but generator emits "
                            f"{want!r} from the current artifact")
        else:
            f.status = "MISMATCH"
            f.reason = (f"committed {f.raw_value!r} but generator emits {want!r}")


# ----------------------------------------------------------------------- driver
CV_BUCKETS = [
    ("non-scalar macro body", "non-scalar (string / list / range / CI / label)"),
    ("no artifact named", "no artifact named in comment"),
    ("not found on disk", "named artifact missing on disk"),
    ("artifact not found on disk", "named artifact missing on disk"),
    ("no parseable field path", "artifact known, field path not machine-parseable"),
    ("declared path did not resolve", "declared field path absent from artifact"),
    ("ambiguous", "field name ambiguous inside artifact"),
    ("derived arithmetic", "derived arithmetic does not close from comment alone"),
]


def bucket_of(reason: str) -> str:
    for needle, label in CV_BUCKETS:
        if needle in reason:
            return label
    return "other / unclassified"


def run_audit() -> list[Finding]:
    findings: list[Finding] = []
    for paper, _desc in PAPERS:
        pdir = os.path.join(SUBMISSIONS, paper)
        mpath = os.path.join(pdir, "macros.tex")
        if not os.path.isfile(mpath):
            print(f"WARNING: no macros.tex for {paper}", file=sys.stderr)
            continue
        for md in parse_macros(mpath):
            findings.append(audit_macro(md, paper, pdir))
        regen = regenerate(paper)
        if regen:
            apply_regeneration(findings, paper, regen)
    return findings


def counts(fs: list[Finding]) -> dict[str, int]:
    out = {"MATCH": 0, "MISMATCH": 0, "CANNOT-VERIFY": 0}
    for f in fs:
        out[f.status] = out.get(f.status, 0) + 1
    return out


def verdict(paper: str, fs: list[Finding]) -> str:
    real = [f for f in fs
            if f.status == "MISMATCH" and not f.expected and not f.stale_tagged]
    if paper in QUARANTINED:
        n = len([f for f in fs if f.status == "MISMATCH"])
        return f"QUARANTINED (informational; {n} mismatch(es) reported, none failed)"
    return "CLEAN" if not real else f"DEFECTS-{len(real)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH", help="write the full findings as JSON")
    ap.add_argument("--macro", metavar="NAME", help="trace a single macro and exit")
    ap.add_argument("--paper", metavar="DIR", help="restrict to one paper directory")
    args = ap.parse_args()

    findings = run_audit()
    if args.paper:
        findings = [f for f in findings if f.paper == args.paper]

    if args.macro:
        for f in findings:
            if f.macro == args.macro:
                for k, v in asdict(f).items():
                    print(f"{k:14s}: {v}")
                return 0
        print(f"macro \\{args.macro} not found")
        return 1

    print("=" * 78)
    print("MACRO SOURCE AUDIT")
    print("=" * 78)
    for paper, desc in PAPERS:
        fs = [f for f in findings if f.paper == paper]
        if not fs:
            continue
        c = counts(fs)
        print(f"\n## {paper}  ({desc})")
        print(f"   macros={len(fs)}  MATCH={c['MATCH']}  "
              f"MISMATCH={c['MISMATCH']}  CANNOT-VERIFY={c['CANNOT-VERIFY']}")
        mm = [f for f in fs if f.status == "MISMATCH"]
        if mm:
            print("   MISMATCHES:")
            for f in mm:
                tag = ""
                if f.expected:
                    tag = " [EXPECTED: Phi refix pending]"
                elif f.stale_tagged:
                    tag = " [comment self-declares stale/pre-refix]"
                print(f"     \\{f.macro} (L{f.line}) = {f.raw_value!r} "
                      f"vs {f.recomputed} <- {f.artifact}::{f.field_path}{tag}")
        print(f"   VERDICT: {verdict(paper, fs)}")

    print("\n" + "-" * 78)
    print("CANNOT-VERIFY grouped by reason (all papers)")
    print("-" * 78)
    groups: dict[str, list[Finding]] = {}
    for f in findings:
        if f.status == "CANNOT-VERIFY":
            groups.setdefault(bucket_of(f.reason), []).append(f)
    for label in sorted(groups, key=lambda k: -len(groups[k])):
        print(f"  {label}: {len(groups[label])}")

    tot = counts(findings)
    print(f"\nTOTAL macros={len(findings)}  MATCH={tot['MATCH']}  "
          f"MISMATCH={tot['MISMATCH']}  CANNOT-VERIFY={tot['CANNOT-VERIFY']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([asdict(f) for f in findings], fh, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
