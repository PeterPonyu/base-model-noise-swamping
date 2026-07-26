"""models.py -- resolve Ollama models to {name, family, architecture, lineage}.

Parses `ollama show <name>` (metadata only; NO inference) for the architecture
field, and derives lineage (base-instruct vs r1-distill) from the model name.

Also encodes the pre-registered design groups for the P3 lineage-vs-architecture
IPI fingerprint:

  Group A  R1-distills (lineage=r1-distill), all Qwen-lineage on this box:
    deepseek-r1:1.5b   arch qwen2  (Qwen2.5-1.5B distill)   scale=small
    deepseek-r1:7b     arch qwen2  (Qwen2.5-7B  distill)    scale=mid
    deepseek-r1:8b     arch qwen3  (Qwen3-8B    distill)    scale=large
  Group B  matched vanilla bases (lineage=base-instruct):
    qwen2.5:1.5b               arch qwen2  scale=small
    qwen2.5:7b-instruct-q8_0   arch qwen2  scale=mid
    qwen3:8b                   arch qwen3  scale=large
  Group C  out-group (different families):
    llama3.1:8b-instruct-q4_K_M   arch llama    scale=large
    gemma2:9b-instruct-q8_0       arch gemma2   scale=large
    mistral:7b-instruct-v0.3-q4_K_M arch llama  scale=mid

Matched (architecture) pairs share (arch_family, scale) and differ in lineage:
  (r1:1.5b, qwen2.5:1.5b), (r1:7b, qwen2.5:7b), (r1:8b, qwen3:8b).
Lineage pairs share lineage and differ in scale (within Group A / within Group B).

Note (real experimental caveat, not needed for CPU validate): `ollama show`
reports r1-distills WITHOUT the `tools` capability (only completion/thinking),
whereas the vanilla bases advertise `tools`. The runner still sends tools; some
r1-distills may ignore them -- that is itself part of the susceptibility signal.

AUTHORING PASS (2026-07-04, round 3): trivial-nit cleanup only -- bound `parsed`
up-front (pyright reportPossiblyUnbound false positive) and dropped a dead
`arch = None` in the except branch. No behavior change. A separate hostile review
gates launch.
"""
from __future__ import annotations

import subprocess
from typing import Any

# arch string (from `ollama show`) -> human family label
ARCH_FAMILY = {
    "qwen2": "Qwen2.5",
    "qwen3": "Qwen3",
    "llama": "Llama",
    "gemma": "Gemma",
    "gemma2": "Gemma2",
    "gemma3": "Gemma3",
    "phi3": "Phi",
    "mistral": "Mistral",
}


def infer_lineage(name: str) -> str:
    n = name.lower()
    if "deepseek-r1" in n or n.startswith("r1") or "-r1" in n or ":r1" in n:
        return "r1-distill"
    return "base-instruct"


def _parse_show(text: str) -> dict[str, Any]:
    """Parse the two-column `ollama show` text block."""
    out: dict[str, Any] = {"architecture": None, "parameters": None,
                           "quantization": None, "capabilities": []}
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low in ("model", "capabilities", "parameters", "system", "license", "projector"):
            section = low
            continue
        parts = line.split()
        if section == "model":
            key = parts[0].lower()
            val = parts[-1]
            if key == "architecture":
                out["architecture"] = val
            elif key == "parameters":
                out["parameters"] = val
            elif key == "quantization":
                out["quantization"] = val
        elif section == "capabilities":
            out["capabilities"].append(parts[0].lower())
    return out


def parse_model(name: str, backend: str = "ollama",
                mock_show: dict[str, str] | None = None) -> dict[str, Any]:
    """Resolve one model to {name, family, architecture, lineage, ...}.

    backend='ollama' runs `ollama show <name>` (subprocess; metadata only).
    backend='mock' uses `mock_show` (name->raw show text) or design defaults.
    """
    raw = None
    # bound up-front so `parsed` is never possibly-unbound (silences a pyright
    # reportPossiblyUnbound false positive; every real path overwrites it below).
    parsed: dict[str, Any] = {"architecture": None, "parameters": None,
                              "quantization": None, "capabilities": []}
    if backend == "mock":
        raw = (mock_show or {}).get(name)
    else:
        # network/subprocess ONLY here.
        try:
            proc = subprocess.run(["ollama", "show", name], capture_output=True,
                                  text=True, timeout=60)
            raw = proc.stdout
        except (OSError, subprocess.SubprocessError) as e:
            raw = None
            parsed = {"architecture": None, "parameters": None,
                      "quantization": None, "capabilities": [], "error": str(e)}
    if raw is not None:
        parsed = _parse_show(raw)
    elif backend == "mock":
        # fall back to the design's known architecture for this name
        d = {m["name"]: m for m in DESIGN}.get(name)
        parsed = {"architecture": (d or {}).get("arch_raw"), "parameters": None,
                  "quantization": None, "capabilities": []}
    arch = parsed.get("architecture")
    family = ARCH_FAMILY.get(arch or "", arch or "unknown")
    return {
        "name": name,
        "architecture": arch,
        "family": family,
        "lineage": infer_lineage(name),
        "parameters": parsed.get("parameters"),
        "quantization": parsed.get("quantization"),
        "capabilities": parsed.get("capabilities", []),
        "supports_tools": "tools" in parsed.get("capabilities", []),
    }


# ----------------------------------------------------------------------------
# Pre-registered design. `match_group` = (arch_family, scale) key used to define
# architecture-matched pairs; `group` = a/b/c experimental cell.
# ----------------------------------------------------------------------------
DESIGN: list[dict[str, Any]] = [
    # Group A -- R1-distills
    {"name": "deepseek-r1:1.5b", "group": "r1", "lineage": "r1-distill",
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "small",
     "match_group": "Qwen2.5/small"},
    {"name": "deepseek-r1:7b", "group": "r1", "lineage": "r1-distill",
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "mid",
     "match_group": "Qwen2.5/mid"},
    {"name": "deepseek-r1:8b", "group": "r1", "lineage": "r1-distill",
     "arch_raw": "qwen3", "arch_family": "Qwen3", "scale": "large",
     "match_group": "Qwen3/large"},
    # Group B -- matched vanilla bases
    {"name": "qwen2.5:1.5b", "group": "base", "lineage": "base-instruct",
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "small",
     "match_group": "Qwen2.5/small"},
    {"name": "qwen2.5:7b-instruct-q8_0", "group": "base", "lineage": "base-instruct",
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "mid",
     "match_group": "Qwen2.5/mid"},
    {"name": "qwen3:8b", "group": "base", "lineage": "base-instruct",
     "arch_raw": "qwen3", "arch_family": "Qwen3", "scale": "large",
     "match_group": "Qwen3/large"},
    # Group C -- out-group
    {"name": "llama3.1:8b-instruct-q4_K_M", "group": "out", "lineage": "base-instruct",
     "arch_raw": "llama", "arch_family": "Llama", "scale": "large",
     "match_group": "Llama/large"},
    {"name": "gemma2:9b-instruct-q8_0", "group": "out", "lineage": "base-instruct",
     "arch_raw": "gemma2", "arch_family": "Gemma2", "scale": "large",
     "match_group": "Gemma2/large"},
    {"name": "mistral:7b-instruct-v0.3-q4_K_M", "group": "out", "lineage": "base-instruct",
     "arch_raw": "llama", "arch_family": "Mistral", "scale": "mid",
     "match_group": "Mistral/mid"},
]


def design_models() -> list[dict[str, Any]]:
    """The pre-registered 9-model panel (metadata dicts, no ollama call)."""
    return [dict(m) for m in DESIGN]


def resolve_models(names: list[str] | None = None, backend: str = "ollama",
                   mock_show: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Resolve a name list to full meta, merged with design labels when known.

    If names is None, resolves the full design panel. For backend='mock' this
    does NOT require ollama at all.
    """
    design_by_name = {m["name"]: m for m in DESIGN}
    names = names or [m["name"] for m in DESIGN]
    out = []
    for nm in names:
        meta = parse_model(nm, backend=backend, mock_show=mock_show)
        d = design_by_name.get(nm)
        if d:
            meta.update({"group": d["group"], "match_group": d["match_group"],
                         "scale": d["scale"], "arch_family": d["arch_family"]})
            # trust design arch_family label for grouping; keep parsed architecture too
        else:
            meta.setdefault("group", "unassigned")
            meta.setdefault("match_group", f"{meta.get('family')}/?")
            meta.setdefault("arch_family", meta.get("family"))
        out.append(meta)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Resolve Ollama models to lineage/arch.")
    ap.add_argument("--backend", choices=["ollama", "mock"], default="mock")
    ap.add_argument("--names", nargs="*", default=None)
    args = ap.parse_args(argv)
    res = resolve_models(args.names, backend=args.backend)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
