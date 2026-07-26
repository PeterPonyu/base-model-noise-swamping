#!/usr/bin/env python3
"""Generate synthetic samples JSONs for CPU validation of diagnostic.py.

  --mode bias    : wrong traces ~1.6x longer than right (planted length bias) -> D>1
  --mode control : right/wrong lengths drawn from the SAME dist              -> D~=1

numpy/stdlib only.  Difficulty is also planted (a per-problem base length) so the
difficulty-controlled D_within is a real test, not trivially satisfied.
"""
import argparse, json
import numpy as np


def build(mode: str, n_problems: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    problems = []
    for i in range(n_problems):
        base = rng.uniform(80, 320)          # per-problem difficulty -> base length
        p_correct = rng.uniform(0.2, 0.8)    # per-problem accuracy
        samples = []
        for _ in range(k):
            correct = bool(rng.random() < p_correct)
            if mode == "bias":
                # wrong traces are systematically longer (overthinking bias)
                mult = 1.0 if correct else 1.6
            else:  # control: length independent of correctness
                mult = 1.0
            length = max(1.0, rng.normal(base * mult, 0.15 * base))
            samples.append({"len": round(float(length), 1), "correct": correct})
        problems.append({"problem": f"gsm8k/synth/{i}", "samples": samples})
    return {"checkpoint": f"SYNTH-{mode}", "problems": problems}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bias", "control"], required=True)
    ap.add_argument("--n-problems", type=int, default=150)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w") as fh:
        json.dump(build(a.mode, a.n_problems, a.k, a.seed), fh, indent=2)
    print(f"wrote {a.out}  (mode={a.mode}, n_problems={a.n_problems}, k={a.k})")
