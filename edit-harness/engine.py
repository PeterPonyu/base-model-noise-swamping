"""engine.py — self-advancing fan-out engine (overnight breadth campaign).

GPU is serial. This daemon keeps it busy back-to-back:
  1. waits for the in-flight layer sweep (run_sweep.sh) to finish,
  2. runs each round's jobs; a job waits for its model's COMPLETE marker
     (downloads run concurrently) before starting,
  3. skips jobs whose output JSON already exists (idempotent / restartable),
  4. per-job hang watchdog; 2 consecutive *run* failures => stop (likely GPU
     wedge from suspend; needs root nvidia_uvm reload),
  5. tabulates the full breadth table when the plan is exhausted.

Breadth axes: architecture {Llama, Qwen, Gemma, Phi} x scale {0.5B..3.8B}
x editor {rome, ft} x dataset {counterfact, zsre} x layer x seed.
All fp32 (<=3.8B fits 24GB); models fetched by download_models.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

H = os.path.dirname(os.path.abspath(__file__))
PY = "/home/zeyufu/miniconda3/envs/dl/bin/python3"  # direct env python: avoids nested
# conda-run activation (the dl env's gcc activate.d errors and breaks `conda run`)
RES = os.path.join(H, "results")
ENG = os.path.join(H, "engine")
os.makedirs(ENG, exist_ok=True)
LOG = os.path.join(ENG, "engine.log")
STATE = os.path.join(ENG, "state.json")

CF = ("counterfact", "data/counterfact.json")
ZSRE = ("zsre", "data/zsre_eval.json")
N, M = 150, 400
PER_JOB_TIMEOUT = 3000       # 50 min; 3.8B fp32 @150x400 ~ <30 min
MODEL_WAIT = 7200            # wait up to 2h for a model to finish downloading

L1B = "data/models/Llama-3.2-1B"
L3B = "data/models/Llama-3.2-3B"
Q05 = "data/models/Qwen2.5-0.5B"
Q15 = "data/models/Qwen2.5-1.5B"
Q3B = "data/models/Qwen2.5-3B"
G2B = "data/models/gemma-2-2b"
PHI = "data/models/Phi-3.5-mini"


def job(model, editor, ds, layer, tag, seed=0):
    d, data = ds
    return {"model": model, "editor": editor, "dataset": d, "data": data,
            "layer": str(layer), "seed": seed, "tag": tag,
            "out": os.path.join(RES, f"sweep_{tag}.json")}


# Round 1 (Llama-1B layer sweep) is run_sweep.sh. Engine handles rounds 2..8.
ROUNDS = [
    # R2 — editor + dataset on the two already-local models (runs while downloads proceed)
    [job(L1B, "ft",   CF,   8, "llama1b_ft_cf_L8"),
     job(L1B, "rome", ZSRE, 8, "llama1b_rome_zsre_L8"),
     job(Q15, "rome", CF,  14, "qwen1.5b_rome_cf_L14"),
     job(Q15, "ft",   CF,  14, "qwen1.5b_ft_cf_L14"),
     job(Q15, "rome", ZSRE, 14, "qwen1.5b_rome_zsre_L14")],
    # R3 — Llama-3.2-3B (scale within Llama)
    [job(L3B, "rome", CF,   6, "llama3b_rome_cf_L6"),
     job(L3B, "rome", CF,  14, "llama3b_rome_cf_L14"),
     job(L3B, "ft",   CF,  14, "llama3b_ft_cf_L14")],
    # R4 — Qwen2.5-3B (scale within Qwen)
    [job(Q3B, "rome", CF,   9, "qwen3b_rome_cf_L9"),
     job(Q3B, "rome", CF,  18, "qwen3b_rome_cf_L18"),
     job(Q3B, "rome", ZSRE, 18, "qwen3b_rome_zsre_L18")],
    # R5 — Qwen2.5-0.5B (small end of the scale axis)
    [job(Q05, "rome", CF,   6, "qwen0.5b_rome_cf_L6"),
     job(Q05, "rome", CF,  12, "qwen0.5b_rome_cf_L12"),
     job(Q05, "rome", CF,  18, "qwen0.5b_rome_cf_L18")],
    # R6 — Gemma-2-2B (3rd architecture)
    [job(G2B, "rome", CF,   6, "gemma2b_rome_cf_L6"),
     job(G2B, "rome", CF,  13, "gemma2b_rome_cf_L13"),
     job(G2B, "rome", ZSRE, 13, "gemma2b_rome_zsre_L13")],
    # R7 — seed robustness on the headline config (done before the riskiest round)
    [job(L1B, "rome", CF, 8, "llama1b_rome_cf_L8_s1", seed=1),
     job(L1B, "rome", CF, 8, "llama1b_rome_cf_L8_s2", seed=2)],
    # R8 — Phi-3.5-mini (4th architecture; last because newest/riskiest to load)
    [job(PHI, "rome", CF,  8, "phi35_rome_cf_L8"),
     job(PHI, "rome", CF, 16, "phi35_rome_cf_L16")],
]


def logline(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def write_state(**kw):
    json.dump(kw, open(STATE, "w"), indent=2)


def model_ready(model):
    return os.path.exists(os.path.join(H, model, "COMPLETE"))


def wait_for_model(model):
    waited = 0
    while not model_ready(model):
        if waited >= MODEL_WAIT:
            return False
        if waited % 300 == 0:
            logline(f"waiting for model download: {model}")
        time.sleep(60)
        waited += 60
    return True


def run_job(j, attempts=3):
    """Run a job. Returns 'ok' | 'fail' | 'timeout'.
    - transient failures are retried (e.g. GPU still releasing at a handoff),
    - a genuine hang TIMES OUT (not retried) -> 'timeout' (GPU-wide wedge signal),
    - a fast repeated failure -> 'fail' (usually a bad model/config, skip just it).
    Each attempt's stdout/stderr is captured to engine/jobs/<tag>.log."""
    cmd = [PY, "experiments/killgate_keygeom.py",
           "--model", j["model"], "--editor", j["editor"], "--dataset", j["dataset"],
           "--data", j["data"], "--n_edits", str(N), "--n_probes", str(M),
           "--steps", "20" if j["editor"] == "rome" else "25",
           "--lr", "0.1", "--ft_lr", "5e-3",
           "--layer", j["layer"], "--seed", str(j["seed"]), "--out", j["out"]]
    env = dict(os.environ)
    for k in ("ALL_PROXY", "all_proxy"):
        env.pop(k, None)
    env["HF_HUB_OFFLINE"] = "1"
    jobs_dir = os.path.join(ENG, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    jlog = os.path.join(jobs_dir, j["tag"] + ".log")
    for a in range(1, attempts + 1):
        with open(jlog, "a") as lf:
            lf.write(f"\n=== attempt {a}/{attempts} {time.strftime('%F %T')} ===\n")
            lf.flush()
            try:
                r = subprocess.run(cmd, cwd=H, env=env, timeout=PER_JOB_TIMEOUT,
                                   stdout=lf, stderr=subprocess.STDOUT)
            except subprocess.TimeoutExpired:
                lf.write("TIMEOUT — GPU-wide hang/wedge signal (no retry)\n")
                return "timeout"
        if r.returncode == 0 and os.path.exists(j["out"]):
            return "ok"
        logline(f"  attempt {a} failed (rc={r.returncode}) for {j['tag']}; "
                f"{'retrying' if a < attempts else 'giving up'} (see jobs/{j['tag']}.log)")
        if a < attempts:
            time.sleep(25)  # let the GPU settle before retrying
    return "fail"


def wait_for_layer_sweep():
    sweep_log = os.path.join(RES, "sweep.log")
    waited = 0
    while waited < MODEL_WAIT:                       # bounded: a dead sweep must not hang forever
        if os.path.exists(sweep_log) and "SWEEP_LAYERS_DONE" in open(sweep_log).read():
            logline("layer sweep finished — engine taking over GPU")
            return
        if waited % 300 == 0:
            logline("waiting for layer sweep to finish...")
        time.sleep(30)
        waited += 30
    logline(f"WARNING: layer sweep not done after {MODEL_WAIT}s — proceeding anyway "
            "(check run_sweep.sh; round-1 layer results may be missing)")


def main():
    logline("=== fan-out engine start (overnight breadth campaign) ===")
    wait_for_layer_sweep()
    time.sleep(30)  # let the just-finished sweep fully release the GPU before round 2
    total = sum(len(r) for r in ROUNDS)
    done = failed = skipped = 0
    model_fails = {}                   # per-model fast-fail count
    bad_models = set()                 # models flagged bad (skip their remaining jobs)
    for ri, rnd in enumerate(ROUNDS, start=2):
        logline(f"--- ROUND {ri} ({len(rnd)} jobs) ---")
        for j in rnd:
            if os.path.exists(j["out"]):
                skipped += 1
                logline(f"skip (done): {j['tag']}")
                continue
            if j["model"] in bad_models:
                skipped += 1
                logline(f"skip (model flagged bad): {j['tag']} [{j['model']}]")
                continue
            if not wait_for_model(j["model"]):
                skipped += 1
                logline(f"skip (model never ready): {j['tag']} [{j['model']}]")
                continue
            logline(f"run: {j['tag']}")
            write_state(round=ri, current=j["tag"], done=done, failed=failed,
                        skipped=skipped, total=total)
            t0 = time.time()
            status = run_job(j)          # 'ok' | 'fail' | 'timeout'
            dt = time.time() - t0
            if status == "ok":
                done += 1
                logline(f"done: {j['tag']} ({dt:.0f}s)")
            elif status == "timeout":
                # a hang is the GPU-wide wedge signal (suspend etc.) -> stop, needs a reload
                logline("STOP: job TIMED OUT — GPU-wide hang/wedge (suspend?). Fix: "
                        "sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm, then re-run "
                        "engine.py (skips done jobs).")
                write_state(round=ri, current="STOPPED_GPU_WEDGE", done=done,
                            failed=failed, skipped=skipped, total=total)
                return
            else:  # fast fail -> almost always a bad model/config, NOT a GPU wedge
                failed += 1
                model_fails[j["model"]] = model_fails.get(j["model"], 0) + 1
                logline(f"FAIL: {j['tag']} ({dt:.0f}s) [{j['model']} "
                        f"fails={model_fails[j['model']]}]")
                if model_fails[j["model"]] >= 2:
                    bad_models.add(j["model"])
                    logline(f"MODEL FLAGGED BAD: {j['model']} (2 fails) — skipping its "
                            f"remaining jobs, CONTINUING with other models")
        logline(f"--- ROUND {ri} complete ---")
    logline("all rounds complete — tabulating")
    try:
        env = dict(os.environ); env["HF_HUB_OFFLINE"] = "1"
        out = subprocess.run([PY, "experiments/collate.py"], cwd=H, env=env,
                             capture_output=True, text=True, timeout=120).stdout
        with open(os.path.join(ENG, "breadth_table.txt"), "w") as f:
            f.write(out)
        logline("wrote engine/breadth_table.txt")
    except Exception as e:
        logline(f"tabulate failed: {e}")
    write_state(round="DONE", current=None, done=done, failed=failed,
                skipped=skipped, total=total)
    logline(f"=== engine done: {done} ran, {skipped} skipped, {failed} failed"
            + (f", bad_models=[{','.join(sorted(bad_models))}]" if bad_models else "")
            + " ===")


if __name__ == "__main__":
    main()
