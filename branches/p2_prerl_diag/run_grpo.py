#!/usr/bin/env python3
"""
run_grpo.py — single-checkpoint GRPO training + LoRA merge (dl-rl ONLY).

The per-checkpoint work item of the PREREG-P2-GRPO-20260710.md validation wave,
orchestrated serially by run_p2_grpo.sh.  One invocation = train one checkpoint
under the frozen matched budget (grpo_config.GRPOScaffold — §3, do not tune per
model), then materialize the post-RL model as a FULL merged checkpoint:

    grpo_out/<ckpt_id>/train/          trainer output (transformers checkpoints)
    grpo_out/<ckpt_id>/adapter/        the LoRA adapter alone (provenance)
    grpo_out/<ckpt_id>/merged/         merge_and_unload() full checkpoint + tokenizer
                                       -> post-RL sampling reuses sample_ckpt.py
                                          UNCHANGED via --model .../merged
    grpo_out/<ckpt_id>/train_status.json   terminal status (the downstream contract)

train_status.json statuses:
    completed      — trained, merged, ready for post-RL sampling
    diverged       — prereg §7 exclusion: NaN/Inf loss, or trailing-5 mean reward
                     <= 0.02 after >= 20 reward logs (~200/500 steps)
    failed_oom     — CUDA OOM.  Prereg §3 forbids per-model tuning, so we NEVER
                     shrink the batch to fit; the checkpoint is excluded and logged.
    killed_timeout — SIGTERM received (driver's per-job cap)
    failed         — any other exception

Exit codes: 0 ok/skip, 1 failed, 3 diverged, 4 oom, 5 timeout-kill, 2 usage/env.

Env guard: real runs REQUIRE the patched dl-rl clone (see trl_mergekit_fix.md §4b).
--smoke-steps N redirects everything to grpo_out_smoke/ (never read by the science
pipeline) for an end-to-end wiring test with a tiny step count.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from grpo_config import CHECKPOINT_PANEL, GRPOScaffold, build_grpo_trainer  # noqa: E402

# deviations from the frozen prereg text, stamped into every train_status.json
DEVIATIONS = [
    "D1 build_grpo_trainer: GSM8K prompt-column mapping added "
    "(template = sample_ckpt.PROMPT_TEMPLATE; identical prompting pre/train/post)",
    "D2 build_grpo_trainer: model_init_kwargs torch_dtype=bfloat16, uniform across "
    "the panel (fp32 load of the 3-3.8B rows cannot fit 24GB beside k=8 rollouts)",
    "NOTE trl 0.24 counts per_device_train_batch_size in COMPLETIONS: 8 = 1 prompt "
    "x 8 generations per step (scaffold comment 'prompts per step' is stale); "
    "budget stays frozen as configured, matched across the panel",
]


class DivergenceError(RuntimeError):
    def __init__(self, reason: str, step: int):
        super().__init__(f"{reason} at step {step}")
        self.reason = reason
        self.step = step


class TimeoutKill(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def _resolve_ckpt(ckpt_id: str) -> str:
    for p in CHECKPOINT_PANEL:
        if os.path.basename(p.rstrip("/")) == ckpt_id:
            return p
    raise SystemExit(f"run_grpo: --checkpoint {ckpt_id!r} not in CHECKPOINT_PANEL "
                     f"({[os.path.basename(p) for p in CHECKPOINT_PANEL]})")


def _in_dlrl() -> bool:
    return "/envs/dl-rl/" in (sys.prefix + "/") or \
        os.environ.get("CONDA_DEFAULT_ENV") == "dl-rl"


def build_divergence_monitor():
    """TrainerCallback: abort on NaN/Inf loss immediately; abort 'reward_collapse'
    if the trailing-5 mean reward <= 0.02 after >= 20 reward log events
    (logging_steps=10 -> ~200 of 500 steps of runway before judging collapse).
    Raising inside on_log propagates out of trainer.train()."""
    from transformers import TrainerCallback

    class DivergenceMonitor(TrainerCallback):
        REWARD_FLOOR = 0.02
        MIN_REWARD_LOGS = 20
        WINDOW = 5

        def __init__(self):
            self.rewards: list[float] = []
            self.last_reward_mean: float | None = None

        @staticmethod
        def _extract_reward(logs: dict) -> float | None:
            v = logs.get("reward")
            if isinstance(v, (int, float)):
                return float(v)
            means = [float(x) for k, x in logs.items()
                     if k.startswith("rewards/") and k.endswith("/mean")
                     and isinstance(x, (int, float))]
            return sum(means) / len(means) if means else None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            loss = logs.get("loss")
            if isinstance(loss, (int, float)) and not math.isfinite(loss):
                raise DivergenceError("nan_loss", state.global_step)
            r = self._extract_reward(logs)
            if r is not None:
                if not math.isfinite(r):
                    raise DivergenceError("nan_reward", state.global_step)
                self.rewards.append(r)
                tail = self.rewards[-self.WINDOW:]
                self.last_reward_mean = sum(tail) / len(tail)
                if (len(self.rewards) >= self.MIN_REWARD_LOGS
                        and self.last_reward_mean <= self.REWARD_FLOOR):
                    raise DivergenceError("reward_collapse", state.global_step)

    return DivergenceMonitor()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint id (basename in CHECKPOINT_PANEL), e.g. Qwen2.5-0.5B")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve + verify trl imports; no model load, no training")
    ap.add_argument("--smoke-steps", type=int, default=None,
                    help="NON-SCIENCE wiring test: override max_steps, write to "
                         "grpo_out_smoke/ (never read by the science pipeline)")
    args = ap.parse_args(argv)

    cid = args.checkpoint
    model_path = _resolve_ckpt(cid)
    smoke = args.smoke_steps is not None
    root = os.path.join(HERE, "grpo_out_smoke" if smoke else "grpo_out", cid)
    train_dir = os.path.join(root, "train")
    adapter_dir = os.path.join(root, "adapter")
    merged_dir = os.path.join(root, "merged")
    status_path = os.path.join(root, "train_status.json")

    if args.dry_run:
        try:
            from trl import GRPOTrainer  # noqa: F401 — proves the mergekit patch
        except Exception as e:
            print(f"run_grpo DRY-RUN FAIL: trl import broken: {e}", file=sys.stderr)
            return 2
        cfg = GRPOScaffold(output_dir=train_dir)
        print(json.dumps({"dry_run": True, "checkpoint": cid, "model_path": model_path,
                          "output_root": root, "smoke": smoke,
                          "max_steps": args.smoke_steps or cfg.max_steps,
                          "trl_import": "OK", "deviations": DEVIATIONS}, indent=2))
        return 0

    if not _in_dlrl():
        print("run_grpo: REFUSING — real training must run inside the patched dl-rl "
              "clone (conda run -n dl-rl ...); see SETUP.md + trl_mergekit_fix.md",
              file=sys.stderr)
        return 2

    # idempotency: completed + merged checkpoint on disk -> skip
    if os.path.exists(status_path):
        try:
            with open(status_path) as fh:
                prev = json.load(fh)
        except (json.JSONDecodeError, OSError):
            prev = {}
        if prev.get("status") == "completed" \
                and os.path.exists(os.path.join(merged_dir, "config.json")):
            print(f"run_grpo: skip {cid} (already completed; merged checkpoint exists)")
            return 0

    status = {
        "checkpoint_id": cid, "model_path": model_path, "smoke": smoke,
        "status": "running", "reason": None, "pid": os.getpid(),
        "started_at": _now(), "ended_at": None, "wall_seconds": None,
        "steps_completed": 0, "final_reward_mean_trailing5": None,
        "adapter_dir": None, "merged_dir": None, "deviations": DEVIATIONS,
        "max_steps_effective": None,
    }
    _atomic_write_json(status_path, status)

    def _finish(st: str, reason: str | None, rc: int, monitor=None, trainer=None) -> int:
        status["status"] = st
        status["reason"] = reason
        status["ended_at"] = _now()
        status["wall_seconds"] = round(time.time() - t0, 1)
        if trainer is not None:
            try:
                status["steps_completed"] = int(trainer.state.global_step)
            except Exception:
                pass
        if monitor is not None and monitor.last_reward_mean is not None:
            status["final_reward_mean_trailing5"] = monitor.last_reward_mean
        _atomic_write_json(status_path, status)
        print(f"run_grpo: {cid} -> {st}" + (f" ({reason})" if reason else ""))
        return rc

    # the driver kills with SIGTERM at its per-job cap: record a clean terminal state
    def _on_term(signum, frame):
        raise TimeoutKill()
    signal.signal(signal.SIGTERM, _on_term)

    t0 = time.time()
    monitor = None
    trainer = None
    try:
        cfg = GRPOScaffold(output_dir=train_dir)
        if smoke:
            cfg.max_steps = int(args.smoke_steps)
            cfg.save_steps = 10_000_000  # no mid-run checkpoints in smoke
        status["max_steps_effective"] = cfg.max_steps
        _atomic_write_json(status_path, status)

        print(f"run_grpo: building trainer for {cid} "
              f"(max_steps={cfg.max_steps}, smoke={smoke})")
        trainer = build_grpo_trainer(model_path, cfg)
        monitor = build_divergence_monitor()
        trainer.add_callback(monitor)

        trainer.train()

        # provenance: adapter alone, then the merged full checkpoint
        print(f"run_grpo: saving adapter -> {adapter_dir}")
        trainer.model.save_pretrained(adapter_dir)
        status["adapter_dir"] = adapter_dir

        print(f"run_grpo: merging LoRA -> {merged_dir}")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.save_pretrained(merged_dir)
        status["merged_dir"] = merged_dir

        return _finish("completed", None, 0, monitor, trainer)

    except DivergenceError as e:
        return _finish("diverged", f"{e.reason} (step {e.step})", 3, monitor, trainer)
    except TimeoutKill:
        return _finish("killed_timeout", "SIGTERM (driver per-job cap)", 5,
                       monitor, trainer)
    except Exception as e:  # noqa: BLE001 — terminal-status contract needs breadth
        is_oom = False
        try:
            import torch
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
        except Exception:
            pass
        if is_oom or "out of memory" in str(e).lower():
            return _finish("failed_oom",
                           "CUDA OOM — prereg §3 forbids per-model batch tuning; "
                           "checkpoint excluded and logged", 4, monitor, trainer)
        import traceback
        traceback.print_exc()
        return _finish("failed", f"{type(e).__name__}: {e}", 1, monitor, trainer)


if __name__ == "__main__":
    raise SystemExit(main())
