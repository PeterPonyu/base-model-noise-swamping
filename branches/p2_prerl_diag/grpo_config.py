#!/usr/bin/env python3
"""
grpo_config.py — matched-budget LoRA GRPO config SCAFFOLD for the P2 post-RL run.

Pure config + data + a rule-based reward function.  NOTHING here imports trl,
unsloth, transformers or torch AT MODULE LOAD, so this file imports cleanly on
CPU in the broken `dl` env.  The only place trl is touched is inside
`build_grpo_trainer()`, whose import is lazy AND which must be executed only
inside the patched `dl-rl` clone (see SETUP.md + trl_mergekit_fix.md).

"Matched-budget" = every checkpoint in the 7-model panel is trained with the
SAME optimization budget (updates x group size x gen length x LoRA capacity) so
that cross-checkpoint differences in the post-GRPO overthinking gap are
attributable to the model, not to unequal compute.  Do not tune these per model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------------- #
# The 7-checkpoint panel (fp16 weights already on disk; 0 download).
# Paths are relative to the repo root; RESOURCES.md wrongly claimed an 8B base.
# ----------------------------------------------------------------------------- #
CHECKPOINT_PANEL: List[str] = [
    "edit-harness/data/models/Qwen2.5-0.5B",
    "edit-harness/data/models/Qwen2.5-1.5B",
    "edit-harness/data/models/Qwen2.5-3B",
    "edit-harness/data/models/Llama-3.2-1B",
    "edit-harness/data/models/Llama-3.2-3B",
    "edit-harness/data/models/gemma-2-2b",
    "edit-harness/data/models/Phi-3.5-mini",
]


@dataclass
class LoRAConfig:
    """LoRA capacity — identical across the panel (matched budget)."""
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    task_type: str = "CAUSAL_LM"


@dataclass
class GenConfig:
    """Rollout / generation budget."""
    num_generations: int = 8          # k=8, matches the pre-RL diagnostic sampling
    max_prompt_length: int = 512
    max_completion_length: int = 640  # cap CoT length; same for all checkpoints
    temperature: float = 0.9
    top_p: float = 1.0


@dataclass
class GRPOScaffold:
    """Matched-budget GRPO hyperparameters (config only — no trl types here)."""
    dataset: str = "openai/gsm8k"      # 'main' split; cached / small fetch
    dataset_config: str = "main"
    train_split: str = "train"
    eval_split: str = "test"

    # optimization budget — held constant across the panel
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_ratio: float = 0.03
    max_steps: int = 500               # matched budget: 500 updates for every ckpt
    per_device_train_batch_size: int = 8   # == prompts per step
    gradient_accumulation_steps: int = 1
    num_generations: int = 8
    beta: float = 0.04                 # KL coeff to the frozen reference
    max_grad_norm: float = 1.0
    seed: int = 0

    # boxed-answer rule reward
    reward_correct: float = 1.0
    reward_format_bonus: float = 0.1   # small bonus for producing a \boxed{...}
    reward_wrong: float = 0.0

    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 250
    output_dir: str = "branches/p2_prerl_diag/grpo_out"

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    gen: GenConfig = field(default_factory=GenConfig)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------- #
# Boxed-answer rule reward function (signature + pure implementation).
# GRPO reward fns receive `completions` (+ any dataset columns as kwargs, e.g.
# `answer`) and return a list[float] of per-sample rewards.  No trl import
# needed — this is a plain callable trl will invoke.
# ----------------------------------------------------------------------------- #

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_boxed(text: str) -> Optional[str]:
    """Return the content of the LAST \\boxed{...} in `text`, else None."""
    matches = _BOXED_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def _normalize_number(s: Optional[str]) -> Optional[str]:
    """Normalize a numeric string for comparison: strip commas/$/%/space, drop a
    trailing '.0'.  Returns None if no number found."""
    if s is None:
        return None
    m = _NUM_RE.search(s.replace(",", ""))
    if not m:
        return None
    v = m.group(0)
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else repr(f)
    except ValueError:
        return None


def gsm8k_gold(answer_field: str) -> Optional[str]:
    """GSM8K gold answers look like '... #### 18'.  Extract the final number."""
    if answer_field is None:
        return None
    tail = answer_field.split("####")[-1]
    return _normalize_number(tail)


def boxed_reward(
    completions: List[Any],
    answer: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """Rule reward for GRPO.

    Signature matches trl's reward-function protocol: `completions` plus dataset
    columns forwarded as kwargs (here `answer`, GSM8K's gold column).  Each
    completion is a chat-style list[{"role","content"}] OR a raw string.

    Reward = reward_correct if the boxed final answer matches the gold number,
    else reward_wrong; plus reward_format_bonus if a \\boxed{...} is present at
    all (shapes the format without paying for correctness)."""
    cfg = GRPOScaffold()
    golds = answer if answer is not None else [None] * len(completions)
    rewards: List[float] = []
    for comp, gold in zip(completions, golds):
        text = _completion_text(comp)
        boxed = extract_boxed(text)
        r = cfg.reward_format_bonus if boxed is not None else 0.0
        if boxed is not None and gsm8k_gold(gold) is not None:
            if _normalize_number(boxed) == gsm8k_gold(gold):
                r += cfg.reward_correct
            else:
                r += cfg.reward_wrong
        rewards.append(float(r))
    return rewards


def _completion_text(comp: Any) -> str:
    """Accept either a raw string or a chat list [{'role','content'}, ...]."""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp:
        last = comp[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(comp)


# ----------------------------------------------------------------------------- #
# Lazy trainer builder — IMPORTS trl ONLY WHEN CALLED (never at module load).
# Do NOT call this in the shared `dl` env; run only inside patched `dl-rl`.
# ----------------------------------------------------------------------------- #

def build_grpo_trainer(model_path: str, cfg: Optional[GRPOScaffold] = None):
    """Construct a trl GRPOTrainer for one checkpoint.  trl / peft imports happen
    HERE, lazily, so importing grpo_config.py stays dependency-free on CPU.

    Guarded so an accidental call in the broken env fails loudly with guidance
    instead of a cryptic mergekit traceback."""
    cfg = cfg or GRPOScaffold()
    try:
        import transformers                       # noqa: F401  (lazy)
        from trl import GRPOConfig, GRPOTrainer  # noqa: F401  (lazy)
        from peft import LoraConfig              # noqa: F401  (lazy)
    except Exception as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "trl/peft failed to import. Run this ONLY inside the patched `dl-rl` "
            "clone (see SETUP.md + trl_mergekit_fix.md); never in shared `dl`. "
            f"Underlying error: {e}"
        ) from e

    # COMPAT SHIM (dl-rl: trl 0.24.0 + transformers 5.12.1): every trl trainer's
    # __init__ (GRPO/DPO/KTO/ORPO/CPO/BCO/RLOO/online-DPO) unconditionally does
    # `model.warnings_issued["estimate_tokens"] = True` to suppress a duplicate
    # token-estimation warning. That attribute used to be set in
    # `PreTrainedModel.__init__`; transformers 5.12.1 removed it entirely (grep
    # confirms zero occurrences in the installed package), so GRPOTrainer crashes
    # in __init__ before training starts, for every model transformers builds
    # (PEFT-wrapped or not). Patch `PreTrainedModel.__init__` to restore the dict
    # if absent; idempotent via the sentinel so re-entry (e.g. multiple checkpoints
    # in one process) is a no-op.
    if not hasattr(transformers.PreTrainedModel, "_p2_warnings_issued_shim"):
        _orig_pretrained_init = transformers.PreTrainedModel.__init__

        def _patched_pretrained_init(self, *a, **kw):
            _orig_pretrained_init(self, *a, **kw)
            if not hasattr(self, "warnings_issued"):
                self.warnings_issued = {}

        transformers.PreTrainedModel.__init__ = _patched_pretrained_init
        transformers.PreTrainedModel._p2_warnings_issued_shim = True

    peft_cfg = LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout, bias=cfg.lora.bias,
        target_modules=cfg.lora.target_modules, task_type=cfg.lora.task_type,
    )
    grpo_cfg = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.gen.num_generations,
        max_prompt_length=cfg.gen.max_prompt_length,
        max_completion_length=cfg.gen.max_completion_length,
        temperature=cfg.gen.temperature,
        top_p=cfg.gen.top_p,
        beta=cfg.beta,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        seed=cfg.seed,
        # P2 DEVIATION D2 (logged in train_status.json): load weights in bf16,
        # UNIFORM across the panel — fp32 load of the 3-3.8B rows cannot fit 24GB
        # beside k=8 rollout activations.  Matched budget unaffected (identical
        # setting for every checkpoint).  With peft_config, trl uses the
        # adapter-disabled base as the KL reference (no second model copy).
        # Key MUST be "dtype": trl 0.24 reads model_init_kwargs.get("dtype")
        # (grpo_trainer.py:240) and normalizes the string itself; "torch_dtype"
        # would only work through a deprecated transformers alias (review M1).
        model_init_kwargs={"dtype": "bfloat16"},
    )
    # dataset loading also deferred to call time (transformers/datasets are heavy)
    from datasets import load_dataset  # noqa: E402  (lazy)
    ds = load_dataset(cfg.dataset, cfg.dataset_config, split=cfg.train_split)
    # P2 DEVIATION D1 (logged in train_status.json): GRPOTrainer requires a
    # "prompt" column; raw GSM8K carries only question/answer.  Build the prompt
    # with the SAME template the pre-RL and post-RL sampler uses, so training,
    # pre-RL sampling and post-RL sampling all see identical prompting.  The
    # "answer" column is kept — trl forwards it to boxed_reward as a kwarg.
    from sample_ckpt import PROMPT_TEMPLATE  # noqa: E402  (light: no torch at module level)
    ds = ds.map(lambda ex: {"prompt": PROMPT_TEMPLATE.format(question=ex["question"])})

    return GRPOTrainer(
        model=model_path,
        reward_funcs=[boxed_reward],
        args=grpo_cfg,
        train_dataset=ds,
        peft_config=peft_cfg,
    )


if __name__ == "__main__":
    import json
    print(json.dumps(GRPOScaffold().as_dict(), indent=2, default=str))
    # smoke-test the reward fn on toy data (no trl needed)
    demo = boxed_reward(
        ["The answer is \\boxed{18}.", "I think it is \\boxed{7}.", "no box here"],
        answer=["... #### 18", "... #### 42", "... #### 3"],
    )
    print("boxed_reward demo:", demo)
