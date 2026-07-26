"""runner.py — load model -> apply edit -> run metrics -> write JSON.

Usage
-----
    conda run -n dl python3 runner.py <config.json>
    conda run -n dl python3 runner.py --smoke          # built-in tiny-Llama demo

Config schema (dict / JSON)
---------------------------
{
  "editor": "ft_editor" | "rome_native",
  "model":  "<hf model id>",          # default tiny-random-Llama for smoke
  "dtype":  "float32" | "bfloat16",   # optional; default float32 (grad-safe)
  "device": "auto" | "cpu" | "cuda",  # optional; default auto
  "editor_config": { ... },            # passed verbatim to the editor
  "edit_request": {
      "subject": "France",
      "prompt":  "The capital of France is",
      "target_new":  "London",
      "target_true": "Paris",                 # optional
      "paraphrase_prompts":   ["France's capital city is", ...],   # optional
      "neighborhood_prompts": ["The capital of Italy is", ...]     # optional
  }
}

Restorability: the model is freshly loaded per run, and (when the same process
runs multiple configs) we snapshot the original state_dict before editing and
restore it afterwards, so edits never bleed across runs.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import sys
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics  # noqa: E402

DEFAULT_MODEL = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
RESULTS_DIR = os.path.join(HERE, "results")
EDITORS_DIR = os.path.join(HERE, "editors")


def _load_editor(name: str):
    """Import the editor module by file path (no package install needed)."""
    path = os.path.join(EDITORS_DIR, f"{name}.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"editor module not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "apply_edit"):
        raise AttributeError(f"editor {name} has no apply_edit()")
    return mod


def _resolve_device(want: str) -> str:
    if want == "cpu":
        return "cpu"
    if want == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(name: str):
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        name, torch.float32
    )


def run(config: Dict) -> Dict:
    editor_name = config["editor"]
    model_id = config.get("model", DEFAULT_MODEL)
    device = _resolve_device(config.get("device", "auto"))
    dtype = _resolve_dtype(config.get("dtype", "float32"))
    editor_config = config.get("editor_config", {})
    req = config["edit_request"]

    prompt = req["prompt"]
    target_new = req["target_new"]
    target_true = req.get("target_true")
    paraphrases = req.get("paraphrase_prompts", []) or []
    neighborhood = req.get("neighborhood_prompts", []) or []

    # --- load model + tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()

    # snapshot for restorability (clone every tensor onto CPU)
    original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # --- PRE-edit measurements ---
    pre_efficacy = metrics.efficacy(model, tokenizer, prompt, target_new, target_true, device)
    pre_neighbor_argmax = metrics.argmax_tokens(model, tokenizer, neighborhood, device)
    pre_fluency = metrics.fluency(model, tokenizer, prompt, max_new_tokens=20, device=device)

    # --- APPLY edit ---
    editor = _load_editor(editor_name)
    edit_info = editor.apply_edit(model, tokenizer, req, editor_config, device)

    # --- POST-edit measurements ---
    post_efficacy = metrics.efficacy(model, tokenizer, prompt, target_new, target_true, device)
    post_generalization = metrics.generalization(model, tokenizer, paraphrases, target_new, device)
    post_neighbor_argmax = metrics.argmax_tokens(model, tokenizer, neighborhood, device)
    locality = metrics.locality_score(pre_neighbor_argmax, post_neighbor_argmax)
    post_fluency = metrics.fluency(model, tokenizer, prompt, max_new_tokens=20, device=device)

    # --- restore model so this process can run further edits cleanly ---
    model.load_state_dict(original_state)

    run_id = "{editor}_{model}_{ts}".format(
        editor=editor_name,
        model=model_id.split("/")[-1],
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )

    result = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "editor": editor_name,
            "model": model_id,
            "device": device,
            "dtype": str(dtype),
            "editor_config": editor_config,
        },
        "edit_request": req,
        "edit_info": edit_info,
        "metrics": {
            "efficacy": {
                "pre": pre_efficacy,
                "post": post_efficacy,
                "improved": bool(post_efficacy["success"] > pre_efficacy["success"]
                                 or post_efficacy["p_new"] > pre_efficacy["p_new"]),
            },
            "generalization": post_generalization,
            "locality": locality,
            "fluency": {
                "pre": pre_fluency,
                "post": post_fluency,
            },
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    result["_path"] = out_path
    return result


def _smoke_config() -> Dict:
    return {
        "editor": "ft_editor",
        "model": DEFAULT_MODEL,
        "dtype": "float32",
        "device": "cpu",
        "editor_config": {"layers": [1], "steps": 150, "lr": 0.05, "lambda_l2": 1e-3, "lambda_kl": 0.05},
        "edit_request": {
            "subject": "France",
            "prompt": "The capital of France is",
            # ' user' is NOT the model's pre-edit argmax but is reachable, so the smoke
            # test demonstrates a genuine flip: pre.success=0 -> post.success=1, with
            # P(new) > P(true). Use a real fact (e.g. 'London') on a real model.
            "target_new": "user",
            "target_true": "Paris",
            "paraphrase_prompts": [
                "France's capital city is",
                "The capital city of France is called",
            ],
            "neighborhood_prompts": [
                "The capital of Italy is",
                "The capital of Spain is",
                "The sky is",
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", help="path to a JSON config file")
    ap.add_argument("--smoke", action="store_true", help="run the built-in tiny-Llama smoke config")
    args = ap.parse_args()

    if args.smoke or not args.config:
        cfg = _smoke_config()
    else:
        with open(args.config) as f:
            cfg = json.load(f)

    result = run(cfg)
    # concise stdout summary for the queue runner / logs
    m = result["metrics"]
    print(f"[runner] wrote {result['_path']}")
    print(f"[runner] editor={result['config']['editor']} model={result['config']['model']}")
    print(f"[runner] efficacy pre.success={m['efficacy']['pre']['success']} "
          f"post.success={m['efficacy']['post']['success']} "
          f"pre.p_new={m['efficacy']['pre']['p_new']:.4g} post.p_new={m['efficacy']['post']['p_new']:.4g}")
    print(f"[runner] generalization={m['generalization']['generalization']:.3g} "
          f"locality={m['locality']['locality']:.3g} "
          f"fluency.post={m['fluency']['post']['fluency_ngram_entropy']:.3g}")


if __name__ == "__main__":
    main()
