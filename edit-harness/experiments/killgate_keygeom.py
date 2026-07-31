"""killgate_keygeom.py — First-paper kill-gate experiment.

Hypothesis (H1): the pre-edit cosine between an edit's subject-key k_edit and a
probe fact's key k_probe (same edited MLP layer) predicts how much that ROME
edit damages the probe (drop in P / logit of the probe's correct object token).

Kill-gate: ABANDON if |Spearman(cosine, damage)| < 0.2 AND broken-fact AUROC <
0.6 AND key-overlap does not beat the ENCORE-style norm-growth baseline.

Design note: weights are restored after every edit, so ALL keys are base-model
keys → capture every k_edit and k_probe ONCE, build the full N×M cosine matrix
upfront, then loop edits only to measure per-probe damage.

Usage:
  python experiments/killgate_keygeom.py --model <id> --data data/counterfact.json \
        --n_edits 200 --n_probes 500 --layer auto --out results/killgate.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)
from metrics import (  # noqa: E402
    next_token_logits, first_target_token_id, efficacy, assert_targets_distinguishable,
)
from editors.rome_native import (  # noqa: E402
    _capture_key, find_subject_last_token_index,
)
from editors.arch_compat import normalize_arch  # noqa: E402  (GPT-2 load-time normalization)


def _runner_stamp(start_time):
    with open(__file__, "rb") as f:
        code_sha256 = hashlib.sha256(f.read()).hexdigest()
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "unavailable"
    end_time = time.time()
    return {
        "stamp_version": "runner_stamp.v1",
        "code_sha256": code_sha256,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "wall_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "wall_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time)),
        "elapsed_s": round(end_time - start_time, 3),
        "nvidia_smi_sample": gpu,
    }


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ar = a.argsort().argsort().astype(float)
    br = b.argsort().argsort().astype(float)
    if ar.std() == 0 or br.std() == 0:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def auroc(scores, labels):
    """AUROC of `scores` predicting binary `labels` (1=positive). Rank-based, no sklearn."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks); ranks = (sums / cnt)[inv]
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_counterfact(path, n_edits, n_probes, seed=0, n_holdout=0):
    data = json.load(open(path))
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    recs = []
    for d in data:
        rr = d.get("requested_rewrite", d)
        try:
            subj = rr["subject"]
            prompt = rr["prompt"].format(subj) if "{}" in rr["prompt"] else rr["prompt"]
            tnew = rr["target_new"]["str"] if isinstance(rr["target_new"], dict) else rr["target_new"]
            ttrue = rr["target_true"]["str"] if isinstance(rr["target_true"], dict) else rr["target_true"]
        except Exception:
            continue
        recs.append({"subject": subj, "prompt": prompt, "target_new": tnew, "target_true": ttrue})
        if len(recs) >= n_edits + n_probes + n_holdout:
            break
    edits = recs[:n_edits]
    probes = recs[n_edits:n_edits + n_probes]
    holdout = recs[n_edits + n_probes:n_edits + n_probes + n_holdout]
    return edits, probes, holdout


def load_zsre(path, n_edits, n_probes, seed=0, n_holdout=0):
    """zsRE editing format: src=prompt, subject, alt=target_new, pred=target_true."""
    data = json.load(open(path))
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    recs = []
    for d in data:
        s, p, alt, pred = d.get("subject"), d.get("src"), d.get("alt"), d.get("pred")
        if not (s and p and alt and pred):
            continue
        recs.append({"subject": s, "prompt": p, "target_new": alt, "target_true": pred})
        if len(recs) >= n_edits + n_probes + n_holdout:
            break
    edits = recs[:n_edits]
    probes = recs[n_edits:n_edits + n_probes]
    holdout = recs[n_edits + n_probes:n_edits + n_probes + n_holdout]
    return edits, probes, holdout


def load_mquake(path, n_edits, n_probes, seed=0, n_holdout=0):
    """MQuAKE-CF-3k format (princeton-nlp/MQuAKE, file MQuAKE-CF-3k.json): each record's
    ``requested_rewrite`` is a LIST of CounterFact-schema rewrite dicts forming a multi-hop
    edit chain (length 1-4). We use ONLY rewrite[0] as a single-hop edit — same
    subject/prompt/target_new/target_true keys as load_counterfact, so the COS-matrix key
    capture and within-probe damage metric are unchanged. Probes are built the same way from
    OTHER records' first rewrites (matches the CF probe construction).

    Each record additionally carries the multi-hop fields under mh_-prefixed keys
    (mh_questions/mh_answer_new/mh_answer_true) and n_rewrites (chain length), so a later
    multi-hop analysis can use them without another loader change; today's law-replication
    driver ignores them.
    """
    data = json.load(open(path))
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    recs = []
    for d in data:
        rrs = d.get("requested_rewrite") or []
        if not rrs:
            continue
        rr = rrs[0]
        try:
            subj = rr["subject"]
            prompt = rr["prompt"].format(subj) if "{}" in rr["prompt"] else rr["prompt"]
            tnew = rr["target_new"]["str"] if isinstance(rr["target_new"], dict) else rr["target_new"]
            ttrue = rr["target_true"]["str"] if isinstance(rr["target_true"], dict) else rr["target_true"]
        except Exception:
            continue
        recs.append({
            "subject": subj, "prompt": prompt, "target_new": tnew, "target_true": ttrue,
            "n_rewrites": len(rrs),
            "mh_questions": list(d.get("questions") or []),
            "mh_answer_new": d.get("new_answer"),
            "mh_answer_true": d.get("answer"),
        })
        if len(recs) >= n_edits + n_probes + n_holdout:
            break
    edits = recs[:n_edits]
    probes = recs[n_edits:n_edits + n_probes]
    holdout = recs[n_edits + n_probes:n_edits + n_probes + n_holdout]
    return edits, probes, holdout


@torch.no_grad()
def prob_of_token(model, tok, prompt, token_id, device):
    logits = next_token_logits(model, tok, prompt, device)  # [V] cpu float
    probs = torch.softmax(logits, dim=-1)
    return float(probs[token_id].item()), float(logits[token_id].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_edits", type=int, default=200)
    ap.add_argument("--n_probes", type=int, default=500)
    ap.add_argument("--layer", default="auto")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--editor", choices=["rome", "ft", "alpha", "memit", "grace"], default="rome")
    ap.add_argument("--dataset", choices=["counterfact", "zsre", "mquake"], default="counterfact",
                    help="mquake expects MQuAKE-CF-3k.json (princeton-nlp/MQuAKE); pass its path "
                         "via --data (this harness's convention: data/mquake_cf3k.json)")
    ap.add_argument("--ft_lr", type=float, default=5e-3)
    ap.add_argument("--keep_ratio", type=float, default=0.99,
                    help="AlphaEdit: fraction of preserved-key energy to project out")
    ap.add_argument("--alpha_proj_source", choices=["probes", "holdout", "generic"],
                    default="probes",
                    help="AlphaEdit projector fit set. 'probes'=fit on the SAME probe bank "
                         "damage is measured on (by-construction; reference only). "
                         "'holdout'=fit on a DISJOINT bank of held-out facts (the honest "
                         "causal test). 'generic'=fit on generic non-subject token keys "
                         "from the held-out prompts (distribution-shifted control).")
    ap.add_argument("--holdout_frac", type=float, default=1.0,
                    help="size of the held-out projector-fit bank as a multiple of n_probes "
                         "(only used when --alpha_proj_source in {holdout,generic})")
    ap.add_argument("--ft_kl", type=float, default=0.0,
                    help="D1 control: FT-L KL-locality weight (>0 => regularized FT)")
    ap.add_argument("--ft_kl_n", type=int, default=5,
                    help="number of neighbor prompts used as the FT KL anchor")
    ap.add_argument("--out", default=os.path.join(HARNESS, "results", "killgate.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_matrices", action="store_true",
                    help="dump raw per-pair COS/damage matrices (.npz) for the "
                         "partialled-correlation GATE (within-probe Spearman + permutation null)")
    ap.add_argument("--matrix_dir", default=os.path.join(HARNESS, "results", "matrices"))
    # ---- infra: CPU smokes while the GPU is busy (default keeps every driver unchanged) ----
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="cuda (default; byte-identical to the old hardcode) or cpu (tiny smokes)")
    ap.add_argument("--model_dtype", choices=["fp32", "bf16"], default="fp32",
                    help="model LOAD dtype. fp32 = default, byte-identical to the old hardcode. "
                         "bf16 halves weight RAM for big-model cells; the value-optimization "
                         "math (Adam, log_softmax, the v parameter) stays fp32 inside the "
                         "editors (editors/rome_native.py) — only the frozen forward runs bf16.")
    ap.add_argument("--device_map", choices=["none", "auto", "balanced", "balanced_low_0",
                                             "sequential"], default="none",
                    help="accelerate device_map for TENSOR-PARALLEL sharded loading across "
                         "multiple GPUs (e.g. NeoX-20B on 2x24GB — see run_neox20b.sh). "
                         "'none' (default) is byte-identical to the old single-.to(device) "
                         "load path. Any other value hands loading to accelerate/transformers "
                         "and SKIPS the .to(device) call (which would otherwise collapse the "
                         "shard placement back onto one card); the killgate's `device` variable "
                         "is then re-resolved to the model's INPUT-embedding device (accelerate's "
                         "dispatch hooks move activations across shards downstream, so every "
                         "existing enc.to(device) call site in this file/metrics.py/the editors "
                         "stays correct unchanged — only the per-EDITED-LAYER math, which can "
                         "land on a different card, needed fixing; see tp_edit_util.py). "
                         "Requires --device cuda and the `accelerate` package.")
    # ---- MEMIT (multi-layer whitened spread) — editors/memit.py ----
    ap.add_argument("--memit_layers", default="auto",
                    help="comma ints ('9,10,11,12') or 'auto' = span of 4 ending at --layer. "
                         "HARD requirement: max(memit_layers) == --layer (the z-layer, where "
                         "the precomputed COS geometry lives).")
    ap.add_argument("--memit_cov_source", choices=["generic", "probes", "identity"],
                    default="generic",
                    help="generic = C_l fit on ALL token positions of the DISJOINT holdout "
                         "prompt bank (NOTE: different from alpha's 'generic', which uses "
                         "random non-subject token KEYS); probes = by-construction reference; "
                         "identity = multi-layer-spread-only ablation (C=I).")
    ap.add_argument("--memit_cov_tokens", type=int, default=20000,
                    help="token cap for covariance accumulation")
    ap.add_argument("--memit_cov_reg", type=float, default=1e-2,
                    help="ridge as fraction of mean diag: A = C_hat + reg*mean(diag)*I")
    # ---- GRACE (codebook key-value memory, ΔW == 0) — editors/grace_editor.py ----
    ap.add_argument("--grace_eps_cos", type=float, default=0.99,
                    help="GRACE: cosine-similarity threshold for a codebook key match "
                         "(hard-replacement variant, one shared eps per layer; ΔW is "
                         "always 0 for this editor)")
    # ---- sequential (no-restore) mode — analyzed by experiments/seq/analyze_sequential.py ----
    ap.add_argument("--no_restore", action="store_true",
                    help="SEQ mode: skip per-edit weight restore. REQUIRES --save_matrices; "
                         "editor must be rome/ft/alpha (MEMIT multi-layer restore is fenced "
                         "at the snapshot site).")
    ap.add_argument("--recheck_every", type=int, default=0,
                    help="SEQ mode: every K edits (and at the final edit) re-run efficacy on "
                         "ALL prior edits (prior-edit overwrite panel). 0=off. Requires --no_restore.")
    ap.add_argument("--probe_stride", type=int, default=1,
                    help="SEQ mode: measure the probe sweep only every S edits (skipped rows "
                         "= NaN). stride>1 requires --no_restore.")
    ap.add_argument("--order_seed", type=int, default=-1,
                    help="SEQ mode ordering-controlled design: -1 (default) = off, byte-identical "
                         "to the old behavior. >=0 = permute the ORDER of the already-SELECTED "
                         "edits (np.random.default_rng(order_seed)) so --seed controls SELECTION "
                         "only and --order_seed controls INSERTION ORDER only — same edit set, "
                         "different sequence. Requires --no_restore.")
    # ---- deletion-edit mode (U1-E0) — refusal is data-layer; eos/suppress use editors/rome_deletion.py ----
    ap.add_argument("--edit_mode", choices=["rewrite", "delete"], default="rewrite")
    ap.add_argument("--delete_variant", choices=["refusal", "eos", "suppress"], default="refusal")
    ap.add_argument("--refusal_string", default="I cannot answer")
    # ---- canonical E/G/L metrics (CounterFact ES/PS/NS + zsRE) — experiments/egl_metrics.py ----
    ap.add_argument("--egl", action="store_true",
                    help="in-run canonical Efficacy/Generality/Locality metrics "
                         "(CF ES/PS/NS + zsRE E/G/L); ~+5%% cell runtime")
    ap.add_argument("--egl_max_neighborhood", type=int, default=10)
    ap.add_argument("--egl_max_paraphrase", type=int, default=2)
    # ---- QuantEdit E0 rank-one vectors dump — scored by experiments/quantedit_e0.py ----
    ap.add_argument("--save_vectors", action="store_true",
                    help="dump per-edit rank-one factors (a,b) + edit key k + base weight "
                         "for the QuantEdit CPU oracle (rome/alpha only)")
    ap.add_argument("--vector_dir", default=os.path.join(HARNESS, "results", "vectors"))
    args = ap.parse_args()

    # ---- mquake data-file guard: fail BEFORE any model loading (fast, clean, actionable) ----
    if args.dataset == "mquake" and not os.path.isfile(args.data):
        raise SystemExit(
            f"[kg] --dataset mquake: data file not found at --data {args.data!r}.\n"
            f"      Expected path (this harness's convention): data/mquake_cf3k.json\n"
            f"      Source: princeton-nlp/MQuAKE repository, file MQuAKE-CF-3k.json.\n"
            f"      Downloads are ask-first in this workspace (HF_HUB_OFFLINE=1 standing "
            f"policy) — do not fetch this automatically; ask the user to provide it."
        )

    # ---- post-parse guards (all new modes are default-off; guards never fire on old CLIs) ----
    if args.device_map != "none" and args.device != "cuda":
        raise SystemExit("[kg] --device_map requires --device cuda (accelerate device_map "
                         "places shards on CUDA devices; --device cpu has no multi-device "
                         "concept here — use --device_map none for CPU smokes)")
    if args.model_dtype == "bf16" and args.editor == "ft":
        # ft runs torch Adam DIRECTLY on the live down_proj Parameter, so under bf16 the
        # optimizer state, CE loss and L2/KL anchors all inherit bf16 — the same
        # silent-degradation class as the fp16 value-opt NaN. Giving FT fp32 master math
        # would mean reparameterizing the live module weight mid-forward (not a trivial
        # bolt-on), so bf16 is fenced to the editors whose only weight op is a
        # dtype-cast rank-one/whitened write-back.
        raise SystemExit("[kg] --model_dtype bf16 supports editor in {rome, alpha, memit} only "
                         "(ft's Adam runs directly on the bf16 Parameter — fp32 master weights "
                         "are not implemented; use --model_dtype fp32 for ft)")
    if args.no_restore and not args.save_matrices:
        raise SystemExit("[kg] --no_restore requires --save_matrices (sequential data lives only in the npz)")
    if args.no_restore and args.editor not in ("rome", "ft", "alpha"):
        # MEMIT restore fence: a multi-layer editor needs the {layer: clone} restore dict AND a
        # multi-layer-aware sequential analysis; not designed yet -> hard stop.
        raise SystemExit("[kg] --no_restore supports editor in {rome,ft,alpha} only")
    if args.recheck_every > 0 and not args.no_restore:
        raise SystemExit("[kg] --recheck_every requires --no_restore")
    if args.probe_stride != 1 and not args.no_restore:
        raise SystemExit("[kg] --probe_stride>1 requires --no_restore (would write NaN rows "
                         "into a restore-mode gate npz)")
    if args.probe_stride < 1:
        raise SystemExit("[kg] --probe_stride must be >= 1")
    if args.order_seed >= 0 and not args.no_restore:
        raise SystemExit("[kg] --order_seed requires --no_restore (ordering-controlled design "
                         "is a SEQ-mode-only concept)")
    if args.edit_mode == "delete":
        if args.editor == "grace":
            # codebook-replacement deletion semantics are undesigned (what would a
            # "deleted" codebook entry even replace the output WITH?) — explicit fence
            # rather than falling through to the eos/suppress/refusal checks below.
            raise SystemExit("[kg] --edit_mode delete is not supported for --editor grace "
                             "(GRACE's ΔW==0 codebook-replacement semantics for a deletion "
                             "objective are undesigned; use --editor rome or alpha)")
        if args.delete_variant in ("eos", "suppress") and args.editor != "rome":
            raise SystemExit("[kg] --delete_variant eos/suppress requires --editor rome")
        if args.delete_variant == "refusal" and args.editor not in ("rome", "alpha"):
            raise SystemExit("[kg] --delete_variant refusal requires --editor rome or alpha")
        if args.no_restore:
            raise SystemExit("[kg] --edit_mode delete + --no_restore is not a designed combination")
        if args.egl:
            raise SystemExit("[kg] --egl is defined for rewrite mode only (ES/PS semantics)")
    if args.editor == "grace" and args.memit_layers != "auto":
        # grace is single-layer (like rome/ft/alpha) — a MEMIT-style multi-layer spec
        # would otherwise be silently ignored (memit_layers is only consumed when
        # args.editor == "memit"), which could read as "it worked" when it did nothing.
        raise SystemExit("[kg] --memit_layers is not applicable to --editor grace "
                         "(grace edits a single down_proj layer via --layer, like rome/alpha)")
    if args.egl and args.no_restore:
        raise SystemExit("[kg] --egl requires the restore-every-edit invariant (baselines are "
                         "base-model state); no sequential EGL")
    if args.egl and args.dataset == "mquake":
        # FENCED (not graceful-degrade): egl_metrics.attach_egl_fields has a CF branch (expects
        # requested_rewrite as a single dict — mquake's is a LIST) and a zsre branch (expects
        # top-level subject/src/alt/pred — mquake has neither); dataset="mquake" would silently
        # fall into the zsre branch, match 0 records, and only be caught later by the coverage
        # gate with a confusing "loader/data mismatch" message. Fence explicitly instead.
        raise SystemExit("[kg] --egl is not supported for --dataset mquake yet (MQuAKE-CF-3k "
                         "reuses the CounterFact rewrite schema but its records carry neither "
                         "CF's paraphrase_prompts/neighborhood_prompts nor zsRE's rephrase/loc "
                         "fields) — canonical ES/PS/NS or E/G/L cannot be computed here; run "
                         "without --egl for mquake cells")
    if args.save_vectors:
        if args.editor not in ("rome", "alpha"):
            raise SystemExit("[kg] --save_vectors requires a rank-one editor (rome/alpha); "
                             "ft is not rank-one and memit is multi-layer")
        if args.no_restore:
            raise SystemExit("[kg] --save_vectors requires restore mode (per-edit delta must be "
                             "isolated against the base weight)")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.editor == "rome":
        if args.edit_mode == "delete" and args.delete_variant in ("eos", "suppress"):
            from editors.rome_deletion import apply_edit
        else:
            from editors.rome_native import apply_edit
    elif args.editor == "ft":
        from editors.ft_editor import apply_edit
    elif args.editor == "memit":
        from editors.memit import apply_edit, estimate_layer_covariances, parse_memit_layers
    elif args.editor == "grace":
        from editors.grace_editor import apply_edit, clear_grace
    else:  # alpha (AlphaEdit null-space projected ROME)
        from editors.alphaedit import apply_edit, build_null_projector
    load_fn = {"counterfact": load_counterfact, "zsre": load_zsre,
              "mquake": load_mquake}[args.dataset]
    device = args.device
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    # fp32 for editing (default): the ROME value-optimization (Adam+log_softmax) NaNs in fp16
    # on real weights; a 1B model in fp32 (~5GB) fits 24GB comfortably. bf16 is opt-in
    # (--model_dtype bf16): the frozen forward runs bf16, the value-opt math stays fp32
    # inside the editors, and every weight write-back casts the fp32 delta to W.dtype.
    load_dtype = torch.float32 if args.model_dtype == "fp32" else torch.bfloat16
    if args.device_map == "none":
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=load_dtype).to(device).eval()
    else:
        # TP path: accelerate shards the model's submodules across every card visible
        # via CUDA_VISIBLE_DEVICES, per `device_map`. Do NOT call .to(device) here — on
        # an accelerate-dispatched model that silently COLLAPSES the whole model back
        # onto one device, destroying the shard placement (this is why every editor's
        # apply_edit() now routes its own model.to(device) through
        # tp_edit_util.safe_model_to(), which no-ops once model.hf_device_map is set).
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=load_dtype, device_map=args.device_map).eval()
        # re-resolve `device` (used everywhere downstream as the INPUT device for
        # tokenizer(...).to(device) encodes) to wherever accelerate actually placed the
        # embedding — under device_map this need not be "cuda:0" (e.g. balanced_low_0
        # reserves card 0 for other work). accelerate's forward hooks move activations
        # across shard boundaries automatically past this point, so every existing
        # `.to(device)` call site in this file / metrics.py / the editors is correct
        # unchanged as long as it targets the INPUT device — only the per-EDITED-LAYER
        # math (value-opt v, AlphaEdit projector) needs the edited layer's own device,
        # handled inside the editors via tp_edit_util.resolve_layer_device.
        device = model.get_input_embeddings().weight.device
        print(f"[kg] --device_map={args.device_map}: hf_device_map={model.hf_device_map}",
              flush=True)
    # actual (not requested) dtype — recorded in provenance + npz so a mislabeled cell
    # can never masquerade as fp32
    actual_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    # TASK-2 load-time arch normalization (editors/arch_compat.py): GPT-2's Conv1D
    # mlp.c_proj is swapped for a byte-equivalent nn.Linear (with an in-run equivalence
    # proof) and a Llama-shaped model.model.layers[li].mlp.down_proj view is grafted on;
    # native Llama-family models return untouched — ZERO downstream call sites change.
    arch = normalize_arch(model, tok, device)
    if arch in ("gpt2", "gptj", "gptneox") and args.editor == "memit":
        # memit's _hidden_at hooks the DECODER-LAYER Module (model.model.layers[l]) for the
        # residual stream; under the GPT-2/GPT-J/GPT-NeoX graft that object is a plain
        # SimpleNamespace with no register_forward_hook — fence it out rather than crash
        # mid-run.
        raise SystemExit(f"[kg] editor=memit is not supported on {arch}-family models "
                         "(residual-stream hook needs the real decoder-layer Module; "
                         "the arch_compat graft only exposes mlp.down_proj)")
    nL = model.config.num_hidden_layers
    layer = nL // 2 if args.layer == "auto" else int(args.layer)
    print(f"[kg] loaded {args.model} ({nL} layers, edit layer={layer}, device={device}, "
          f"dtype={actual_dtype}, arch={arch}, "
          f"~{sum(p.numel() for p in model.parameters())/1e6:.0f}M params) {time.time()-t0:.1f}s", flush=True)

    # MEMIT: resolve + validate the edited layer span (max MUST be the z-layer = --layer)
    memit_layers = None
    if args.editor == "memit":
        try:
            memit_layers = parse_memit_layers(args.memit_layers, layer, nL)
        except ValueError as e:
            raise SystemExit(f"[kg] {e}")
        print(f"[kg] memit layers={memit_layers} (z-layer={layer}, "
              f"cov_source={args.memit_cov_source})", flush=True)

    # held-out bank: disjoint AlphaEdit projector fit OR MEMIT generic covariance prompts
    n_holdout = (int(round(args.holdout_frac * args.n_probes))
                 if ((args.editor == "alpha" and args.alpha_proj_source != "probes")
                     or (args.editor == "memit" and args.memit_cov_source == "generic")) else 0)
    edits, probes, holdout = load_fn(args.data, args.n_edits, args.n_probes, args.seed, n_holdout)
    print(f"[kg] {args.editor}/{args.dataset}: {len(edits)} edits, {len(probes)} probes, "
          f"{len(holdout)} holdout(proj/cov-fit)", flush=True)
    if n_holdout and len(holdout) < 5:
        src = (args.alpha_proj_source if args.editor == "alpha" else args.memit_cov_source)
        raise SystemExit(f"[kg] editor={args.editor} source={src} needs a holdout bank "
                         f"but only {len(holdout)} disjoint records available — lower --holdout_frac "
                         f"or n_edits/n_probes.")

    # ---- --order_seed: permute INSERTION ORDER of the already-selected `edits`, right after
    # selection/key-capture bookkeeping is aligned (selection is done; no key has been captured
    # yet). Every downstream artifact (K_edit, COS, GRAM_pre, the edit loop, npz row order) is
    # built by iterating over `edits` in list order, so permuting the list HERE is sufficient to
    # keep all of them consistent with the same permutation — no separate re-indexing needed.
    # `orig_index[k]` records which pre-permutation position now sits at row k (identity when off).
    orig_index = np.arange(len(edits), dtype=np.int64)
    if args.order_seed >= 0:
        perm = np.random.default_rng(args.order_seed).permutation(len(edits))
        edits = [edits[k] for k in perm]
        orig_index = perm.astype(np.int64)

    # ---- U1-E0 deletion swap — at the DATA layer, AFTER load_fn (loader internals are
    # reimplemented VERBATIM by u1_transplant.py / lexical_sbert_baseline.py; mutating them
    # would silently desync row/col alignment). Keys depend only on prompt/subject, so the
    # precomputed COS geometry is unaffected by the target swap.
    if args.edit_mode == "delete" and args.delete_variant == "refusal":
        for e in edits:
            e["target_insert_orig"] = e["target_new"]
            e["target_new"] = args.refusal_string
    # eos/suppress: target_new left intact; the deletion objective lives in editors/rome_deletion.py

    # ---- TOKENIZER DISTINGUISHABILITY GATE (added 2026-07-30) ----
    # Fails BEFORE any GPU work when the tokenizer cannot tell two distinct targets
    # apart by their first content token. Phi-3.5's SentencePiece collapsed every
    # target onto the whitespace-marker id 29871, which made ROME optimise toward
    # whitespace and the scorer read that same token back as "success" — 7 cells were
    # silently meaningless before this gate existed. One cheap CPU check per run.
    # See docs/findings/findings-PHI35-TOKENIZER-COLLISION-2026-07-30.md
    _gate_targets = [e.get("target_new") for e in edits] + [e.get("target_true") for e in edits] \
        + [p.get("target_true") for p in probes]
    tok_gate = assert_targets_distinguishable(tok, [t for t in _gate_targets if t])
    print(f"[kg] tokenizer distinguishability gate PASS: "
          f"{tok_gate['n_first_tokens']}/{tok_gate['n_targets']} distinct first tokens "
          f"(ratio {tok_gate['ratio']:.3f}, {tok_gate['n_collisions']} colliding pairs)",
          flush=True)
    if tok_gate["n_collisions"]:
        # Benign prefix sharing: NOT a defect, but a scoring caveat the paper must
        # disclose for this model. Logged per-run so it cannot go unnoticed again.
        _ex = "; ".join(f"{a!r}/{b!r}" for a, b, _ in tok_gate["collisions"][:5])
        print(f"[kg] WARN first-token prefix sharing on {tok_gate['n_collisions']} pair(s): "
              f"{_ex} — argmax efficacy cannot separate these", flush=True)

    # ---- canonical E/G/L: attach paraphrase/neighborhood fields by content match-back ----
    egl_records = []
    egl_funcs = None
    if args.egl:
        # Dual-path import: invoked as `python3 experiments/killgate_keygeom.py` the script
        # dir (experiments/) is sys.path[0], so the package-qualified form fails — never
        # caught before because --egl had NEVER been exercised (found by r3 smoke 2026-07-03).
        try:
            from experiments.egl_metrics import (  # noqa: E402
            attach_egl_fields, precompute_egl_baselines, measure_egl_one,
            summarize_egl, egl_npz_arrays, write_egl_sidecar,
            )
        except ModuleNotFoundError:
            from egl_metrics import (  # noqa: E402
            attach_egl_fields, precompute_egl_baselines, measure_egl_one,
            summarize_egl, egl_npz_arrays, write_egl_sidecar,
            )
        egl_funcs = True
        n_match = attach_egl_fields(edits, args.data, args.dataset)
        cov_req = 0.95 if args.dataset == "counterfact" else 0.99
        if n_match < cov_req * len(edits):
            raise SystemExit(f"[kg] --egl coverage gate: matched {n_match}/{len(edits)} "
                             f"< {cov_req:.0%} — loader/data mismatch, run INVALID")
        print(f"[kg] egl match-back: {n_match}/{len(edits)} edits carry paraphrase/"
              f"neighborhood fields", flush=True)

    # ---- capture all base-model keys ONCE ----
    def key_for(prompt, subject):
        idx = find_subject_last_token_index(tok, prompt, subject)
        return _capture_key(model, tok, layer, prompt, idx, device).float().cpu().numpy()

    def generic_key_for(prompt, rng):
        """Key at a RANDOM non-final token position (generic-activation covariance,
        NOT the fact's subject key) — the distribution-shifted projector control."""
        n = len(tok.encode(prompt, add_special_tokens=True))
        idx = int(rng.integers(0, max(1, n - 1))) if n > 1 else 0
        return _capture_key(model, tok, layer, prompt, idx, device).float().cpu().numpy()

    K_edit = np.stack([key_for(e["prompt"], e["subject"]) for e in edits])       # [N, d]
    K_probe = np.stack([key_for(p["prompt"], p["subject"]) for p in probes])     # [M, d]
    # cosine matrix N x M
    Ke = K_edit / (np.linalg.norm(K_edit, axis=1, keepdims=True) + 1e-8)
    Kp = K_probe / (np.linalg.norm(K_probe, axis=1, keepdims=True) + 1e-8)
    COS = Ke @ Kp.T                                                              # [N, M]
    # SEQ mode: edit-edit Gram of PRE-SEQUENCE keys. Live keys are mathematically identical
    # to pre keys for rome/ft/alpha (the only edited weight is down_proj@L; the key is its
    # INPUT, a function of strictly upstream params) — verified per-step by the invariance
    # tripwire in the loop, so no separate live-key bank is stored.
    GRAM_pre = (Ke @ Ke.T).astype(np.float32) if args.no_restore else None
    print(f"[kg] keys+cosine done {time.time()-t0:.1f}s", flush=True)

    # AlphaEdit: build the null-space projector ONCE from the probe-bank keys
    # (= the preserved set; tests whether AlphaEdit disproportionately protects
    # high-cosine probes — the G4 causal check / D3 routing signal).
    alpha_proj = None
    proj_fit_keys = None   # the key bank the projector was actually fit on (for provenance)
    if args.editor == "alpha":
        if args.alpha_proj_source == "probes":
            proj_fit_keys = K_probe                                   # by-construction reference
        elif args.alpha_proj_source == "holdout":
            proj_fit_keys = np.stack([key_for(h["prompt"], h["subject"]) for h in holdout])
        else:  # generic — random non-subject token keys from the held-out prompts
            grng = np.random.default_rng(args.seed + 777)
            proj_fit_keys = np.stack([generic_key_for(h["prompt"], grng) for h in holdout])
        # built on `device` (input device) here — under --device_map TP that may not be
        # the edited layer's device; harmless (not a correctness bug), because
        # alphaedit._resolve_projector re-homes config["projector"] onto
        # resolve_layer_device(model, layer_idx) before every use (tp_edit_util.py).
        alpha_proj = build_null_projector(torch.tensor(proj_fit_keys, device=device), args.keep_ratio)
        _removed = int(round(alpha_proj.shape[0] - float(alpha_proj.diagonal().sum())))
        print(f"[kg] alphaedit projector [{args.alpha_proj_source}]: fit on "
              f"{proj_fit_keys.shape[0]} keys, removed {_removed} dims", flush=True)

    # MEMIT: build the per-layer key covariance ONCE (CPU-resident chol, zero VRAM)
    memit_cov = None
    if args.editor == "memit" and args.memit_cov_source != "identity":
        cov_prompts = ([h["prompt"] for h in holdout] if args.memit_cov_source == "generic"
                       else [p["prompt"] for p in probes])
        memit_cov = estimate_layer_covariances(model, tok, cov_prompts, memit_layers, device,
                                               max_tokens=args.memit_cov_tokens,
                                               reg=args.memit_cov_reg)
        for _l in memit_layers:
            print(f"[kg] memit cov L{_l}: n_tokens={memit_cov[_l]['n_tokens']} "
                  f"reg_used={memit_cov[_l]['reg_used']:.3g}", flush=True)

    # ---- probe baseline P/logit of correct object (pre-edit) ----
    probe_tok = [first_target_token_id(tok, p["target_true"]) for p in probes]
    pre_p = np.zeros(len(probes)); pre_l = np.zeros(len(probes))
    for j, p in enumerate(probes):
        pre_p[j], pre_l[j] = prob_of_token(model, tok, p["prompt"], probe_tok[j], device)

    # ---- EGL base-model baselines (valid ONLY under the restore-every-edit invariant;
    #      any future sequential EGL must recompute — same trap class as the COS matrix) ----
    egl_base = None
    if args.egl:
        egl_base = precompute_egl_baselines(model, tok, edits, device, args.dataset,
                                            args.egl_max_neighborhood)
        print(f"[kg] egl baselines precomputed {time.time()-t0:.1f}s", flush=True)

    # ---- deletion-mode pre-edit P(target_true) on the EDIT prompts (delete mode ONLY —
    #      rewrite runs are byte-identical to the pre-change harness) ----
    edit_true_tok = None
    edit_ptrue_pre = None
    if args.edit_mode == "delete":
        edit_true_tok = [first_target_token_id(tok, e["target_true"]) for e in edits]
        edit_ptrue_pre = np.zeros(len(edits))
        for i, e in enumerate(edits):
            edit_ptrue_pre[i], _ = prob_of_token(model, tok, e["prompt"], edit_true_tok[i], device)

    # ---- snapshot the editable weight(s) for fast restore ----
    # Restore is a dict keyed by layer: single-element for rome/ft/alpha (byte-identical
    # behavior to the old single W/W_base pair), multi-element for MEMIT. Anyone adding a
    # new multi-layer editor MUST extend restore_layers or the restore silently corrupts
    # every subsequent edit's baselines (harness-map restore invariant (a)).
    restore_layers = memit_layers if args.editor == "memit" else [layer]
    W_refs = {li: model.model.layers[li].mlp.down_proj.weight for li in restore_layers}
    # dtype note (bf16 audit): clone() inherits W's dtype and copy_() below is then a
    # same-dtype copy, so the restore stays EXACT under fp32 and bf16 alike; the only
    # lossy cast in the pipeline is the editors' delta.to(W.dtype) at write-back.
    W_bases = {li: w.detach().clone() for li, w in W_refs.items()}
    W = W_refs[layer]
    W_base = W_bases[layer]

    N_e, M_p = len(edits), len(probes)
    if args.no_restore:
        # SEQ mode: skipped probe rows (probe_stride) must be NaN, never fake zeros
        damage_p = np.full((N_e, M_p), np.nan)
        damage_l = np.full((N_e, M_p), np.nan)
    else:
        damage_p = np.zeros((N_e, M_p))   # P_pre - P_post  (positive=damaged)
        damage_l = np.zeros((N_e, M_p))
    norm_growth = np.zeros(N_e)
    delta_norm_total = np.zeros(N_e)  # total ΔW norm over ALL edited layers (== norm_growth
    #   for single-layer editors; the PRIMARY NG confound for memit cells — prereg, see
    #   editors/memit.py docstring + experiments/analyze_memit_ngtotal.py)
    edit_ok = np.zeros(N_e)
    resid_norm = np.full(N_e, np.nan)  # ‖v−Wk‖ per ROME edit (S factor / mechanism test); NaN for ft
    neighbor_prompts = [p["prompt"] for p in probes[:max(1, args.ft_kl_n)]]  # D1 KL locality anchor
    # deletion-mode receipts
    edit_ptrue_post = np.full(N_e, np.nan)
    edit_argmax_ok = np.full(N_e, np.nan)
    # SEQ-mode panels
    eff_pnew_post = np.full(N_e, np.nan)
    eff_ptrue_post = np.full(N_e, np.nan)
    recheck_at = []
    prior_eff_rows, prior_pnew_rows, prior_ptrue_rows = [], [], []
    key_check_every = max(args.recheck_every, 10)
    # QuantEdit vectors accumulators
    vec_A, vec_B, vec_K, vec_recon = [], [], [], []

    for i, e in enumerate(edits):
        # SEQ-mode key-invariance tripwire: for rome/ft/alpha the key at layer L is a
        # function of strictly UPSTREAM params, so it must equal the pre-sequence key
        # bitwise-closely even under accumulated edits; drift => state corruption.
        if args.no_restore and (i % key_check_every == 0):
            k_now = key_for(e["prompt"], e["subject"])
            if not np.allclose(k_now, K_edit[i], rtol=1e-5, atol=1e-5):
                raise SystemExit(f"[kg] SEQ key-invariance TRIPWIRE at edit {i}: max|Δk|="
                                 f"{float(np.abs(k_now - K_edit[i]).max()):.3g} — upstream "
                                 f"state corrupted, aborting")
        if args.editor == "rome":
            cfg = {"layer": layer, "steps": args.steps, "lr": args.lr}
            if args.edit_mode == "delete" and args.delete_variant in ("eos", "suppress"):
                cfg["delete_variant"] = args.delete_variant
        elif args.editor == "ft":  # ft edits the same down_proj; FT-L lr is much smaller
            cfg = {"layers": [layer], "steps": args.steps, "lr": args.ft_lr}
            if args.ft_kl > 0:  # D1 control: KL-locality regularized FT
                cfg["lambda_kl"] = args.ft_kl
                e = {**e, "neighborhood_prompts": neighbor_prompts}
        elif args.editor == "memit":
            cfg = {"layers": memit_layers, "z_layer": layer, "steps": args.steps,
                   "lr": args.lr, "cov": memit_cov, "cov_source": args.memit_cov_source}
        elif args.editor == "grace":  # codebook replacement, ΔW == 0
            cfg = {"layer": layer, "steps": args.steps, "lr": args.lr,
                   "grace_eps_cos": args.grace_eps_cos}
        else:  # alpha: ROME value-opt + null-space-projected rank-one update
            cfg = {"layer": layer, "steps": args.steps, "lr": args.lr, "projector": alpha_proj}
        info = apply_edit(model, tok, e, cfg, device)
        # bf16 NaN tripwire (1/2): a NaN value-opt loss under bf16 must be LOUD, never a
        # silent all-NaN cell (fp16 lesson). Gated on bf16 so the fp32 path is untouched.
        if args.model_dtype == "bf16":
            _fvl = info.get("final_value_loss")
            if _fvl is not None and not np.isfinite(_fvl):
                print(f"[kg] WARN bf16 tripwire: value-opt loss non-finite ({_fvl}) at "
                      f"edit {i} — inspect this cell before trusting the npz", flush=True)
        ng = info["delta_weight_norm"]
        norm_growth[i] = float(ng[layer]) if isinstance(ng, dict) else float(ng)
        delta_norm_total[i] = float(info.get(
            "delta_weight_norm_total",
            sum(ng.values()) if isinstance(ng, dict) else float(ng)))
        resid_norm[i] = float(info.get("residual_norm", np.nan))
        eff = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
        if args.edit_mode == "delete":
            # suppression receipt: 2x drop of P(target_true) on the edit prompt
            edit_ptrue_post[i] = eff["p_true"]
            edit_ok[i] = 1.0 if eff["p_true"] < 0.5 * edit_ptrue_pre[i] else 0.0
            if args.delete_variant == "refusal":
                edit_argmax_ok[i] = eff["success"]          # argmax == first refusal token
            elif args.delete_variant == "eos":
                edit_argmax_ok[i] = 1.0 if int(eff["argmax_id"]) == int(tok.eos_token_id) else 0.0
            else:  # suppress: argmax moved OFF the true target
                edit_argmax_ok[i] = 1.0 if int(eff["argmax_id"]) != int(edit_true_tok[i]) else 0.0
        else:
            edit_ok[i] = eff["success"]  # real efficacy: did edited model argmax the new target?
        if args.no_restore:
            eff_pnew_post[i] = eff["p_new"]
            eff_ptrue_post[i] = eff.get("p_true", float("nan"))
        # probe sweep — SEQ mode measures every probe_stride-th edit (and always the last);
        # SEQ damage semantics = CUMULATIVE drop vs the fixed PRE-SEQUENCE baseline
        # (per-step marginal damage = np.diff in the analyzer, never stored)
        if (not args.no_restore) or ((i + 1) % args.probe_stride == 0) or (i + 1) == len(edits):
            for j, p in enumerate(probes):
                pp, ll = prob_of_token(model, tok, p["prompt"], probe_tok[j], device)
                damage_p[i, j] = pre_p[j] - pp
                damage_l[i, j] = pre_l[j] - ll
            # bf16 NaN tripwire (2/2): an all-NaN damage row means the edited forward
            # NaN'd — warn loudly (inside this block so SEQ probe_stride's intentional
            # NaN rows never trip it). bf16-gated: zero fp32-path behavior change.
            if args.model_dtype == "bf16" and np.isnan(damage_l[i]).all():
                print(f"[kg] WARN bf16 tripwire: damage row {i} is ALL-NaN — edited "
                      f"model output non-finite under bf16", flush=True)
        # SEQ prior-edit overwrite panel (H1): recheck ALL prior edits every K edits
        if args.no_restore and args.recheck_every > 0 and (
                (i + 1) % args.recheck_every == 0 or (i + 1) == len(edits)):
            row_e = np.full(N_e, np.nan); row_pn = np.full(N_e, np.nan); row_pt = np.full(N_e, np.nan)
            for jj in range(i + 1):
                e2 = edits[jj]
                ef2 = efficacy(model, tok, e2["prompt"], e2["target_new"],
                               e2.get("target_true"), device)
                row_e[jj] = ef2["success"]; row_pn[jj] = ef2["p_new"]
                row_pt[jj] = ef2.get("p_true", float("nan"))
            prior_eff_rows.append(row_e); prior_pnew_rows.append(row_pn)
            prior_ptrue_rows.append(row_pt); recheck_at.append(i + 1)
        # EGL: measure under the EDITED weights, strictly before restore
        if args.egl:
            egl_records.append(measure_egl_one(
                model, tok, e, egl_base[i], device, args.dataset, eff=eff,
                max_neighborhood=args.egl_max_neighborhood,
                max_paraphrase=args.egl_max_paraphrase))
        # QuantEdit vectors: exact rank-one factors of ΔW at the z-layer (rome/alpha),
        # strictly before restore. NEVER hard-crashes the cell (a bad recon is recorded
        # + warned; the hard gate is quantedit_e0.py --validate_npz on CPU).
        if args.save_vectors:
            with torch.no_grad():
                # device=W.device (the EDITED layer's own device), not the ambient
                # `device` (INPUT/embedding device) — combined below with Dw = W -
                # W_base, which lives wherever the edited layer actually sits; under
                # --device_map TP these can differ (tp_edit_util.py).
                k_t = torch.tensor(K_edit[i], device=W.device, dtype=torch.float32)
                Dw = (W.detach().float() - W_base.float())
                a = Dw @ k_t
                b = (Dw.t() @ a) / (float((a @ a).item()) + 1e-30)
                Db = Dw @ b
                aTDb = float((a @ Db).item())
                Dfro2 = float((Dw * Dw).sum().item())
                an2 = float((a @ a).item()); bn2 = float((b @ b).item())
                recon = (np.sqrt(max(0.0, Dfro2 + an2 * bn2 - 2.0 * aTDb))
                         / (np.sqrt(Dfro2) + 1e-30))
                if recon > 1e-4:
                    print(f"[kg] WARN save_vectors: recon_rel_err={recon:.3g} at edit {i} "
                          f"(>1e-4); vectors_valid will reflect this", flush=True)
                vec_A.append(a.detach().cpu().numpy().astype(np.float32))
                vec_B.append(b.detach().cpu().numpy().astype(np.float32))
                vec_K.append(K_edit[i].astype(np.float32))
                vec_recon.append(float(recon))
        if not args.no_restore:
            with torch.no_grad():
                for li in restore_layers:
                    W_refs[li].copy_(W_bases[li])          # restore (dict; 1 layer unless memit)
            if args.editor == "grace":
                # grace's "restore" is codebook state, not weights (ΔW==0 by
                # construction) — clear it here so edit i+1 starts from an empty
                # codebook, mirroring the weight-restore invariant every other
                # editor gets for free (see editors/grace_editor.py docstring).
                clear_grace(model)
        if (i + 1) % 10 == 0:
            print(f"[kg] edit {i+1}/{len(edits)}  {time.time()-t0:.1f}s", flush=True)

    # SEQ mode: unconditional final restore so downstream state is clean, + verification
    if args.no_restore:
        with torch.no_grad():
            for li in restore_layers:
                W_refs[li].copy_(W_bases[li])
        assert all(torch.allclose(W_refs[li], W_bases[li]) for li in restore_layers), \
            "[kg] SEQ final restore FAILED"
        print("[kg] SEQ_MODE final restore verified (allclose on all snapshotted layers)",
              flush=True)

    # ---- analysis: flatten (edit,probe) pairs ----
    cos_flat = COS.reshape(-1)
    dmg_flat = damage_l.reshape(-1)          # primary damage = drop in correct-token logit
    ng_flat = np.repeat(norm_growth, len(probes))
    known = np.repeat((pre_p > 0.05).astype(int)[None, :], len(edits), axis=0).reshape(-1)

    def report(mask, name):
        c, d, n = cos_flat[mask], dmg_flat[mask], ng_flat[mask]
        if len(c) < 20:
            return {"subset": name, "n_pairs": int(len(c)), "note": "too few"}
        broken = (d >= np.quantile(d, 0.9)).astype(int)   # top-decile damaged
        return {
            "subset": name, "n_pairs": int(len(c)),
            "spearman_cos_damage": round(spearman(c, d), 4),
            "spearman_normgrowth_damage": round(spearman(n, d), 4),
            "auroc_cos_broken": round(auroc(c, broken), 4),
            "auroc_normgrowth_broken": round(auroc(n, broken), 4),
            "mean_damage_logit": round(float(d.mean()), 5),
            "mean_cosine": round(float(c.mean()), 4),
        }

    import transformers as _tf
    provenance = {
        "torch": torch.__version__,
        "transformers": _tf.__version__,
        "numpy": np.__version__,
        # actual loaded dtype (was a hardcoded "float32"; now truthful under --model_dtype bf16)
        "dtype": actual_dtype,
        "model_dtype_arg": args.model_dtype,
        "arch": arch,
        "lr": args.lr, "ft_lr": args.ft_lr, "steps": args.steps,
    }
    if args.editor == "alpha":
        provenance["alpha_proj_source"] = args.alpha_proj_source
        provenance["keep_ratio"] = args.keep_ratio
        provenance["n_proj_fit_keys"] = int(proj_fit_keys.shape[0]) if proj_fit_keys is not None else 0
        provenance["proj_disjoint_from_probes"] = (args.alpha_proj_source != "probes")
    if args.editor == "memit":
        provenance["memit_layers"] = memit_layers
        provenance["memit_cov_source"] = args.memit_cov_source
        provenance["memit_cov_reg"] = args.memit_cov_reg
        provenance["memit_cov_tokens_cap"] = args.memit_cov_tokens
        if memit_cov is not None:
            provenance["memit_cov_n_tokens"] = {int(l): memit_cov[l]["n_tokens"] for l in memit_layers}
            provenance["memit_cov_reg_used"] = {int(l): memit_cov[l]["reg_used"] for l in memit_layers}
    if args.editor == "grace":
        provenance["grace_eps_cos"] = args.grace_eps_cos
    if args.edit_mode == "delete":
        provenance["edit_mode"] = "delete"
        provenance["delete_variant"] = args.delete_variant
        if args.delete_variant == "refusal":
            provenance["refusal_string"] = args.refusal_string

    # legacy flat reports are NaN-unsafe on cumulative/strided SEQ damage — bypassed there
    if args.no_restore:
        _seq_note = {"note": "SEQ_MODE — cumulative damage vs fixed pre-sequence baseline; "
                             "gate lives in experiments/seq/analyze_sequential.py"}
        all_pairs_rep, known_rep = dict(_seq_note), dict(_seq_note)
    else:
        all_pairs_rep = report(np.ones_like(cos_flat, bool), "all")
        known_rep = report(known.astype(bool), "known_only")

    res = {
        "model": args.model, "editor": args.editor, "dataset": args.dataset,
        "layer": layer, "n_edits": len(edits), "n_probes": len(probes),
        "steps": args.steps, "seed": args.seed,
        "alpha_proj_source": (args.alpha_proj_source if args.editor == "alpha" else None),
        "provenance": provenance,
        "edit_success_rate": round(float(edit_ok.mean()), 3),
        "mean_residual_norm": (None if np.all(np.isnan(resid_norm))
                               else round(float(np.nanmean(resid_norm)), 4)),  # ROME edit-strength S
        "frac_probes_known(pre_p>0.05)": round(float((pre_p > 0.05).mean()), 3),
        "ALL_PAIRS": all_pairs_rep,
        "KNOWN_PROBES": known_rep,
        "runtime_s": round(time.time() - t0, 1),
        "runner_stamp": _runner_stamp(t0),
    }
    if args.editor == "memit":
        res["memit_layers"] = memit_layers
        res["memit_cov_source"] = args.memit_cov_source
        res["mean_delta_norm_total"] = round(float(delta_norm_total.mean()), 4)
    if args.editor == "grace":
        res["grace_eps_cos"] = args.grace_eps_cos
    if args.edit_mode == "delete":
        res["edit_mode"] = "delete"
        res["delete_variant"] = args.delete_variant
        if args.delete_variant == "refusal":
            res["refusal_string"] = args.refusal_string
        res["suppression_rate"] = round(float(edit_ok.mean()), 3)
        res["mean_ptrue_pre"] = round(float(edit_ptrue_pre.mean()), 5)
        res["mean_ptrue_post"] = round(float(np.nanmean(edit_ptrue_post)), 5)
        res["edit_argmax_ok_rate"] = round(float(np.nanmean(edit_argmax_ok)), 3)
        res["verdict_note"] = ("legacy flat gate — the U1-E0 verdict comes from "
                               "experiments/u1_deletion_gate.py")
    if args.no_restore:
        res["no_restore"] = True
        res["recheck_every"] = args.recheck_every
        res["probe_stride"] = args.probe_stride
        if args.order_seed >= 0:  # additive-only: absent (byte-identical json) when off
            res["order_seed"] = args.order_seed
        # ADVISORY ONLY — the binding collapse call is analysis-side (analyze_sequential.py)
        collapse_adv = None
        if N_e >= 10:
            roll = np.convolve(edit_ok, np.ones(10) / 10.0, mode="valid")
            hits = np.where(roll < 0.5)[0]
            collapse_adv = int(hits[0] + 10) if hits.size else None
        res["collapse_advisory_first_edit"] = collapse_adv
    # KILL-GATE verdict on the KNOWN-probes subset (cleanest signal)
    if args.no_restore:
        res["VERDICT"] = "SEQ_MODE — gate lives in experiments/seq/analyze_sequential.py"
    else:
        kp = res["KNOWN_PROBES"]
        if isinstance(kp.get("spearman_cos_damage"), float):
            rho = abs(kp["spearman_cos_damage"]); au = kp["auroc_cos_broken"]
            beats_ng = au > kp["auroc_normgrowth_broken"]
            res["VERDICT"] = (
                "PASS — key geometry predicts damage" if (rho >= 0.2 and au >= 0.6 and beats_ng)
                else "WEAK/FAIL — see kill-gate criteria"
            )
            res["verdict_detail"] = {"abs_spearman": round(rho, 4), "auroc": au,
                                     "beats_normgrowth": bool(beats_ng)}
    # ---- G0: dump raw matrices for the partialled-correlation GATE ----
    if args.save_matrices:
        os.makedirs(args.matrix_dir, exist_ok=True)
        tag = os.path.splitext(os.path.basename(args.out))[0]
        npz = os.path.join(args.matrix_dir, tag + ".npz")
        # base dict = the LEGACY schema, names/shapes untouched (analyze_matrices.analyze_one
        # contract); `extra` keys are ADDITIVE only — consumers access by name.
        arrs = dict(
            COS=COS.astype(np.float32),                  # [N,M] pre-edit key cosine(k_edit,k_probe)
            damage_logit=damage_l.astype(np.float32),    # [N,M] pre-post drop in probe correct-token logit
            damage_prob=damage_p.astype(np.float32),     # [N,M] pre-post drop in probe correct-token prob
            pre_l=pre_l.astype(np.float32),              # [M]   probe baseline logit
            pre_p=pre_p.astype(np.float32),              # [M]   probe baseline prob (for known-probe filter)
            norm_growth=norm_growth.astype(np.float32),  # [N]   ENCORE norm-growth predictor
            #   NOTE (memit): norm_growth records the Z-LAYER delta only (keeps the NG
            #   confound comparable across editors at the geometry layer); the TOTAL
            #   multi-layer spread norm lives in delta_norm_total. Mixing the two in an
            #   analysis is a silent NG-partialling error — delta_norm_total is the
            #   PRIMARY NG confound for memit cells (prereg; analyze_memit_ngtotal.py).
            edit_ok=edit_ok.astype(np.float32),          # [N]   per-edit efficacy (argmax==target;
            #                                                    delete mode: 2x-suppression criterion)
            resid_norm=resid_norm.astype(np.float32),    # [N]   ‖v−Wk‖ ROME edit-strength S (NaN for ft)
            # provenance so C4 aggregation can honestly separate by-construction vs disjoint projectors
            alpha_proj_source=np.array(
                (args.alpha_proj_source if args.editor == "alpha" else "n/a"), dtype="U16"),
            proj_disjoint=np.array(
                int(args.editor == "alpha" and args.alpha_proj_source != "probes"), dtype=np.int8),
        )
        # additive provenance (always present on new runs; "n/a" outside their mode)
        arrs["model_dtype"] = np.array(actual_dtype, dtype="U16")   # actual load dtype ("float32"/"bfloat16")
        arrs["edit_mode"] = np.array(args.edit_mode, dtype="U16")
        arrs["delete_variant"] = np.array(
            args.delete_variant if args.edit_mode == "delete" else "n/a", dtype="U16")
        arrs["refusal_string"] = np.array(
            args.refusal_string if (args.edit_mode == "delete"
                                    and args.delete_variant == "refusal") else "n/a", dtype="U32")
        arrs["delta_norm_total"] = delta_norm_total.astype(np.float32)   # [N] total ΔW over layers
        arrs["memit_layers"] = np.array(
            ",".join(map(str, memit_layers)) if args.editor == "memit" else "n/a", dtype="U32")
        arrs["memit_cov_source"] = np.array(
            args.memit_cov_source if args.editor == "memit" else "n/a", dtype="U16")
        arrs["grace_eps_cos"] = np.array(
            args.grace_eps_cos if args.editor == "grace" else -1.0, dtype=np.float32)
        arrs["runner_stamp_json"] = np.array(
            json.dumps(res["runner_stamp"], sort_keys=True), dtype="U2048")
        if args.edit_mode == "delete":
            arrs["edit_ptrue_pre"] = edit_ptrue_pre.astype(np.float32)     # [N]
            arrs["edit_ptrue_post"] = edit_ptrue_post.astype(np.float32)   # [N]
            arrs["edit_argmax_ok"] = edit_argmax_ok.astype(np.float32)     # [N]
        if args.no_restore:
            arrs["seq_no_restore"] = np.array(1, dtype=np.int8)
            arrs["recheck_every"] = np.array(args.recheck_every, dtype=np.int32)
            arrs["probe_stride"] = np.array(args.probe_stride, dtype=np.int32)
            if args.order_seed >= 0:  # additive-only: absent from the npz (byte-identical) when off
                arrs["order_seed"] = np.array(args.order_seed, dtype=np.int64)
                arrs["orig_index"] = orig_index.astype(np.int64)  # [N] pre-permutation position of each row
            arrs["GRAM_pre"] = GRAM_pre                                    # [N,N] pre-sequence key cosines
            arrs["key_norm"] = np.linalg.norm(K_edit, axis=1).astype(np.float32)  # [N] (H2 mechanical null)
            arrs["eff_pnew_post"] = eff_pnew_post.astype(np.float32)       # [N] p_new right after edit i
            arrs["eff_ptrue_post"] = eff_ptrue_post.astype(np.float32)
            arrs["recheck_at"] = np.array(recheck_at, dtype=np.int32)      # [C]
            if prior_eff_rows:
                arrs["prior_eff"] = np.stack(prior_eff_rows).astype(np.float32)    # [C,N] NaN j>checkpoint
                arrs["prior_pnew"] = np.stack(prior_pnew_rows).astype(np.float32)
                arrs["prior_ptrue"] = np.stack(prior_ptrue_rows).astype(np.float32)
            # convenience copy — the analyzer recomputes from GRAM_pre + resid_norm (source of truth)
            S_safe = np.nan_to_num(resid_norm, nan=0.0)
            absG = np.abs(GRAM_pre)
            cum_int = np.array([float((S_safe[:t] * absG[:t, t]).sum()) for t in range(N_e)])
            arrs["cum_interference_pre"] = cum_int.astype(np.float32)
        if args.egl and egl_records:
            arrs.update(egl_npz_arrays(egl_records))
        # atomic write (tmp + os.replace), mirroring the vectors/json writers below —
        # so a same-path concurrent write can never truncate the gate's only input npz
        # (mopup review MAJOR #4, 2026-07-13)
        npztmp = npz + ".tmp.npz"   # MUST end in .npz — savez appends .npz otherwise (cf. vtmp)
        np.savez_compressed(npztmp, **arrs)
        os.replace(npztmp, npz)
        res["matrices_npz"] = npz
        print(f"[kg] saved raw matrices -> {npz}", flush=True)

    # ---- QuantEdit E0: rank-one vectors dump (atomic npz BEFORE the json commit marker) ----
    if args.save_vectors:
        os.makedirs(args.vector_dir, exist_ok=True)
        vtag = os.path.splitext(os.path.basename(args.out))[0]
        vpath = os.path.join(args.vector_dir, "vectors_" + vtag + ".npz")
        recon_arr = np.array(vec_recon, dtype=np.float32)
        vectors_valid = bool(recon_arr.size and float(recon_arr.max()) <= 1e-3)
        if not vectors_valid:
            print(f"[kg] WARN vectors dump: max recon_rel_err="
                  f"{float(recon_arr.max()) if recon_arr.size else float('nan'):.3g} > 1e-3 — "
                  f"vectors_valid=False (cell NOT aborted; quantedit_e0.py --validate_npz is "
                  f"the hard gate)", flush=True)
        # NB: np.savez appends ".npz" to any path not ending in it — keep the suffix so the
        # tmp name is deterministic (bug caught by run8h smoke 2026-07-02: wrote .tmp.npz,
        # os.replace then missed .tmp).
        vtmp = vpath + ".tmp.npz"
        np.savez_compressed(
            vtmp,
            K=np.stack(vec_K),                            # [N, d_in]  base-model edit keys
            A=np.stack(vec_A),                            # [N, d_out] exact left factors (scaled)
            B=np.stack(vec_B),                            # [N, d_in]  exact right factors
            recon_rel_err=recon_arr,                      # [N]
            vectors_valid=np.array(int(vectors_valid), dtype=np.int8),
            Wbase=W_base.detach().float().cpu().numpy(),  # [d_out, d_in] once
            resid_norm=resid_norm.astype(np.float32),
            norm_growth=norm_growth.astype(np.float32),
            edit_ok=edit_ok.astype(np.float32),
            knorm=np.linalg.norm(np.stack(vec_K), axis=1).astype(np.float32),
            model=np.array(args.model, dtype="U64"),
            editor=np.array(args.editor, dtype="U16"),
            dataset=np.array(args.dataset, dtype="U16"),
            layer=np.array(layer, dtype=np.int64),
            seed=np.array(args.seed, dtype=np.int64),
            steps=np.array(args.steps, dtype=np.int64),
            lr=np.array(args.lr, dtype=np.float64),
            alpha_proj_source=np.array(
                (args.alpha_proj_source if args.editor == "alpha" else "n/a"), dtype="U16"),
            n_edits=np.array(len(edits), dtype=np.int64),
            torch_version=np.array(torch.__version__, dtype="U16"),
            transformers_version=np.array(_tf.__version__, dtype="U16"),
            edit_mode=np.array(args.edit_mode, dtype="U16"),
        )
        os.replace(vtmp, vpath)
        res["vectors_npz"] = vpath
        res["vectors_valid"] = vectors_valid
        print(f"[kg] saved rank-one vectors -> {vpath} (valid={vectors_valid})", flush=True)

    # ---- EGL sidecar (written BEFORE the main json commit marker) ----
    if args.egl:
        res["EGL"] = summarize_egl(egl_records, dataset=args.dataset)
        sidecar = os.path.splitext(args.out)[0] + ".egl.json"
        write_egl_sidecar(sidecar, {
            "records": egl_records,
            "summary": res["EGL"],
            "model": args.model, "editor": args.editor, "dataset": args.dataset,
            "layer": layer, "seed": args.seed, "steps": args.steps,
            # carried so aggregation can filter on projector provenance (C4 circularity lesson)
            "alpha_proj_source": (args.alpha_proj_source if args.editor == "alpha" else None),
            "egl_max_neighborhood": args.egl_max_neighborhood,
            "egl_max_paraphrase": args.egl_max_paraphrase,
        })
        res["egl_sidecar"] = sidecar
        print(f"[kg] saved EGL sidecar -> {sidecar}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=2)
    os.replace(tmp, args.out)  # atomic: a crash mid-write never leaves a truncated "done" file
    print("\n=== KILL-GATE RESULT ===", flush=True)
    print(json.dumps(res, indent=2), flush=True)
    print(f"[kg] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
