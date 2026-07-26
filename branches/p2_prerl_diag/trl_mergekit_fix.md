# trl 0.24.0 × transformers 5.x — the mergekit import break, and the opt-in source patch

**Status: DOCUMENTATION ONLY.** Do **not** apply any of this to the installed `trl`
in the shared `dl` env. Apply it **only** inside a `dl-rl` clone (see `SETUP.md`).
Do **not** `pip install mergekit` — that pulls a resolver that downgrades
`accelerate` / `huggingface_hub` / `safetensors` / `pydantic` in `dl`, breaking the
other projects that share it.

---

## 1. Symptom (verified live 2026-06-30, env `dl`)

```
$ conda run -n dl python3 -c "from trl import GRPOTrainer"
RuntimeError: Failed to import trl.trainer.grpo_trainer because of the following error:
No module named 'mergekit'
```

Same for `from trl import DPOTrainer`. `mergekit` is genuinely **not installed**
(`importlib.util.find_spec('mergekit')` → `None`), so a guarded import should
skip it — but it doesn't.

## 2. Actual root cause (not "mergekit is missing")

`trl` **is** guarded; the guard is just broken by a transformers 5.x API change.

`site-packages/trl/import_utils.py:35`

```python
_mergekit_available = _is_package_available("mergekit")
```

In transformers 5.x, `transformers.utils.import_utils._is_package_available("mergekit")`
returns a **2-tuple** `(False, None)` even when called **without**
`return_version=True`. `trl` stores that tuple as-is, so:

```python
>>> from trl.import_utils import is_mergekit_available
>>> is_mergekit_available()
(False, None)          # a NON-EMPTY tuple  ->  TRUTHY
```

Then `site-packages/trl/mergekit_utils.py:21`

```python
if is_mergekit_available():                       # (False, None) is truthy!
    from mergekit.config import MergeConfiguration # -> ModuleNotFoundError
    from mergekit.merge import MergeOptions, run_merge
```

fires the import and dies. The import chain that drags this into every trainer:

```
trl/trainer/grpo_trainer.py:56  from .callbacks import SyncRefModelCallback
trl/trainer/dpo_trainer.py:58   from .callbacks import SyncRefModelCallback
      -> trl/trainer/callbacks.py:40  from ..mergekit_utils import MergeConfig, merge_models, upload_model_to_hf   (UNCONDITIONAL)
            -> trl/mergekit_utils.py:21  if is_mergekit_available(): from mergekit.config import ...
```

So GRPOTrainer/DPOTrainer are collateral damage of a truthy-tuple guard in an
unrelated model-merging utility.

## 3. The patch (opt-in, `dl-rl` only)

Two independent one-edit fixes. **Fix A** is the minimal, most faithful "lazy /
guarded" patch on the file the trainers pull in; **Fix B** is the true one-line
root-cause fix. Applying **either** makes `GRPOTrainer`/`DPOTrainer` import.
Recommended: apply **Fix A** (surgical, does not touch shared availability flags).

### Fix A — make the mergekit import lazy + correctly guarded

File: `site-packages/trl/mergekit_utils.py`

Offending top-level block (lines ~21–23):

```python
if is_mergekit_available():
    from mergekit.config import MergeConfiguration
    from mergekit.merge import MergeOptions, run_merge
```

Guarded replacement (delete the top-level import; import lazily where used, and
compare against `True` so a truthy `(False, None)` tuple can't slip through):

```python
# --- P2 dl-rl patch: never import mergekit at module load. The transformers 5.x
# --- _is_package_available() returns a truthy (bool, ver) tuple, so the old
# --- top-level `if is_mergekit_available():` fired even when mergekit is absent,
# --- breaking GRPOTrainer/DPOTrainer import via callbacks.py. Import lazily.
def _load_mergekit():
    """Import mergekit only when a merge is actually requested."""
    avail = is_mergekit_available()
    if avail is not True:          # tolerate both bool and (bool, ver) tuple
        avail = bool(avail[0]) if isinstance(avail, tuple) else bool(avail)
    if not avail:
        raise ImportError(
            "mergekit is required for model merging. Install it in an ISOLATED "
            "env (not shared `dl`): `pip install mergekit`."
        )
    from mergekit.config import MergeConfiguration
    from mergekit.merge import MergeOptions, run_merge
    return MergeConfiguration, MergeOptions, run_merge
```

Then, inside the functions that actually merge (`merge_models`, and the
`MergeConfig.create`/validation paths that reference `MergeConfiguration`),
replace the module-global names with a local `_load_mergekit()` call, e.g.:

```python
def merge_models(config, out_path):
    MergeConfiguration, MergeOptions, run_merge = _load_mergekit()   # lazy
    ...
```

No other module imports `MergeConfiguration`/`run_merge` at top level, so this is
self-contained. GRPO/DPO never call `_load_mergekit`, so they now import cleanly.

### Fix B — one-line root-cause fix (unpack the tuple)

File: `site-packages/trl/import_utils.py:35`

```python
# before
_mergekit_available = _is_package_available("mergekit")
# after  (transformers 5.x always returns (bool, version); take the bool)
_mergekit_available = _is_package_available("mergekit", return_version=True)[0]
```

This makes `is_mergekit_available()` return a real `False`, so the existing
`if is_mergekit_available():` guard in `mergekit_utils.py` short-circuits and the
mergekit import never runs. (Note: the same truthy-tuple footgun exists anywhere
`_is_package_available` is called without `return_version=True` — mergekit is the
one that breaks the trainer path.)

## 4. Applying the patch as a diff (reproducible, `dl-rl` only)

Save as `mergekit_lazy.patch` and `git apply`/`patch -p0` against the site-packages
tree of the **clone** (paths shown for Fix B, the one-liner — easiest to automate):

```diff
--- a/trl/import_utils.py
+++ b/trl/import_utils.py
@@ -32,7 +32,7 @@
-_mergekit_available = _is_package_available("mergekit")
+_mergekit_available = _is_package_available("mergekit", return_version=True)[0]
```

`SETUP.md` shows the exact `sed`/`python` in-place edit + a verification import.

## 4b. APPLIED 2026-07-11 — Fix B alone is NOT sufficient; 12 sites patched

Applying Fix B live surfaced the predicted follow-on: after mergekit, the SAME
truthy-tuple guard fired next on `llm_blender` (dragged in via
`trl/trainer/utils.py` → judges). Every `_is_package_available("X")` call without
`return_version=True` in `import_utils.py` carries the bug — 12 total (mergekit,
deepspeed, fastapi, joblib, llm_blender, math_verify, pydantic, requests, unsloth,
uvicorn, vllm_ascend, weave). All 12 were patched in `dl-rl` with the uniform
transformation (append `, return_version=True)[0]`), applied by regex:

```python
pat = re.compile(r'^(_\w+_available = _is_package_available\("([\w.]+)")\)$', re.M)
new, n = pat.subn(r'\1, return_version=True)[0]', src)
```

The pristine pre-patch file is preserved at `import_utils.py.bak` (rollback per §
"Rollback" below). Verified after: `from trl import GRPOTrainer, DPOTrainer` → OK.

## 5. Verify (inside `dl-rl` only)

```
$ conda run -n dl-rl python3 -c "from trl import GRPOTrainer, DPOTrainer; print('OK', GRPOTrainer, DPOTrainer)"
OK <class 'trl.trainer.grpo_trainer.GRPOTrainer'> <class 'trl.trainer.dpo_trainer.DPOTrainer'>
```

(unsloth's separate failure — `transformers 5.x` too new — is out of scope for
P2; the LoRA GRPO scaffold in `grpo_config.py` uses `peft` directly and does not
require unsloth.)
