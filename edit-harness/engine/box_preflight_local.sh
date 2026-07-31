#!/usr/bin/env bash
# Local pre-rental audit: proves every laptop-side prerequisite exists before renting a box.
set -u
WAVE="${1:-}"; ROOT="$(cd "$(dirname "$0")/.." && pwd)"; WS="$(cd "$ROOT/.." && pwd)"
FAILED=0
fail(){ echo "FAIL $*"; FAILED=1; }
ok(){ echo "OK   $*"; }
need_file(){ [ -f "$1" ] && ok "$1" || fail "missing $1"; }
need_ratified(){ need_file "$1"; [ -f "$1" ] && grep -qx 'STATUS: RATIFIED' "$1" && ok "ratified $1" || fail "not ratified $1"; }
case "$WAVE" in
 deletion-wave1)
  prereg="$WS/docs/plans/PREREG-DELETION-PREDICTOR-2026-07-26.md"
  driver="$ROOT/run_deletion_wave1.sh"
  manifests=("$ROOT/engine/manifests/deletion_wave1_card0.txt" "$ROOT/engine/manifests/deletion_wave1_card1.txt")
  receipts=(DELETION_PHASEL_GD1_PASS.ok DELETION_PHASEL_GD2_PASS.ok DELETION_PHASEL_TEXT_PASS.ok)
  ;;
 deletion-wave2)
  prereg="$WS/docs/plans/PREREG-DELETION-PREDICTOR-2026-07-26.md"
  driver="$ROOT/run_deletion_wave2.sh"; manifests=("$ROOT/engine/manifests/deletion_wave2.txt")
  receipts=(DELETION_WAVE1_GD3_PASS.ok)
  ;;
 paperb-curve)
  prereg="$WS/docs/plans/PREREG-PAPERB-CURVE-2026-07-26.md"
  driver="$ROOT/run_paperb_curve_cloud.sh"; manifests=("$ROOT/engine/manifests/paperb_curve_cloud.txt")
  receipts=(PAPERB_CURVE_GS3_PASS.ok)
  ;;
 d2-prospective)
  prereg="$WS/docs/plans/PREREG-D2-PROSPECTIVE-2026-07-26.md"
  driver="$ROOT/run_d2_prospective_cloud.sh"; manifests=("$ROOT/engine/manifests/d2_prospective_cloud.txt")
  receipts=()
  ;;
 *) echo "usage: $0 {deletion-wave1|deletion-wave2|paperb-curve|d2-prospective}" >&2; exit 2 ;;
esac
need_ratified "$prereg"
need_file "$driver"
need_file "$ROOT/engine/box_sync_up.sh"
need_file "$ROOT/engine/box_prepare_wave.sh"
need_file "$ROOT/engine/box_launch_wave.sh"
need_file "$ROOT/engine/box_pull_down.sh"
need_file "$ROOT/data/counterfact.json"
[ "$(sha256sum "$ROOT/data/counterfact.json" | cut -d' ' -f1)" = d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f ] \
 && ok "counterfact sha256" || fail "counterfact sha256 mismatch"
for r in "${receipts[@]}"; do need_file "$ROOT/engine/$r"; done
for m in "${manifests[@]}"; do
 need_file "$m"
 if [ -f "$m" ] && grep -vE '^\s*(#|$)' "$m" | grep -qE '[*?\[]'; then fail "manifest contains glob: $m"; else ok "manifest exact paths: $m"; fi
done
if [ "$WAVE" = deletion-wave1 ]; then
 for tag in gate_mistral7b_rome_cf_L24 gate_llama8b_rome_cf_L24; do
  for s in 0 1 2; do need_file "$ROOT/results/matrices/${tag}_s${s}.npz"; done
 done
fi
bash -n "$driver" "$ROOT/engine/box_sync_up.sh" "$ROOT/engine/box_prepare_wave.sh" \
 "$ROOT/engine/box_launch_wave.sh" "$ROOT/engine/box_pull_down.sh" || fail "shell syntax"
python -m py_compile "$ROOT/experiments/killgate_keygeom.py" "$ROOT/experiments/quant_survival_phase1.py" \
 "$ROOT/experiments/prospective_admission.py" || fail "python compile"
if [ "$FAILED" -eq 0 ]; then
 echo "READY_TO_RENT wave=$WAVE"
 exit 0
fi
echo "BLOCKED_LOCAL_PREREQ wave=$WAVE"
exit 3
