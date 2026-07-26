#!/bin/bash
# cloud/selftest.sh — CPU-only simulation test for cloud/*.sh (2026-07-08). No real
# GPUs, no network, no downloads, and NEVER touches the live repo's results/ or
# engine/ (everything runs against copies in throwaway tempdirs). Run this before ever
# pointing the launcher at a paid AutoDL box.
set -u
H="$(cd "$(dirname "$0")/.." && pwd)"
cd "$H" || exit 1
FAIL=0
pass(){ echo "PASS: $1"; }
fail(){ echo "FAIL: $1"; FAIL=1; }

echo "=================================================================="
echo "(a) syntax + shellcheck"
echo "=================================================================="
for f in cloud/*.sh; do
  errf=$(mktemp)
  if bash -n "$f" 2>"$errf"; then
    pass "bash -n $f"
  else
    fail "bash -n $f: $(cat "$errf")"
  fi
  rm -f "$errf"
done
if command -v shellcheck >/dev/null 2>&1; then
  for f in cloud/*.sh; do
    if shellcheck -e SC1091 "$f"; then pass "shellcheck $f"; else fail "shellcheck $f"; fi
  done
else
  echo "NOTE: shellcheck not installed on this box — skipped (bash -n above already ran on every file)"
fi

echo "=================================================================="
echo "(a2) forbidden-pattern + convention checks"
echo "=================================================================="
# Scan actual (non-comment) lines only, and skip this file itself — selftest.sh's own
# guard-check code necessarily contains the literal words "pgrep"/"pkill" to look for them.
badpat=0
for f in cloud/*.sh; do
  [ "$f" = "cloud/selftest.sh" ] && continue
  if sed 's/#.*//' "$f" | grep -nE '\bpgrep\b|\bpkill\b'; then
    fail "found pgrep/pkill command usage in $f — forbidden (self-match deadlock risk)"
    badpat=1
  fi
done
[ "$badpat" -eq 0 ] && pass "no pgrep/pkill command usage in cloud/*.sh"
if grep -q 'while kill -0' cloud/run_cloud_wave.sh; then
  pass "run_cloud_wave.sh waits by PID via kill -0"
else
  fail "run_cloud_wave.sh does not appear to wait by kill -0"
fi
if grep -q 'echo \$! >' cloud/run_cloud_wave.sh; then
  pass "run_cloud_wave.sh writes PID files"
else
  fail "run_cloud_wave.sh does not write PID files"
fi
if grep -qF 'ssh -p ${PORT}' cloud/sync_results.sh && grep -qF -- '-i ${KEY}' cloud/sync_results.sh; then
  pass "sync_results.sh SSH invocation uses -p/-i placeholders"
else
  fail "sync_results.sh SSH invocation missing -p/-i wiring"
fi

echo "=================================================================="
echo "(b) simulate 2 CPU workers via run_cloud_wave.sh (fake driver, CUDA_VISIBLE_DEVICES=\"\")"
echo "=================================================================="
WORK=$(mktemp -d)
mkdir -p "$WORK/cloud"
cp cloud/gpu_idle_lib.sh cloud/selftest_fake_driver.sh cloud/run_cloud_wave.sh "$WORK/cloud/"

(
  cd "$WORK" || exit 1
  DRIVERS="cloud/selftest_fake_driver.sh" SEED_CARD0=1 SEED_CARD1=2 SKIP_IDLE_GATE=1 \
    bash cloud/run_cloud_wave.sh both
  bash cloud/run_cloud_wave.sh wait
)
rc=$?
if [ "$rc" -eq 0 ]; then pass "run_cloud_wave.sh both + wait exited 0"; else fail "run_cloud_wave.sh both + wait exited ${rc}"; fi

OUT0="$WORK/results/selftest/fake_driver_card0_s1.json"
OUT1="$WORK/results/selftest/fake_driver_card1_s2.json"
if [ -f "$OUT0" ] && [ -f "$OUT1" ]; then
  pass "both workers wrote distinct, non-colliding outputs"
  echo "  card0: $(cat "$OUT0")"
  echo "  card1: $(cat "$OUT1")"
else
  fail "expected outputs missing: $OUT0 / $OUT1"
fi
if [ -f "$OUT0" ] && [ -f "$OUT1" ] && \
   grep -q '"card": "0"' "$OUT0" && grep -q '"seed": 1' "$OUT0" && \
   grep -q '"card": "1"' "$OUT1" && grep -q '"seed": 2' "$OUT1"; then
  pass "card/seed assignment matches the shard map (card0=seed1, card1=seed2)"
else
  fail "card/seed assignment in output JSON does not match the shard map"
fi
if [ -f "$WORK/cloud/logs/card0.pid" ] && [ -f "$WORK/cloud/logs/card1.pid" ]; then
  pass "PID files written for both workers"
else
  fail "PID files missing"
fi
echo "  --- card0.log ---"; sed 's/^/  /' "$WORK/cloud/logs/card0.log" 2>/dev/null
echo "  --- card1.log ---"; sed 's/^/  /' "$WORK/cloud/logs/card1.log" 2>/dev/null

echo "=================================================================="
echo "(c) sync_results.sh --dry-run + real merge against a local fake \"remote\""
echo "=================================================================="
FAKE_REMOTE=$(mktemp -d)
LOCAL_TESTDIR=$(mktemp -d)
mkdir -p "$FAKE_REMOTE/results/matrices" "$LOCAL_TESTDIR/cloud" "$LOCAL_TESTDIR/results"
cp cloud/sync_results.sh "$LOCAL_TESTDIR/cloud/"

# a brand-new file only the "remote" has -> must be pulled in
echo '{"seed":1,"src":"remote"}' > "$FAKE_REMOTE/results/gate_llama1b_rome_mquake_L8_s1.json"
echo 'npzdata' > "$FAKE_REMOTE/results/matrices/gate_llama1b_rome_mquake_L8_s1.npz"

# a same-named file on both sides, local NEWER -> --update must NOT overwrite it
echo '{"seed":0,"src":"local","authoritative":true}' > "$LOCAL_TESTDIR/results/gate_llama1b_rome_mquake_L8_s0.json"
echo '{"seed":0,"src":"remote","would_overwrite":true}' > "$FAKE_REMOTE/results/gate_llama1b_rome_mquake_L8_s0.json"
touch -d "2020-01-01" "$FAKE_REMOTE/results/gate_llama1b_rome_mquake_L8_s0.json"

(
  cd "$LOCAL_TESTDIR" || exit 1
  echo "  -- dry-run --"
  bash cloud/sync_results.sh --local-src "$FAKE_REMOTE" --dry-run
  echo "  -- real merge --"
  bash cloud/sync_results.sh --local-src "$FAKE_REMOTE"
)

if [ -f "$LOCAL_TESTDIR/results/gate_llama1b_rome_mquake_L8_s1.json" ] && \
   [ -f "$LOCAL_TESTDIR/results/matrices/gate_llama1b_rome_mquake_L8_s1.npz" ]; then
  pass "new remote-only files (json + matrices/npz) were pulled in"
else
  fail "new remote-only files were not pulled in"
fi
if grep -q '"authoritative":true' "$LOCAL_TESTDIR/results/gate_llama1b_rome_mquake_L8_s0.json" 2>/dev/null; then
  pass "--update preserved the newer local file instead of clobbering it with the older remote copy"
else
  fail "local file was overwritten by an older remote copy — --update is not working"
fi

echo "=================================================================="
if [ "$FAIL" -eq 0 ]; then
  echo "SELFTEST: ALL CHECKS PASSED"
else
  echo "SELFTEST: FAILURES ABOVE"
fi
exit "$FAIL"
