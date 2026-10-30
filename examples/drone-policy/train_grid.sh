#!/bin/bash
# Training grids, at most JOBS runs at a time (8 XLA devices each), every name x SEEDS (0 1 2);
# a NAMES entry NAME@0,1 runs only those seeds.
#   GRID=t2 (default): main / A1 / A2, from-scratch PPO with reward v2 (RUNS ~/drone-policy-runs)
#   GRID=t3: S-main (from scratch, reward v3) and R-main (residual over B0-DR); R-A1 and R-A2
#            are defined too (RUNS ~/drone-policy-runs/t3)
#   GRID=t4: S-main ablations and budget (RUNS ~/drone-policy-runs/t4): S-main8k (8000 iters),
#            S-main8k-intol1 (8000 iters, in-tol weight 1.0, seed 0 only), S-A1 (mekf6,
#            --nominal) and S-A2 (truth, DR + throw curriculum); all other flags are S-main's
#
#   GRID=t3 setsid nohup examples/drone-policy/train_grid.sh > ~/drone-policy-runs/t3/grid.log \
#     2>&1 < /dev/null &
#
# DRY=1 prints every run's command line instead of training.
#
# Each run writes RUNS/<name>_s<seed>/ (cmd.txt, stdout.log, stderr.log and train.py's outputs) and
# then a `done` marker holding the exit status. Runs whose marker reads 0 are skipped on a restart;
# a failed run is retried once, keeping the first attempt's logs as *.attempt1.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$HOME/elodin-head-venv/bin/python}"
GRID="${GRID:-t2}"
JOBS="${JOBS:-2}"
SEEDS="${SEEDS:-0 1 2}"
case "$GRID" in
  t2)
    RUNS="${RUNS:-$HOME/drone-policy-runs}"
    NAMES="${NAMES:-main A1 A2}"
    ;;
  t3)
    RUNS="${RUNS:-$HOME/drone-policy-runs/t3}"
    NAMES="${NAMES:-S-main R-main}"
    ;;
  t4)
    RUNS="${RUNS:-$HOME/drone-policy-runs/t4}"
    NAMES="${NAMES:-S-main8k S-main8k-intol1@0 S-A1 S-A2}"
    ;;
  *) echo "unknown GRID $GRID" >&2; exit 1 ;;
esac

T2_COMMON=(
  --curriculum-iters 300 --iters 1500 --num-envs 1024 --steps 32 --devices 8
  --eval-every 100 --eval-envs 1024
  --final-eval 1,1 0,1 --final-eval-envs 2048 --final-eval-seed 10007
  --lr 3e-4 --lr-end 3e-5 --ent-coef 1e-3 --init-std 0.1,0.1,0.1,0.2 --in-tol-weight 0.25
)
T3_COMMON=(
  --num-envs 1024 --steps 32 --devices 8 --eval-every 100 --eval-envs 1024
  --final-eval 1,1 0.5,1 0,1 --final-eval-envs 4096 --final-eval-seed 20011
)
S_DR=(--estimator mekf6 --dr curriculum --throw curriculum)
R_KNOBS=(
  --residual dr --residual-scale 0.3 --init-std 0.15 --actor-head-gain 0.01 --actor-warmup 20
  --lr 3e-4 --lr-end 3e-5 --ent-coef 1e-3 --critic-head-gain 1.0 --curriculum-iters 100
  --iters 1500
)

# S-main with ITERS and IN_TOL; the remaining arguments pick the estimator and schedules
s_flags() {
  local iters="$1" in_tol="$2"
  shift 2
  echo "$@" --curriculum-iters 300 --iters "$iters" "${T3_COMMON[@]}" --lr 1.5e-4 --lr-end 3e-5 \
    --ent-coef 1e-3 --init-std 0.1,0.1,0.1,0.2 --in-tol-weight "$in_tol" --critic-head-gain 0.01 \
    --actor-warmup 20
}

run_flags() {
  case "$1" in
    main) echo --estimator mekf6 --dr curriculum --throw curriculum "${T2_COMMON[@]}" ;;
    A1) echo --estimator mekf6 --nominal "${T2_COMMON[@]}" ;;
    A2) echo --estimator truth --dr curriculum --throw curriculum "${T2_COMMON[@]}" ;;
    S-main) s_flags 4000 0.25 "${S_DR[@]}" ;;
    S-main8k) s_flags 8000 0.25 "${S_DR[@]}" ;;
    S-main8k-intol1) s_flags 8000 1.0 "${S_DR[@]}" ;;
    S-A1) s_flags 4000 0.25 --estimator mekf6 --nominal ;;
    S-A2) s_flags 4000 0.25 --estimator truth --dr curriculum --throw curriculum ;;
    R-main) echo --estimator mekf6 --dr curriculum --throw curriculum "${T3_COMMON[@]}" "${R_KNOBS[@]}" ;;
    R-A1) echo --estimator mekf6 --nominal "${T3_COMMON[@]}" "${R_KNOBS[@]}" ;;
    R-A2) echo --estimator truth --dr curriculum --throw curriculum "${T3_COMMON[@]}" "${R_KNOBS[@]}" ;;
    *) echo "unknown run $1" >&2; return 1 ;;
  esac
}

attempt() {
  local out="$1"
  shift
  (cd "$HERE" && /usr/bin/time -v "$PYTHON" train.py --out "$out" "$@" \
    > "$out/stdout.log" 2> "$out/stderr.log")
}

run() {
  local name="$1" seed="$2" out status
  out="$RUNS/${name}_s${seed}"
  if [[ "$(cat "$out/done" 2> /dev/null)" == 0 ]]; then
    echo "$(date -Is) skip $name s$seed"
    return 0
  fi
  local flags
  flags="$(run_flags "$name")" || return 1
  local args=($flags --seed "$seed")
  if [[ -n "${DRY:-}" ]]; then
    echo "$name s$seed -> $out: ${args[*]}"
    return 0
  fi
  mkdir -p "$out"
  rm -f "$out/done"
  echo "${args[*]}" > "$out/cmd.txt"
  echo "$(date -Is) start $name s$seed"
  attempt "$out" "${args[@]}"
  status=$?
  if ((status != 0)); then
    echo "$(date -Is) fail $name s$seed status $status, retrying once"
    for f in stdout.log stderr.log log.jsonl final_eval.json; do
      if [[ -e "$out/$f" ]]; then mv "$out/$f" "$out/$f.attempt1"; fi
    done
    attempt "$out" "${args[@]}"
    status=$?
  fi
  echo "$status" > "$out/done"
  echo "$(date -Is) end $name s$seed status $status"
}

[[ -n "${DRY:-}" ]] || mkdir -p "$RUNS"
# names outer: the longest runs (listed first) start first
for entry in $NAMES; do
  name="${entry%@*}"
  seeds="$SEEDS"
  if [[ "$entry" == *@* ]]; then seeds="${entry#*@}"; fi
  for seed in ${seeds//,/ }; do
    while (($(jobs -rp | wc -l) >= JOBS)); do wait -n; done
    run "$name" "$seed" &
  done
done
wait
echo "$(date -Is) grid done"
