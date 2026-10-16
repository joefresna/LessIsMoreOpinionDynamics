#!/bin/bash
# =====================================================================================================
# Simulations behind the figures of "Dynamic networks can trap collective decisions".
# Runs on one full CPU node (96 workers), with the venv and the outputs staged on the node-local SSD,
# and returns the results as one tar.gz archive.
#   - The first argument selects the parameter set:  main | extended | test   (see "Parameter sets").
#   - The simulator is run from a copy with PLOT_EVO = False (no per-run trajectory files). The random
#     stream does not depend on PLOT_EVO, so the results are the same as with trajectory files.
#   - Tasks are distributed dynamically (each worker takes the next task from a shared counter), longest first.
#   - If the job approaches its time limit, the finished results are archived (Slurm sends USR1 30 min before
#     the end); a resubmission with DONE_DIR=<extracted results> skips them.
#
# Submit:
#   sbatch runexps_paper.sh main                          # full (rho, delta) sweep, 10,584 combinations, T = 1e4
#   sbatch --time=24:00:00 runexps_paper.sh extended      # extended-horizon runs (T = 1e6 and 1e7)
#   sbatch --time=00:30:00 runexps_paper.sh test          # short test of the whole pipeline
# Optional environment variables (pass with  sbatch --export=ALL,VAR=value ...):
#   DRY_RUN=1          only write tasks.txt and print the number of simulations, run nothing
#   DONE_DIR=<dir>     skip every combination whose result file already exists in <dir> (resume)
#
# Output: ${PROJECT_DIR_NAME}/results_archives/LIME_<sweep>_<jobid>.tar.gz containing
#   results/<set>/LIME_*.txt (one convergence time per line, maxTime if not converged), run.log, tasks.txt,
#   and aggregated_<set>.csv (one row per combination; see analysis/aggregate_results.py), which is the input
#   of the plotting scripts in analysis/.
# =====================================================================================================
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --job-name=LIME_paper
#SBATCH --output=slurm-%x-%j.out
#SBATCH --signal=B:USR1@1800

set -euo pipefail

SWEEP="${1:-main}"
DRY_RUN="${DRY_RUN:-0}"
DONE_DIR="${DONE_DIR:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
JOBID="${SLURM_JOB_ID:-local$$}"
NTASKS="${SLURM_NTASKS:-4}"

# ----------------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------------
: "${TMPDIR:=/tmp}"
RUN_ROOT="${TMPDIR}/reina_${JOBID}"
mkdir -p "${RUN_ROOT}"
PROJECT_DIR_NAME=${HOME}/LessIsMoreOpinionDynamics
EXEC_FILE=${PROJECT_DIR_NAME}/BestOfNOnRewiringNets/bestOf2OnGraphs.py
TEMPLATE_CONFIG=${PROJECT_DIR_NAME}/config/bestOf2.template.config
VENV_SRC="${HOME}/venvs/lime"

if [[ "${DRY_RUN}" != "1" ]]; then
  VENV_DST="${RUN_ROOT}/venv"
  rsync -a "${VENV_SRC}/" "${VENV_DST}/"
  PYTHON="${VENV_DST}/bin/python"
  "${PYTHON}" -c "import networkx; print('networkx ok', networkx.__version__)"
else
  PYTHON=python3
fi

# Patched copy of the simulator: identical except PLOT_EVO = False
EXEC_LOCAL="${RUN_ROOT}/bestOf2OnGraphs.py"
TEMPLATE_LOCAL="${RUN_ROOT}/bestOf2.template.config"
sed -e 's/^PLOT_EVO = True/PLOT_EVO = False/' "${EXEC_FILE}" > "${EXEC_LOCAL}"
cp "${TEMPLATE_CONFIG}" "${TEMPLATE_LOCAL}"
grep -q '^PLOT_EVO = False' "${EXEC_LOCAL}" || { echo "ERROR: could not switch off PLOT_EVO in ${EXEC_FILE}"; exit 1; }
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/pycache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

TASKS_FILE="${RUN_ROOT}/tasks.txt"
CONFIG_DIR="${RUN_ROOT}/batch_config"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${CONFIG_DIR}" "${LOG_DIR}"
: > "${TASKS_FILE}"

COUNT=0
SKIPPED=0
# add_task SET NUM_REP MAX_TIME SEED N Q PD PK DT RW [SUFFIX]
add_task () {
  local set=$1 nrep=$2 tmax=$3 seed=$4 n=$5 q=$6 pd=$7 pk=$8 dt=$9 rw=${10} suffix=${11:-}
  local out_dir="${RUN_ROOT}/results/${set}"
  mkdir -p "${out_dir}"
  local job="LIME_N-${n}_mt-${tmax}_q-${q}_dp-${pd}_net-erdos-renyi_p-${pk}_dt-${dt}_rwp-${rw}${suffix}"
  if [[ -n "${DONE_DIR}" && -s "${DONE_DIR}/${job}.txt" ]]; then SKIPPED=$((SKIPPED+1)); return; fi
  local cfg="${CONFIG_DIR}/${job}.config"
  sed -e "s|RND_SEED|${seed}|"            -e "s|NUM_REP|${nrep}|"   -e "s|MAX_TIME|${tmax}|" \
      -e "s|DELIB_TIME|${dt}|"            -e "s|NET_SIZE|${n}|"     -e "s|NET_TYPE|erdos-renyi|" \
      -e "s|NET_PARAM|${pk}|"             -e "s|QUALITY_LOW|${q}|"  -e "s|DISCOVERY_PROBABILITY|${pd}|" \
      -e "s|REWIRING_PROBABILITY|${rw}|"  -e "s|OUT_FILE|${out_dir}/${job}.txt|" \
      "${TEMPLATE_LOCAL}" > "${cfg}"
  echo "${cfg}" >> "${TASKS_FILE}"
  COUNT=$((COUNT+1))
}
fmt () { printf '%g' "$1"; }   # 0.00 -> 0, 0.05 -> 0.05, 1.00 -> 1  (names as expected by plotResults.py)

# ----------------------------------------------------------------------------------------------------
# Parameter sets
# ----------------------------------------------------------------------------------------------------
case "${SWEEP}" in
  main)
    # Full sweep behind Figs. 3 and 5, Supplementary Figs. SF1, SF2 and SF3, STab1.
    # 2 (N) x 3 (q_B) x 2 (p_d) x 2 (p_k) x 21 (delta) x 21 (rho) = 10,584 combinations, 100 runs each.
    # Same seed as the original sweeps (123456). Ordered longest first (N = 50, long delta, fast rewiring).
    RESULTS_SET="setRev-main"
    for N in 50 100; do
      for DT in $(seq 100 -5 0); do
        for RW in $(seq 1.0 -0.05 0.0); do
          for PK in 0.035 0.045; do
            for PD in 0.01 0.02; do
              for Q in 0.7 0.8 0.9; do
                add_task "${RESULTS_SET}" 100 10000 123456 "$N" "$Q" "$PD" "$PK" "$DT" "$(fmt "$RW")"
              done; done; done; done; done; done
    ;;
  extended)
    # Extended-horizon runs (Supplementary Text ST3, Supplementary Table STab2, Supplementary Fig. SF5).
    RESULTS_SET="setRev-extended"
    # (a) T = 1e7, N = 50, q_B = 0.8, p_d = 0.01, delta = 75, rho = 0.9: 40 runs, split into 8 x 5 runs
    #     with different seeds so that they run in parallel (longest tasks, submitted first).
    for K in 1 2 3 4 5 6 7 8; do
      add_task "${RESULTS_SET}" 5 10000000 $((200000 + 7919*K)) 50 0.8 0.01 0.035 75 0.9 "_seed-$((200000 + 7919*K))"
    done
    # (b) STab2 rows at T = 1e6 that are not on the grid below (100 runs each)
    add_task "${RESULTS_SET}" 100 1000000 123456  50 0.8 0.01 0.035  75 0.9
    add_task "${RESULTS_SET}" 100 1000000 123456  50 0.8 0.02 0.035  75 0.9
    add_task "${RESULTS_SET}" 100 1000000 123456  50 0.9 0.02 0.035  50 1
    add_task "${RESULTS_SET}" 100 1000000 123456  50 0.7 0.02 0.035 100 1
    add_task "${RESULTS_SET}" 100 1000000 123456 100 0.8 0.02 0.035 100 1
    # (c) SF5 grid: N = 50, q_B = 0.8, p_d = 0.02, p_k = 0.035, rho = 0..1 step 0.1, delta = 0..100 step 10,
    #     T = 1e6, 100 runs (121 combinations; includes the STab2 row delta = 100, rho = 1)
    for DT in $(seq 100 -10 0); do
      for RW in $(seq 1.0 -0.1 0.0); do
        add_task "${RESULTS_SET}" 100 1000000 123456 50 0.8 0.02 0.035 "$DT" "$(fmt "$RW")"
      done
    done
    ;;
  test)
    # Smoke test: 12 short combinations, 2 runs each (checks venv, patching, scheduling, archive, CSV).
    RESULTS_SET="setRev-test"
    for DT in 0 50; do for RW in 0 0.5 1; do for N in 50 100; do
      add_task "${RESULTS_SET}" 2 5000 123456 "$N" 0.8 0.02 0.035 "$DT" "$RW"
    done; done; done
    ;;
  *) echo "Unknown sweep '${SWEEP}'. Use: main | extended | test"; exit 1 ;;
esac

echo "Sweep '${SWEEP}' -> results/${RESULTS_SET}: ${COUNT} simulations to run, ${SKIPPED} skipped (already in DONE_DIR)."
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN: first tasks:"; head -3 "${TASKS_FILE}"; echo "tasks file: ${TASKS_FILE}"; exit 0
fi

# ----------------------------------------------------------------------------------------------------
# Archive + aggregation (also called on timeout)
# ----------------------------------------------------------------------------------------------------
ARCHIVE_DIR="${PROJECT_DIR_NAME}/results_archives"
mkdir -p "${ARCHIVE_DIR}"
AGG_PY="${RUN_ROOT}/aggregate.py"
cat > "${AGG_PY}" <<'PYEOF'
import os, re, sys, glob, csv
pat = re.compile(r"LIME_N-(?P<N>\d+)_mt-(?P<mt>\d+)_q-(?P<q>[0-9.]+)_dp-(?P<dp>[0-9.]+)_net-(?P<net>[a-z-]+)"
                 r"_p-(?P<p>[0-9.]+)_dt-(?P<dt>\d+)_rwp?-(?P<rw>[0-9.]+)(?:_seed-(?P<seed>\d+))?\.txt$")
src, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(src, "LIME_*.txt"))):
    m = pat.search(os.path.basename(f))
    if not m:
        continue
    t = [int(x) for x in open(f).read().split()]
    mt = int(m["mt"]); ts = sorted(t); n = len(t)
    med = (ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])) if n else float('nan')
    rows.append([m["N"], mt, m["q"], m["dp"], m["net"], m["p"], m["dt"], m["rw"], m["seed"] or "", n,
                 sum(1 for x in t if x < mt), med, ";".join(map(str, t))])
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["N", "mt", "q", "dp", "net", "p", "dt", "rw", "seed", "n_runs", "n_success", "median", "times"])
    w.writerows(rows)
print(out, len(rows), "rows")
PYEOF

archive () {
  local tag=$1
  cat "${LOG_DIR}"/run.*.log > "${RUN_ROOT}/run.log" 2>/dev/null || true
  "${PYTHON}" "${AGG_PY}" "${RUN_ROOT}/results/${RESULTS_SET}" "${RUN_ROOT}/aggregated_${RESULTS_SET}.csv" || true
  local done_n; done_n=$(find "${RUN_ROOT}/results/${RESULTS_SET}" -name 'LIME_*.txt' | wc -l)
  local arch="${ARCHIVE_DIR}/LIME_${SWEEP}_${JOBID}${tag}.tar.gz"
  tar -C "${RUN_ROOT}" -czf "${arch}" "results/${RESULTS_SET}" "aggregated_${RESULTS_SET}.csv" run.log tasks.txt
  echo "Wrote ${arch} (${done_n} of ${COUNT} result files)."
}

on_timeout () {
  echo "Time limit approaching: archiving finished results and stopping workers."
  [[ -n "${SRUN_PID:-}" ]] && kill -TERM "${SRUN_PID}" 2>/dev/null || true
  sleep 20
  archive "_PARTIAL"
  echo "To resume: extract the archive, then"
  echo "  sbatch --export=ALL,DONE_DIR=<path>/results/${RESULTS_SET} runexps_paper.sh ${SWEEP}"
  exit 99
}
trap on_timeout USR1

# ----------------------------------------------------------------------------------------------------
# Worker pool with dynamic scheduling: every worker takes the next line of tasks.txt under a file lock.
# A result file is written by the simulator only when all repetitions of a combination are done, so a
# killed task leaves no (partial) file and is re-run on resubmission.
# ----------------------------------------------------------------------------------------------------
COUNTER="${RUN_ROOT}/next_task"; echo 1 > "${COUNTER}"
export PYTHON EXEC_LOCAL TASKS_FILE LOG_DIR COUNTER
echo "Launching ${NTASKS} workers at $(date)"
set +e
srun -N1 -n "${NTASKS}" -c1 --cpu-bind=cores --hint=nomultithread \
  bash -c '
    set -uo pipefail
    rank=${SLURM_PROCID}
    total=$(wc -l < "$TASKS_FILE")
    while true; do
      i=$(flock "$COUNTER.lock" bash -c "n=\$(cat \"$COUNTER\"); echo \$((n+1)) > \"$COUNTER\"; echo \$n")
      (( i > total )) && break
      cfg=$(sed -n "${i}p" "$TASKS_FILE")
      start=$(date +%s)
      "$PYTHON" -u "$EXEC_LOCAL" "$cfg" >> "$LOG_DIR/run.${rank}.log" 2>&1 || echo "FAILED $cfg" >> "$LOG_DIR/run.${rank}.log"
      echo "task $i/$total rank $rank $(( $(date +%s) - start ))s $(basename "$cfg")" >> "$LOG_DIR/run.${rank}.log"
    done
  ' &
SRUN_PID=$!
wait "${SRUN_PID}"
SRUN_RC=$?
set -e
echo "Workers finished at $(date) (srun exit code ${SRUN_RC})"
grep -h FAILED "${LOG_DIR}"/run.*.log || true

archive ""
exit "${SRUN_RC}"
