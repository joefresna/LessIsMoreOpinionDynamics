#!/bin/bash
# Run the whole parameter sweep in ONE Slurm job, filling a full standard node.
# Submit with:
#   sbatch runexps_multi.sh
#
# bwUniCluster 3.0 standard CPU nodes provide 96 physical cores; we request one full node.
# (partition "cpu", 96 tasks per node)
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --job-name=LIME_sweep
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# EXPECTED_ARGS=2
# if [ $# -lt ${EXPECTED_ARGS} ]; then
# 	echo "This script submits the full set of experiments for N repetitions."
# 	echo "Usage $0 <start> <end> <communication_range> <options_size>"
# 	echo $'\t'"[MANDATORY] <start> is the index of the first experiment"
# 	echo $'\t'"[MANDATORY] <end> is a number of experiment files to generate"
# 	exit
# fi


# Run from submission directory (robust relative paths if needed later)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Stage hot I/O (venv + outputs/configs) to local SSD ($TMPDIR) to reduce HOME metadata load.
: "${TMPDIR:=/tmp}"
RUN_ROOT="${TMPDIR}/reina_${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"

TASKS_FILE="${RUN_ROOT}/tasks.txt"
VENV_SRC="${HOME}/venvs/lime"
VENV_DST="${RUN_ROOT}/venv"
rsync -a "${VENV_SRC}/" "${VENV_DST}/"
PYTHON="${VENV_DST}/bin/python"

"${PYTHON}" -c "import networkx; print('networkx ok', networkx.__version__)"

# Path variables    
PROJECT_DIR_NAME=${HOME}/Reina/LessIsMoreOpinionDynamics
EXEC_FILE=${PROJECT_DIR_NAME}/BestOfNOnRewiringNets/bestOf2OnGraphs.py
TEMPLATE_CONFIG=${PROJECT_DIR_NAME}/config/bestOf2.template.config
RESULTS_SET="setRW2"

# Stage code + template to TMPDIR to avoid repeated HOME reads
EXEC_LOCAL="${RUN_ROOT}/bestOf2OnGraphs.py"
TEMPLATE_LOCAL="${RUN_ROOT}/bestOf2.template.config"
cp "${EXEC_FILE}" "${EXEC_LOCAL}"
cp "${TEMPLATE_CONFIG}" "${TEMPLATE_LOCAL}"

# keep python caches off $HOME too
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/pycache"


# write bursty files to local SSD
OUT_DIR=${RUN_ROOT}/results/${RESULTS_SET}
CONFIG_DIR=${RUN_ROOT}/batch_config
RUN_LOG=${RUN_ROOT}/run.log

LOG_DIR=${RUN_ROOT}/logs
mkdir -p "${LOG_DIR}"
: > "${TASKS_FILE}"

mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/evo"
mkdir -p "${CONFIG_DIR}"

# Make sure each Python process stays single-threaded (no accidental BLAS oversubscription).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

test -f "${EXEC_LOCAL}"
test -f "${TEMPLATE_LOCAL}"

# Experiments variables
RND_SEED=123456
NUM_REP=100
MAX_TIME=10000
#QUALITY_LOW=0.8
#DISCOVERY_PROBABILITY=0.02
QUALITY_LOW_VALUES=( 0.7 0.8 0.9 )
#QUALITY_LOW_VALUES=( 0.8 )
DISCOVERY_PROBABILITY_VALUES=( 0.01 0.02 )
#DISCOVERY_PROBABILITY_VALUES=( 0.01 )
#DIGEST_TIME_VALUES=$(seq 0 10 50 )
DIGEST_TIME_VALUES=$(seq 0 5 100 )
#NET_SIZE=100
NET_SIZE_VALUES=(50 100)
# possible netTypes are [ 'full' , 'erdos-renyi', 'barabasi-albert', 'rgg', 'rgg-fixed-degree' ]
#NET_TYPE_VALUES=('erdos-renyi' 'barabasi-albert' 'rgg-fixed-degree')
NET_TYPE_VALUES=('erdos-renyi')
#NET_PARAM_VALUES=$(seq 0.05 0.05 0.6 )
#NET_PARAM_VALUES=$(seq 0.005 0.01 0.05 )
NET_PARAM_VALUES=( 0.035 )
#NET_PARAM_VALUES=$(seq 3 2 11)
#NET_PARAM_VALUES=( 1 )
REWIRING_PROBABILITY_VALUES=$(seq 0.0 0.05 1.0 )
#REWIRING_PROBABILITY_VALUES=$(seq 0.0 0.1 1.0 )

COUNT=0

for DISCOVERY_PROBABILITY in ${DISCOVERY_PROBABILITY_VALUES[*]};   
do
	for QUALITY_LOW in ${QUALITY_LOW_VALUES[*]};   
	do
		for NET_SIZE in ${NET_SIZE_VALUES[*]};   
		do
			for NET_TYPE in ${NET_TYPE_VALUES[*]};   
			do
				for NET_PARAM in ${NET_PARAM_VALUES[*]};                                                                
			    do  
					for DIGEST_TIME in ${DIGEST_TIME_VALUES[*]};
					do				
						for REWIRING_PROBABILITY in ${REWIRING_PROBABILITY_VALUES[*]};
						do				
							JOB_PARAM=LIME_N-${NET_SIZE}_mt-${MAX_TIME}_q-${QUALITY_LOW}_dp-${DISCOVERY_PROBABILITY}_net-${NET_TYPE}_p-${NET_PARAM}_dt-${DIGEST_TIME}_rw-${REWIRING_PROBABILITY}
							
							CONFIG_FILE=${CONFIG_DIR}/${JOB_PARAM}.config
							OUT_FILE=${OUT_DIR}/${JOB_PARAM}.txt
							
						    sed -e "s|RND_SEED|${RND_SEED}|"         \
						        -e "s|NUM_REP|${NUM_REP}|"           \
						        -e "s|MAX_TIME|${MAX_TIME}|"         \
						        -e "s|DIGEST_TIME|${DIGEST_TIME}|"   \
						        -e "s|NET_SIZE|${NET_SIZE}|"         \
						        -e "s|NET_TYPE|${NET_TYPE}|"         \
						        -e "s|NET_PARAM|${NET_PARAM}|"       \
						        -e "s|QUALITY_LOW|${QUALITY_LOW}|"       \
						        -e "s|DISCOVERY_PROBABILITY|${DISCOVERY_PROBABILITY}|"   \
						        -e "s|REWIRING_PROBABILITY|${REWIRING_PROBABILITY}|"     \
						        -e "s|OUT_FILE|${OUT_FILE}|"         \
									"${TEMPLATE_LOCAL}" > "${CONFIG_FILE}"
							echo "${CONFIG_FILE}" >> "${TASKS_FILE}"
							COUNT=$((COUNT+1))
						done
					done 
				done
			done
		done
	done
done

echo "Prepared ${COUNT} simulations. Launching worker pool..."

export PYTHON EXEC_LOCAL TASKS_FILE LOG_DIR
SRUN_RC=0
set +e
srun -N1 -n "${SLURM_NTASKS}" -c1 --cpu-bind=cores --hint=nomultithread \
  bash -lc '
    set -euo pipefail
    rank=${SLURM_PROCID}
    ntasks=${SLURM_NTASKS}
    i=0
    while IFS= read -r cfg; do
      if (( i % ntasks == rank )); then
        "$PYTHON" -u "$EXEC_LOCAL" "$cfg" >> "$LOG_DIR/run.${rank}.log" 2>&1
      fi
      ((i+=1))
    done < "$TASKS_FILE"
  '
SRUN_RC=$?
set -e
cat "${LOG_DIR}"/run.*.log > "${RUN_LOG}" || true

echo "Finished " $COUNT " simulations"

# Copy back results as a single archive (keeps HOME metadata low).
mkdir -p "${PROJECT_DIR_NAME}/results_archives"
tar -C "${RUN_ROOT}" -czf "${PROJECT_DIR_NAME}/results_archives/LIME_sweep_${SLURM_JOB_ID}.tar.gz" results/"${RESULTS_SET}" run.log tasks.txt
echo "Wrote archive: ${PROJECT_DIR_NAME}/results_archives/LIME_sweep_${SLURM_JOB_ID}.tar.gz"

exit "${SRUN_RC}"
