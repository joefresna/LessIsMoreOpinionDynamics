#!/bin/bash

# EXPECTED_ARGS=2
# if [ $# -lt ${EXPECTED_ARGS} ]; then
# 	echo "This script submits the full set of experiments for N repetitions."
# 	echo "Usage $0 <start> <end> <communication_range> <options_size>"
# 	echo $'\t'"[MANDATORY] <start> is the index of the first experiment"
# 	echo $'\t'"[MANDATORY] <end> is a number of experiment files to generate"
# 	exit
# fi

# Path variables    
PROJECT_DIR_NAME=${HOME}/LessIsMoreOpinionDynamics
EXEC_FILE=${PROJECT_DIR_NAME}/BestOfNOnRewiringNets/bestOf2OnGraphs.py
OUT_DIR=${PROJECT_DIR_NAME}/results/setRW1
TEMPLATE_CONFIG=${PROJECT_DIR_NAME}/config/bestOf2.template.config
CONFIG_DIR=${PROJECT_DIR_NAME}/batch_config

mkdir -p ${OUT_DIR}
mkdir -p ${CONFIG_DIR}


# Experiments variables
RND_SEED=123456
NUM_REP=100
MAX_TIME=10000
#QUALITY_LOW=0.8
#DISCOVERY_PROBABILITY=0.02
QUALITY_LOW_VALUES=( 0.7 0.8 0.9 )
#QUALITY_LOW_VALUES=( 0.9 )
DISCOVERY_PROBABILITY_VALUES=( 0.01 0.02 0.05 )
#DISCOVERY_PROBABILITY_VALUES=( 0.02 )
#DIGEST_TIME_VALUES=$(seq 0 10 50 )
DIGEST_TIME_VALUES=$(seq 0 20 100 )
#NET_SIZE=100
NET_SIZE_VALUES=(50 100)
# possible netTypes are [ 'full' , 'erdos-renyi', 'barabasi-albert', 'rgg', 'rgg-fixed-degree' ]
#NET_TYPE_VALUES=('erdos-renyi' 'barabasi-albert' 'rgg-fixed-degree')
NET_TYPE_VALUES=('erdos-renyi')
NET_PARAM_VALUES=$(seq 0.05 0.05 0.6 )
#NET_PARAM_VALUES=(0.05 0.10 0.15 0.20 0.30 0.40)
#NET_PARAM_VALUES=$(seq 3 2 11)
#NET_PARAM_VALUES=( 1 )
REWIRING_PROBABILITY_VALUES=$(seq 0.05 0.05 1.0 )

COUNT=1

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
						JOB_PARAM=LIME_N-${NET_SIZE}_mt-${MAX_TIME}_q-${QUALITY_LOW}_dp-${DISCOVERY_PROBABILITY}_net-${NET_TYPE}_p-${NET_PARAM}_dt-${DIGEST_TIME}
						
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
					        -e "s|DISCOVERY_PROBABILITY|${DISCOVERY_PROBABILITY}|"       \
					        -e "s|OUT_FILE|${OUT_FILE}|"         \
								${TEMPLATE_CONFIG} > ${CONFIG_FILE}
			
						COMMAND="sbatch run_job.sh ${EXEC_FILE} ${CONFIG_FILE}"
						#COMMAND="python3 ${EXEC_FILE} ${CONFIG_FILE}"
						#echo ${COMMAND}
						while ! ${COMMAND}
						do
							sleep 3
						done
						
						((COUNT++))
					done 
				done
			done
		done
	done
done

echo "Submitted " $COUNT " jobs"

