#!/bin/bash
#ARGUMENTS: ./run-benchmark.sh [Output File name][Min_CORE][-g (for gpu) | -c confidence level]

#----SETUP DEFAULT CONFIGURATION
OUTPUT_FILE=./statistics/cpu_times_fixed_num_2.csv
MIN_CORE=6
WL=0
#EXEC=./caliper
EXEC=./gpu_exec
EXTRA_ARG=-g
#EXTRA_ARG=-c\ 0.95\ 0.005
#---ARG0: OUTPUT FILE---------------
if [ -z "$1" ]
    then
        echo "Default Output"
    else
        OUTPUT_FILE=./statistics/$1
fi
echo "OUTPUT: $OUTPUT_FILE"
#---ARG1: MINCORE---------------
if [ -z "$2" ]
    then
        echo "Default MinCore"
    else
        MIN_CORE=$2
fi
echo "MINCORE: $MIN_CORE"
#---ARG2: EXEC FILE---------------
#TODO GPU ARG


#--------------ACTUAL SCRIPT----------------------------------------------
> $OUTPUT_FILE #Empty file content
echo "ROWS,COLS,NUM OF CORES,MIN_CORE,WORKLOAD,NUM_OF_TEST,CONFIDENCE,EXECUTION TIME,MTTF[Hours],MTTF[Years]" >> $OUTPUT_FILE

for num_cores in {4..7}
do
    #Set workload to $MIN_CORE / ($num_cores * $num_cores) 
    #This grant that distributed load for this configuration never goes above 1
    $EXEC $num_cores $num_cores $MIN_CORE $(echo "scale=4; $MIN_CORE / ($num_cores * $num_cores)" | bc) $EXTRA_ARG >> $OUTPUT_FILE 
done
for num_cores in 10 11 15 16 17 18 19 20 22 25 30 35 40 42
do
    #Set workload to $MIN_CORE / ($num_cores * $num_cores) 
    #This grant that distributed load for this configuration never goes above 1
    $EXEC $num_cores $num_cores $MIN_CORE $(echo "scale=4; $MIN_CORE / ($num_cores * $num_cores)" | bc) $EXTRA_ARG  >> $OUTPUT_FILE 
done