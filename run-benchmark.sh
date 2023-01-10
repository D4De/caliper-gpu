

#!/bin/bash
#ARGUMENTS: ./run-benchmark.sh [Output File name][Min_CORE][-g (for gpu) | -c confidence level]

#----SETUP DEFAULT CONFIGURATION
OUTPUT_FILE=./statistics/test.csv
MIN_CORE=6
WL=0
EXEC=./caliper
#EXEC=./gpu_exec
EXTRA_ARG=-x
VERSION=2
BLOCKS=256
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
#Mettere meta dei core come MinCore

#$MIN_CORE = $(echo "ciao")

#Set workload to $MIN_CORE / ($num_cores * $num_cores) 
#This grant that distributed load for this configuration never goes above 1
execute_caliper(){
    #num_cores = $1
    MAX_CORE=`echo "scale=0; ($1 * $1)" | bc`
    MIN_CORE=`echo "scale=0; ($MAX_CORE)/2" | bc`
    WORKLOAD=`echo "scale=4; $MIN_CORE / ($1 * $1)" | bc`
    echo "COMPUTING $MAX_CORE  with mincore = $MIN_CORE"
    echo $EXEC $1 $1 $MIN_CORE $WORKLOAD $EXTRA_ARG $VERSION $BLOCKS
    $EXEC $1 $1 $MIN_CORE $WORKLOAD $EXTRA_ARG $VERSION $BLOCKS>> $OUTPUT_FILE 
}

#for num_cores in 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 24 26 28 30 32 34 36 38 40 42 44 45
for num_cores in 4 28 30 32 34 36 38 40 42 44 45
do
    execute_caliper "$num_cores"
done