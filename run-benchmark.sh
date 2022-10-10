#!/bin/bash
OUTPUT_FILE=./statistics/cpu_times_confidence.csv
MIN_CORE=6
WL=0
EXEC=./caliper
EXTRA_ARG=-c\ 0.95\ 0.005
#--------------ACTUAL SCRIPT----------------------------------------------
> $OUTPUT_FILE #Empty file content
echo "ROWS,COLS,NUM OF CORES,MIN_CORE,WORKLOAD,NUM_OF_TEST,CONFIDENCE,EXECUTION TIME,MTTF[Hours],MTTF[Years]"

for num_cores in {4..7}
do
    #Set workload to $MIN_CORE / ($num_cores * $num_cores) 
    #This grant that distributed load for this configuration never goes above 1
    $EXEC $num_cores $num_cores 6 $(echo "scale=4; $MIN_CORE / ($num_cores * $num_cores)" | bc) $EXTRA_ARG >> $OUTPUT_FILE 
done
for num_cores in 10 11 15 20 22
do
    #Set workload to $MIN_CORE / ($num_cores * $num_cores) 
    #This grant that distributed load for this configuration never goes above 1
    $EXEC $num_cores $num_cores 6 $(echo "scale=4; $MIN_CORE / ($num_cores * $num_cores)" | bc) $EXTRA_ARG  >> $OUTPUT_FILE 
done