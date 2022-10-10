#!/bin/bash
> ./statistics/cpu_times.csv #Empty file content
echo "ROWS,COLS,NUM OF CORES,MIN_CORE,WORKLOAD,EXECUTION TIME"
MIN_CORE=6
WL=0
for num_cores in {4..7}
do
    #Set workload to $MIN_CORE / ($num_cores * $num_cores) 
    #This grant that distributed load for this configuration never goes above 1
    ./caliper $num_cores $num_cores 6 $(echo "scale=4; $MIN_CORE / ($num_cores * $num_cores)" | bc)
done
for num_cores in 10 11 15 20
do
    #Set workload to $MIN_CORE / ($num_cores * $num_cores) 
    #This grant that distributed load for this configuration never goes above 1
    ./caliper $num_cores $num_cores 6 $(echo "scale=4; $MIN_CORE / ($num_cores * $num_cores)" | bc)
done