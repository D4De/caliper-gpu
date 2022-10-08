#!/bin/bash
rm -f ./statistics/sequential_times.csv
echo "ROWS,COLS,NUM OF CORES,MIN_CORE,WORKLOAD,EXECUTION TIME"
for num_cores in {3..5}
do
    ./caliper $num_cores $num_cores 6 0.5 
done
