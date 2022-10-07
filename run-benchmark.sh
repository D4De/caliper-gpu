rm results.csv
echo "CONFIGURATION;NUM OF CORES;MIN_CORE;WORKLOAD;EXECUTION TIME;"
for num_cores in {3..5}
do
    ./caliper $num_cores $num_cores 6 0.5 
done
