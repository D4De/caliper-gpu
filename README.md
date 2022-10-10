# Caliper-GPU-version

## Parameters:
  ./caliper [Rows][Cols][Min Cores Alive][Initial Workload][-c {*conf interval*} {*stop threshold*}]
## IDEAS:
* Remove the TempModel huge file and build it real time
* Use Simple Grid of threads and atomicAdd (Stupid version)
* Use Simple Grid of threads and Parallel reduction like for the montecarlo exercise seen in class (Simple solution)
* Use Dynamic programming to parallelize some internal loops (More Elaborated)
  * The loops of Loads computation
  * The loops of TempModel function
  * The loops to compute the Death time of each core to select the shortest one
* Build a test-bench to see time improvement and save progess of each algo version
* Build a little script to run all cuda statistics commands and save the results in our own format (This make easy to compare different version of our algorithm)
* (See other possible parallelizations)
* Improve the CPU performance by inserting an HASHmap to save dynamically alpha configurations (dynamic programming) so that noo need to recompute alphas if same configuration occur multiple time during simulations
  
## Already Done Steps:
* Now Caliper is stand-alone, it dosnt need any tempModel file and can build it using an online approach
* Created the "Stupid parallel version" (But getting wrong result on simulation)
* Created a basic output file format to store result and execution time of our algo
## Question To ask:
* How calculate the QUOS Constraint, simply check for the condition at the start? or need to check on each cycle like in sequential code
* Is the "R()" function output file necessary? (because it is build on an hashmap incompatible with cuda)
* Is the parallel floating point addition a problem? (in AAPP we have seen that the approximation of floating points make the results of multiple execution different)
* How does the approximation on alpha values work? We implement a naive function which always keeps exactly 6 digits
* How allocate dynamically an array into registers of gpu (instead of global memory)
## Caliper CPU args:

testare il tempo di esecuzione con grossi numeri

## Caliper GPU args:

## Result FILE FORMAT:

[Row][Col][MinCore][InitWorkload]\
ExecutionTime
MTTF\
Mean\
Variance\
Confidence