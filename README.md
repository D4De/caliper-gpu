# Caliper-GPU-version
nvprof --csv --log-file ./bench/redux.csv --metrics gld_efficiency --metrics gst_efficiency --metrics sm_efficiency --metrics branch_efficiency --metrics achieved_occupancy --metrics warp_nonpred_execution_efficiency ./gpu_exec 10 10 50 0.5 -g 1


## Parameters:
  ./caliper [Rows][Cols][Min Cores Alive][Initial Workload][-c {*conf interval*} {*stop threshold*}]
## Caliper CUDA VERSIONS:
- Dummy atomicAdd
- Dummy atomicAdd + Swap optimization (compute simulation only on alive cores)
- Parallel Reduction (Redux) + Swap opt.
- Coalesced Memory access
- Struct coalesced
- Coalesced with + Swap optimization + Temp model opt
- Struct + temp model opt
- 2D Grid
- 2D Grid Linearized
- 2D Grid Dynamic programming

## Benchmarks:
### DONE:
- Redux (4 -> 43) (manca 44, 45)
### TODO:
- Coalesced
- Struct
- Coalesced opt
- Struct opt
- 2D grid
- 2D grid Linearized
- 2D grid Dynamic
