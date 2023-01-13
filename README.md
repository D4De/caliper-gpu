# Caliper-GPU-version

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

Metrics list:
https://gist.github.com/mrprajesh/352cbe661ee27a6b4627ae72d89479e6

nvprof --metrics

## Benchmarks:
### DONE:
(REDO 34 36 on redux and coalesced)
- Cpu dummy
- Redux
- Coalesced
- Struct

### TODO:
- Coalesced opt
- Struct opt
- 2D grid
- 2D grid Linearized
- 2D grid Dynamic
