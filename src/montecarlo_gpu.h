#ifndef CALIPER_GPU
#define CALIPER_GPU
#define CUDA
#define DEBUG_CUDA_1D
//CUDA-UTILS
#include "cuda-utils/cuda_helper.h"
#include "cuda-utils/parallel_reduction.h"
#include "./simulation-utils/simulation_config.h"
//UTILS
#include "utils/benchmark_helper.h"
#include "utils/utils.h"
//THERMAL MODEL
#include "./simulation-utils/thermal_model.h"


/**
 * Calculate all the temps of the cores in the grid
 * https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda
 * https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/
 */
__device__ void tempModel_gpu(float* loads, float* temps,int* indexes,int left_cores,float distributed_wl,int cols, int rows,int offset){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int absolute_index = indexes[i + offset];     //offset in global memory
    int relative_index = absolute_index - offset; //local position in the local cores grid

    int r = relative_index/cols;    //Row position into local grid
    int c = relative_index%cols;    //Col position into local grid

    float temp = 0;
    int k,h;

    if(i<left_cores){
        for (k = -1; k < 2; k++){
            for (h = -1; h < 2; h++){
                if ((k != 0 || h != 0) && k != h && k != -h && r + k >= 0 && r + k < rows && c + h >= 0 && c + h < cols){
                    temp += loads[offset+(r + k)*cols + (c + h)] * NEIGH_TEMP;
                }
            }
        }
        temps[absolute_index] = ENV_TEMP + loads[absolute_index] * SELF_TEMP + temp;
    }
    return;
}
__device__ void tempModel_gpu_struct(simulation_state sim_state, configuration_description config,float distributed_load,int left_alive){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    int max_cores = config.max_cores;

    core_state* cores = sim_state.core_states;
    for(int i=0;i<left_alive;i++){
        int absolute_index = getIndex(i,config.num_of_tests);          //contain position of this core in the original grid
        int relative_index = sim_state.indexes[absolute_index]; //local position (usefull only on gpu global memory,for cpu is same as absolute)

        int r = relative_index/config.cols;    //Row position into local grid
        int c = relative_index%config.cols;    //Col position into local grid

        float temp = 0;
        int k,h;
        //CUDA_DEBUG_MSG("Core index = [%d,%d,%d]\n",23,getIndex(i,config.num_of_tests),33);

        for (k = -1; k < 2; k++)
        {
            for (h = -1; h < 2; h++)
            {
                if ((k != 0 || h != 0) && k != h && k != -h && r + k >= 0 && r + k < config.rows && c + h >= 0 && c + h < config.cols){
                    int idx = getIndex((r + k)*config.cols + (c + h),config.num_of_tests);
                    //int ld = (indexes[idx] < left_alive) ? distributed_load : 0;
                    temp += cores[idx].load * NEIGH_TEMP;
                }
            }
        }

        cores[absolute_index].temp = ENV_TEMP + cores[absolute_index].load * SELF_TEMP + temp;
    }

}

__device__ void tempModel_gpu_coalesced(simulation_state sim_state, configuration_description config,float distributed_load,int left_alive){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    
    int max_cores = config.max_cores;

    float* loads   = sim_state.loads;
    float* temps   = sim_state.temps;

    for(int i=0;i<left_alive;i++){
        int absolute_index = getIndex(i,config.num_of_tests);          //contain position of this core in the original grid
        int relative_index = sim_state.indexes[absolute_index]; //local position (usefull only on gpu global memory,for cpu is same as absolute)

        int r = relative_index/config.cols;    //Row position into local grid
        int c = relative_index%config.cols;    //Col position into local grid

        float temp = 0;
        int k,h;
        //CUDA_DEBUG_MSG("Core index = [%d,%d,%d]\n",23,getIndex(i,config.num_of_tests),33);

        for (k = -1; k < 2; k++)
        {
            for (h = -1; h < 2; h++)
            {
                if ((k != 0 || h != 0) && k != h && k != -h && r + k >= 0 && r + k < config.rows && c + h >= 0 && c + h < config.cols){
                    int idx = getIndex((r + k)*config.cols + (c + h),config.num_of_tests);
                    //int ld = (indexes[idx] < left_alive) ? distributed_load : 0;
                    temp += loads[idx] * NEIGH_TEMP;
                }
            }
        }

        temps[absolute_index] = ENV_TEMP + loads[absolute_index] * SELF_TEMP + temp;
    }

}


__device__ void tempModel_gpu_grid(simulation_state sim_state, configuration_description config, int core_id, int walk_id){

    int r = core_id/config.cols;    //Row position into local grid
    int c = core_id%config.cols;    //Col position into local grid

    float temp = 0;
    int k,h;

    if(core_id<config.max_cores){
        for (k = -1; k < 2; k++){
            for (h = -1; h < 2; h++){
                if ((k != 0 || h != 0) && k != h && k != -h && r + k >= 0 && r + k < config.rows && c + h >= 0 && c + h < config.cols){
                    temp += sim_state.core_states[walk_id*config.max_cores+(r + k)*config.cols + (c + h)].load * NEIGH_TEMP;
                }
            }
        }
        sim_state.core_states[core_id+walk_id*config.max_cores].temp = ENV_TEMP + sim_state.core_states[core_id+walk_id*config.max_cores].load * SELF_TEMP + temp;
    }
    return;
}

__device__ void tempModel_gpu_struct_optimized(simulation_state sim_state, configuration_description config,double distributed_load,int left_alive){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int offset = tid * (config.rows * config.cols);
    int max_cores = config.max_cores;

    core_state* cores = sim_state.core_states;
    core_state tmp_core;

    for(int i=0;i<left_alive;i++){

        int absolute_index = GETINDEX(i,tid,config.num_of_tests);
        tmp_core =  sim_state.core_states[absolute_index];          //TMP local copy of current core
        float temp = 0;

        //LOOP UNROOLING (if some neightbour is out of range then the neighbour pointer will point to a false bool (avoid if statement this way))
        bool isNeighbourAlive;
        int  neighbourPos;
        //TOP
        isNeighbourAlive = *tmp_core.top_core;
        temp +=  isNeighbourAlive * distributed_load * NEIGH_TEMP; 
         //BOTTOM
        isNeighbourAlive = *tmp_core.bot_core;
        temp +=  isNeighbourAlive * distributed_load * NEIGH_TEMP; 
         //LEFT
        isNeighbourAlive = *tmp_core.left_core;
        temp +=  isNeighbourAlive * distributed_load * NEIGH_TEMP; 
         //RIGHT
        isNeighbourAlive = *tmp_core.right_core;
        temp +=  isNeighbourAlive * distributed_load * NEIGH_TEMP; 

        cores[absolute_index].temp = ENV_TEMP + distributed_load * SELF_TEMP + temp;
    }

}

/**
 * Calculate for each core the TTF (time till die) and extract the core that will die sooner
*/
__device__ int montecarlo_random_walk_step(simulation_state sim_state,configuration_description config,float*stepT,int* minIndex,int left_cores,int offset){

    int j = threadIdx.x + blockDim.x * blockIdx.x;

    //NEED TO PUT A CHECK TO AVOID PADDING ERROR!!
    if(j<left_cores){
        //Pointers to global memory:
        float* currR   = sim_state.currR;
        float* loads   = sim_state.loads;
        float* temps   = sim_state.temps;
        bool*  alives  = sim_state.alives; //TODO allocate those array on global memory on host side
        int*   indexes = sim_state.indexes;

        float * sumTTF_res   = sim_state.sumTTF;
        float * sumTTFx2_res = sim_state.sumTTFx2;

        curandState_t *states = sim_state.rand_states;

        int index = indexes[offset + j]; //Current alive core
        int random,eqT;
        //Random  is in range [0 : currR[j]]
        random =(double)curand_uniform(&states[j])* currR[index]; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
        double alpha = getAlpha(temps[index]);
        float t;
        t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
        eqT = alpha * pow(-log(currR[index]), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
        t = t - eqT;//save t into shared mem

        /*
        ./src/montecarlo_gpu.h(70): error: calling a constexpr __host__ function("log") from a __device__ function("montecarlo_random_walk_step") is not allowed. The experimental flag '--expt-relaxed-constexpr' can be used to allow this.
        */
        stepT[threadIdx.x] = t;

        //Use parallel reduction to calculate the MINIMUM!!!

        __syncthreads();

        accumulate_min<float>(stepT,blockDim.x,left_cores); //Works only if left_cores < blockDim (if not need global reduction or atomic add..)

        __syncthreads();

        if(stepT[0] == stepT[threadIdx.x ])*minIndex = index;

        //TODO IF MULTIPLE BLOCKS -> NEED GLOBAL ACCUMULATION!!!
    }
}

/**
 * Update the simulation state by updating the currR function of each core ù
 * So calculate probability to die in next iteration for each core
*/
__device__ void montecarlo_update_simulation_state(simulation_state sim_state,int dead_index,float stepT,int left_cores,int offset){

    int j = threadIdx.x + blockDim.x * blockIdx.x;

    float* currR   = sim_state.currR;
    float* loads   = sim_state.loads;
    float* temps   = sim_state.temps;
    int*   indexes = sim_state.indexes;

    int index = indexes[offset + j]; //Current alive core

    //move index of dead core to end
    swap_core_index(sim_state.indexes,dead_index,left_cores,offset);

    // compute remaining reliability for working cores
    if(j < left_cores)
    {
        int index = indexes[offset+j]; //Current alive core
        double alpha = getAlpha(temps[index]);
        int eqT;
        eqT = alpha * pow(-log(currR[index]), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
        currR[index] = exp(-pow((stepT + eqT) / alpha, BETA));
    }
}

/**
 * Execute all necessary steps for an iteration of the random walk sequence
*/
__global__ void montecarlo_simulation_step(simulation_state sim_state,configuration_description config,int distributed_wl,int left_cores,int offset,int position){

    float* currR   = sim_state.currR;
    float* loads   = sim_state.loads;
    float* temps   = sim_state.temps;
    int*   indexes = sim_state.indexes;

    //Update Temperature Model
    tempModel_gpu(loads,temps,indexes,left_cores,distributed_wl,config.cols,config.rows,offset);

    __shared__ float stepT[1024];
    int min_index;
    //Determine which core die in this iteration (by calculating TTF)
    montecarlo_random_walk_step(sim_state,config,stepT,&min_index,left_cores,offset); //Work only if leftcores<blockdim

    __syncthreads();
    //sim_state.sumTTF[position] = stepT[0];  //TODO we need an offset of thread into sumTTF global mem
    if(threadIdx.x == 0 && position==0){
        printf("Val:%f\n",stepT[0]);
    }
    //Update the simulation state for this iteration
    if(left_cores < config.min_cores){
        montecarlo_update_simulation_state(sim_state,min_index,stepT[0],left_cores,offset);
    }
}

__device__ int getMatrixIndex(int col,int row,int id,int offset){
    int i;
    return offset;
}




__global__ void montecarlo_simulation_cuda_redux_coalesced(simulation_state sim_state,configuration_description config ,float * sumTTF_res,float * sumTTFx2_res){
    curandState_t *states = sim_state.rand_states;
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //extern __shared__ unsigned int tmp_sumTTF[];
    //extern __shared__ unsigned int tmp_sumTTFx2[]; //??? dont know if necessary
    extern __shared__ float partial_sumTTF[];

    sumTTF_res[blockIdx.x] = 0;
    partial_sumTTF[threadIdx.x] = 0;
    
    if(tid<config.num_of_tests){
        double random;
        int left_cores;
        double stepT;
        int minIndex;
        double totalTime;
        int j;
        double t, eqT;

        float* currR   = sim_state.currR;
        float* loads   = sim_state.loads;
        float* temps   = sim_state.temps;
        int*   indexes = sim_state.indexes;
        int*   real_pos= sim_state.real_pos;


        left_cores = config.max_cores;
        totalTime = 0;
        minIndex = 0;

        for (j = 0; j < config.max_cores; j++) { //Also parallelizable (dont know if usefull, only for big N)
            int index = GETINDEX(j,tid,config.num_of_tests);

            indexes[index]  = j;
            real_pos[index] = index;
            currR[index]    = 1;
            loads[index]    = config.initial_work_load;
        }

        while (left_cores >= config.min_cores) {
            minIndex = -1;

            //-----------Redistribute Loads among alive cores----------
            double distributedLoad = (double)config.initial_work_load * (double)config.max_cores / (double)left_cores;

            //To improve both performance and divergence we remove the if(alive[j])
            //by using the indexes of alive cores instead of if(alive[j])

            //Set Load of alive cores
            for (j = 0; j < left_cores; j++) {
                int index = GETINDEX(j,tid,config.num_of_tests);
                loads[index] = distributedLoad;    //Each TH will access contiguous loads cell based on tid
            }

            //DA FIXARE PROBABILMENTE, RISULTATO VIENE NAN
            tempModel_gpu_coalesced(sim_state,config,distributedLoad,left_cores);
            
            //-----------Random Walk Step computation-------------------------
            for (j = 0; j < left_cores; j++) {
                //Random  is in range [0 : currR[j]]
                int index = GETINDEX(j,tid,config.num_of_tests);
                random =(double)curand_uniform(&states[tid])* currR[index]; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                double alpha = getAlpha(temps[index]);
                t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                eqT = alpha * pow(-log(currR[index]), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
                t = t - eqT;

                if (minIndex == -1 || (minIndex != -1 && t < stepT)) {
                    minIndex = j;//Set new minimum index
                    stepT = t;   //set new minimum time as timeStep
                    //CUDA_DEBUG_MSG("Min selection : %d con stepT = %f\n",minIndex,stepT);
                } //TODO ADD A CHECK ON MULTIPLE FAILURE IN THE SAME INSTANT OF TIME.
            }
            //CUDA_DEBUG_MSG("MORTO CORE: %d con stepT = %f\n",minIndex,stepT);

            //---------UPDATE TOTAL TIME-----------------
            //Current simulation time
            partial_sumTTF[threadIdx.x] = partial_sumTTF[threadIdx.x] + stepT;
            //totalTime = totalTime + stepT;
            //---------UPDATE Configuration-----------------
            if (left_cores > config.min_cores) {
                //Swap all cells
                swapState(sim_state,minIndex,left_cores,config.num_of_tests);
                // compute remaining reliability for working cores
                for (j = 0; j < left_cores; j++) {
                    int index = GETINDEX(j,tid,config.num_of_tests); //Current alive core
                    double alpha = getAlpha(temps[index]);
                    eqT = alpha * pow(-log(currR[index]), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
                    currR[index] = exp(-pow((stepT + eqT) / alpha, BETA));
                }
            }
            //Set Load of lastly dead core to 0
            loads[GETINDEX(left_cores-1,tid,config.num_of_tests)] = 0;//Potrebbe essere qui lerrore
            left_cores--; //Reduce number of alive core in this simulation
        }//END SIMULATION-----------------------------
        //Sync the threads
        __syncthreads();

        //Acccumulate the results of this block
        accumulate<float>(partial_sumTTF,blockDim.x,config.num_of_tests);

        //Add the partial result of this block to the final result
        __syncthreads();
        if(threadIdx.x == 0){
            //atomicAdd(&(sumTTF_res[0]),partial_sumTTF[0]);
            //USING ATOMIC ADD-> SAME RESULT AS CPU, WITH GLOBAL REDUCE.... NOT:.. CHECK WHY

            //Each Thread 0 assign to his block cell the result of its accumulate
            sumTTF_res[blockIdx.x] = partial_sumTTF[0]; //Each block save its result in its "ID"
        }

        //TODO ACCUMULATE ALSO SUMTTFx2

    }
}


__global__ void montecarlo_simulation_cuda_redux_struct(simulation_state sim_state,configuration_description config ,float * sumTTF_res,float * sumTTFx2_res){
   //CUDA_DEBUG_MSG("Initialize\n");
    curandState_t *states = sim_state.rand_states;

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = tid * (config.rows * config.cols);
    
    //extern __shared__ unsigned int tmp_sumTTF[];
    //extern __shared__ unsigned int tmp_sumTTFx2[]; //??? dont know if necessary
   extern __shared__ float partial_sumTTF[];
   __shared__ core_state local_cores[256];

    sumTTF_res[blockIdx.x] = 0;
    partial_sumTTF[threadIdx.x] = 0;

    if(tid<config.num_of_tests){
        double random;
        int left_cores;
        double stepT;
        int minIndex;
        int actual_minIndex;
        double totalTime;
        int j;
        double t, eqT;

        //Change to struct
        core_state * cores   = sim_state.core_states;
        int* indexes = sim_state.indexes;
        int* real_pos = sim_state.real_pos;

        left_cores = config.max_cores;
        totalTime = 0;
        minIndex = 0;

        for (j = 0; j <  config.max_cores; j++) { //Also parallelizable (dont know if usefull, only for big N)
            //int index = getIndex(j,config.num_of_tests);
            int index = GETINDEX(j,tid,config.num_of_tests);
            indexes[index]  = j;
            real_pos[index] = index;

            //Coalesced access to struct
            local_cores[threadIdx.x] = cores[index];  //TODO CHECK IF WE NEED IT

            local_cores[threadIdx.x].curr_r      = 1;
            local_cores[threadIdx.x].real_index  = j;
            //local_cores[threadIdx.x].load        = config.initial_work_load;

            cores[index] = local_cores[threadIdx.x];
            //CUDA_DEBUG_MSG("INIZIALIZZA %d con %d: \n",j,offset + j);
        }

        //CUDA_DEBUG_MSG("Start Simulation\n");
        while (left_cores >= config.min_cores) {
            minIndex = -1;

            //-----------Redistribute Loads among alive cores----------
            double distributedLoad = (double)config.initial_work_load * (double)config.max_cores / (double)left_cores;

            for (j = 0; j < left_cores; j++) {
                int index = getIndex(j,config.num_of_tests);
                cores[index].load = distributedLoad;
            }
            

            //TODO UPDATE THE TEMPERATURE
            tempModel_gpu_struct(sim_state,config,distributedLoad,left_cores);
        
            //Set Load of alive cores
            for (j = 0; j < left_cores; j++) {
                //int index = getIndex(j,config.num_of_tests); //Current alive core
                int index = GETINDEX(j,tid,config.num_of_tests);

                local_cores[threadIdx.x] = cores[index]; //Coalesced access to struct
                random =(double)curand_uniform(&states[tid])* local_cores[threadIdx.x].curr_r; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                double alpha = getAlpha(local_cores[threadIdx.x].temp);
                t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                eqT = alpha * pow(-log(local_cores[threadIdx.x].curr_r), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
                t = t - eqT;

                //the difference between the two values represents the time elapsed from the previous failure to the current failure
                //(we will sum to the total time the minimum of such values)
                
                if (minIndex == -1 || (minIndex != -1 && t < stepT)) {   
                    //CUDA_DEBUG_MSG("Waaaaa potrebbe morire %d con offset %dcon %f\n",actual_minIndex,offset + actual_minIndex,t);
                    minIndex = j;//Set new minimum index
                    stepT = t;   //set new minimum time as timeStep
                } //TODO ADD A CHECK ON MULTIPLE FAILURE IN THE SAME INSTANT OF TIME.
            }

            //---------UPDATE TOTAL TIME-----------------
            //Current simulation time
            partial_sumTTF[threadIdx.x] = partial_sumTTF[threadIdx.x] + stepT;

            //---------UPDATE Configuration-----------------
            if (left_cores > config.min_cores) {
                //Swap all cells  (TODO SWAP WITH STRUCT)
                swapStateStruct(sim_state,minIndex,left_cores,config.num_of_tests);
                // compute remaining reliability for working cores
                for (j = 0; j < left_cores; j++) {
                    //int index = getIndex(j,config.num_of_tests); //Current alive core
                    int index = GETINDEX(j,tid,config.num_of_tests);
                    //Coalesced Read of core state struct
                    local_cores[threadIdx.x] = cores[index];
                    
                    //Local changes to the core state (in register memory!!)
                    double alpha = getAlpha(local_cores[threadIdx.x].temp);
                    eqT = alpha * pow(-log(local_cores[threadIdx.x].curr_r), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
                    local_cores[threadIdx.x].curr_r = exp(-pow((stepT + eqT) / alpha, BETA));

                    cores[index] = local_cores[threadIdx.x];
                }
            }

            //Set Load of lastly dead core to 0
            int index = getIndex(left_cores-1,config.num_of_tests); //Current alive core
            cores[index].load = 0;

            //Decrement counter of alive core
            left_cores--; //Reduce number of alive core in this simulation
        }

        //SUM ALL THE CORE VALUES INSIDE THIS BLOCK INTO sumTTF[blockid]
        //Sync the threads
        __syncthreads();
        CUDA_DEBUG_MSG("SUMTTF: %f\n", partial_sumTTF[0]);
        //Acccumulate the results of this block
        accumulate<float>(partial_sumTTF,blockDim.x,config.num_of_tests);

        //Add the partial result of this block to the final result
        __syncthreads();
        if(threadIdx.x == 0){
            //Each Thread 0 assign to his block cell the result of its accumulate
            sumTTF_res[blockIdx.x] = partial_sumTTF[0]; //Each block save its result in its "ID"
        }
    }
}


__global__ void montecarlo_simulation_cuda_redux_struct_optimized(simulation_state sim_state,configuration_description config ,float * sumTTF_res,float * sumTTFx2_res){
    //CUDA_DEBUG_MSG("Initialize\n");
    curandState_t *states = sim_state.rand_states;

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = tid * (config.rows * config.cols);

    extern __shared__ float partial_sumTTF[];
    __shared__ core_state local_cores[256];

    sumTTF_res[blockIdx.x]      = 0;
    partial_sumTTF[threadIdx.x] = 0;

    if(tid<config.num_of_tests){
        double random;
        int left_cores;
        double stepT;
        int minIndex;
        int actual_minIndex;
        double totalTime;
        int j;
        double t, eqT;

        //Change to struct
        core_state * cores   = sim_state.core_states;

        left_cores  = config.max_cores;
        totalTime   = 0;
        minIndex    = 0;

        bool false_register = false;
        bool out_of_range;  //True if neightbour is out of range

        for (j = 0; j <  config.max_cores; j++) { //Also parallelizable (dont know if usefull, only for big N)
            //int index = getIndex(j,config.num_of_tests);
            int index = GETINDEX(j,tid,config.num_of_tests);

            int r = j/config.cols;    //Row position into local grid
            int c = j%config.cols;    //Col position into local grid

            //Coalesced access to struct
            //local_cores[threadIdx.x] = cores[index];  //TODO CHECK IF WE NEED IT

            local_cores[threadIdx.x].curr_r      = 1;
            local_cores[threadIdx.x].real_index  = j;

            //Top
            out_of_range = ((r - 1) < 0);
            local_cores[threadIdx.x].top_core = out_of_range ? &false_register : &(sim_state.alives[offset + (r-1)*config.cols + c]); 
            //Bot
            out_of_range = ((r + 1) > config.rows);
            local_cores[threadIdx.x].bot_core = out_of_range ? &false_register : &(sim_state.alives[offset + (r+1)*config.cols + c]); 
            //Left
            out_of_range = ((c - 1) < 0);
            local_cores[threadIdx.x].left_core = out_of_range ? &false_register: &(sim_state.alives[offset + (r)*config.cols + (c-1)]); 
            //Right
            out_of_range = ((c + 1) > config.cols);
            local_cores[threadIdx.x].right_core = out_of_range ? &false_register : &(sim_state.alives[offset + (r)*config.cols + (c+1)]); 

            //Write back result
            cores[index] = local_cores[threadIdx.x];


            sim_state.alives[offset + j] = true;
        }

        //CUDA_DEBUG_MSG("Start Simulation\n");
        while (left_cores >= config.min_cores) {
            minIndex = -1;

            //-----------Redistribute Loads among alive cores----------
            double distributedLoad = (double)config.initial_work_load * (double)config.max_cores / (double)left_cores;

            //TODO UPDATE THE TEMPERATURE
            tempModel_gpu_struct_optimized(sim_state,config,distributedLoad,left_cores);
        
            //Set Load of alive cores
            for (j = 0; j < left_cores; j++) {
                //int index = getIndex(j,config.num_of_tests); //Current alive core
                int index = GETINDEX(j,tid,config.num_of_tests);

                local_cores[threadIdx.x] = cores[index]; //Coalesced access to struct
                random =(double)curand_uniform(&states[tid])* local_cores[threadIdx.x].curr_r; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                double alpha = getAlpha(local_cores[threadIdx.x].temp);
                t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                eqT = alpha * pow(-log(local_cores[threadIdx.x].curr_r), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
                t = t - eqT;

                //the difference between the two values represents the time elapsed from the previous failure to the current failure
                //(we will sum to the total time the minimum of such values)
                
                if (minIndex == -1 || (minIndex != -1 && t < stepT)) {   
                    actual_minIndex = cores[index].real_index;
                    //CUDA_DEBUG_MSG("potrebbe morire %d con offset %dcon %f\n",actual_minIndex,offset + actual_minIndex,t);
                    minIndex = j;//Set new minimum index
                    stepT = t;   //set new minimum time as timeStep
                } //TODO ADD A CHECK ON MULTIPLE FAILURE IN THE SAME INSTANT OF TIME.
            }

            //---------UPDATE TOTAL TIME-----------------
            //Current simulation time
            partial_sumTTF[threadIdx.x] = partial_sumTTF[threadIdx.x] + stepT;
            sim_state.alives[offset + actual_minIndex] = false;//Mark core as dead!!
            //CUDA_DEBUG_MSG("-------\n");
            //CUDA_DEBUG_MSG("Dead core: %d\n",actual_minIndex);

            //---------UPDATE Configuration-----------------
            if (left_cores > config.min_cores) {
                swapStateStructOptimized(sim_state,minIndex,left_cores,config.num_of_tests);
                // compute remaining reliability for working cores
                for (j = 0; j < left_cores; j++) {
                    //int index = getIndex(j,config.num_of_tests); //Current alive core
                    int index = GETINDEX(j,tid,config.num_of_tests);
                    //Coalesced Read of core state struct
                    local_cores[threadIdx.x] = cores[index];
                    
                    //Local changes to the core state (in register memory!!)
                    double alpha = getAlpha(local_cores[threadIdx.x].temp);
                    eqT = alpha * pow(-log(local_cores[threadIdx.x].curr_r), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
                    local_cores[threadIdx.x].curr_r = exp(-pow((stepT + eqT) / alpha, BETA));

                    cores[index] = local_cores[threadIdx.x];
                }
            }

            //Decrement counter of alive core
            left_cores--; //Reduce number of alive core in this simulation
        }

        //SUM ALL THE CORE VALUES INSIDE THIS BLOCK INTO sumTTF[blockid]
        //Sync the threads
        __syncthreads();
        //Acccumulate the results of this block
        accumulate<float>(partial_sumTTF,blockDim.x,config.num_of_tests);
        //Add the partial result of this block to the final result
        __syncthreads();
        if(threadIdx.x == 0){
            //Each Thread 0 assign to his block cell the result of its accumulate
            sumTTF_res[blockIdx.x] = partial_sumTTF[0]; //Each block save its result in its "ID"
        }
    }
}


__global__ void montecarlo_simulation_cuda_redux(simulation_state sim_state,configuration_description config ,float * sumTTF_res,float * sumTTFx2_res){

    curandState_t *states = sim_state.rand_states;

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = tid * (config.rows * config.cols);
    //extern __shared__ unsigned int tmp_sumTTF[];
    //extern __shared__ unsigned int tmp_sumTTFx2[]; //??? dont know if necessary
    extern __shared__ float partial_sumTTF[];

    sumTTF_res[blockIdx.x] = 0;
    partial_sumTTF[threadIdx.x] = 0;

    if(tid<config.num_of_tests){
        double random;
        int left_cores;
        double stepT;
        int minIndex;
        double totalTime;
        int j;
        double t, eqT;

        float* currR   = sim_state.currR;
        float* loads   = sim_state.loads;
        float* temps   = sim_state.temps;
        int*   indexes = sim_state.indexes;

        left_cores = config.max_cores;
        totalTime = 0;
        minIndex = 0;

        for (j = 0; j < config.max_cores; j++) { //Also parallelizable (dont know if usefull, only for big N)
            int index = offset + j;
            currR[index]    = 1;
            indexes[index]  = index;
            loads[index]    = config.initial_work_load;
        }

        while (left_cores >= config.min_cores) {
            minIndex = -1;


            //-----------Redistribute Loads among alive cores----------
            double distributedLoad = (double)config.initial_work_load * (double)config.max_cores / (double)left_cores;

            //To improve both performance and divergence we remove the if(alive[j])
            //by using the indexes of alive cores instead of if(alive[j])

            //Set Load of alive cores
            for (j = 0; j < left_cores; j++) {
                int index = indexes[offset + j];
                loads[index] = distributedLoad;
            }

            //FUNZIONE OTTIMIZZATA DA FIXARE(??)
            tempModel(loads, temps,indexes,left_cores, config.rows, config.cols,offset);
            //Versione non ottimizzata
            //tempModel(loads, temps,indexes,left_cores, rows, cols,offset);

            //-----------Random Walk Step computation-------------------------
            for (j = 0; j < left_cores; j++) {
                //Random  is in range [0 : currR[j]]
                int index = indexes[offset + j]; //Current alive core
                random =(double)curand_uniform(&states[tid])* currR[index]; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                double alpha = getAlpha(temps[index]);
                t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                eqT = alpha * pow(-log(currR[index]), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
                t = t - eqT;

                //the difference between the two values represents the time elapsed from the previous failure to the current failure
                //(we will sum to the total time the minimum of such values)

                if (minIndex == -1 || (minIndex != -1 && t < stepT)) {
                    minIndex = index;//Set new minimum index
                    stepT = t;   //set new minimum time as timeStep
                } //TODO ADD A CHECK ON MULTIPLE FAILURE IN THE SAME INSTANT OF TIME.
            }
            //-------Check if No failed core founded-----------
            if (minIndex == -1) {
                //TODO HOW THROW ERRORS IN CUDA?????
                return;
            }
            //---------UPDATE TOTAL TIME-----------------
            //Current simulation time
            partial_sumTTF[threadIdx.x] = partial_sumTTF[threadIdx.x] + stepT;
            //totalTime = totalTime + stepT;
            //---------UPDATE Configuration-----------------
            if (left_cores > config.min_cores) {
                swap_core_index(sim_state.indexes,minIndex,left_cores,offset); //move index of dead core to end
                // compute remaining reliability for working cores
                for (j = 0; j < left_cores; j++) {
                    int index = indexes[offset+j]; //Current alive core
                    double alpha = getAlpha(temps[index]);
                    eqT = alpha * pow(-log(currR[index]), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
                    currR[index] = exp(-pow((stepT + eqT) / alpha, BETA));
                }
            }
            //Set Load of lastly dead core to 0
            loads[indexes[offset + left_cores-1]] = 0;//Potrebbe essere qui lerrore
            left_cores--; //Reduce number of alive core in this simulation
        }//END SIMULATION-----------------------------
        //Sync the threads
        __syncthreads();

        //Acccumulate the results of this block
        accumulate<float>(partial_sumTTF,blockDim.x,config.num_of_tests);
        //atomicAdd(&(partial_sumTTF[0]),partial_sumTTF[threadIdx.x]);
        //Add the partial result of this block to the final result
        __syncthreads();
        if(threadIdx.x == 0){
            //atomicAdd(&(sumTTF_res[0]),partial_sumTTF[0]);
            //USING ATOMIC ADD-> SAME RESULT AS CPU, WITH GLOBAL REDUCE.... NOT:.. CHECK WHY
            //printf("SUMTTF [ %d ] : %f\n",tid,partial_sumTTF[0]);
            //Each Thread 0 assign to his block cell the result of its accumulate
            sumTTF_res[blockIdx.x] = partial_sumTTF[0]; //Each block save its result in its "ID"
        }

        //TODO ACCUMULATE ALSO SUMTTFx2

    }
}

 //TODO look at what can be done with warp_reduce
//assuming a x_index first organization of the warps, no better improvement can be done
//IDEA can be to flip (x, y) dimensions with (cores, walks)
//input should be shared (not global)
__device__ int accumulate_min(float *input, size_t dim, configuration_description config, int* minis)
{
    size_t threadId = threadIdx.y;
    size_t global_offset = (blockIdx.x*blockDim.x + threadIdx.x)*config.max_cores + blockIdx.y*blockDim.y;
    //check for padding
    float block_height = config.max_cores/(blockDim.y*(blockIdx.y+1));
    if(block_height < 1.0)
        dim = config.max_cores%(blockDim.y);
    minis[threadId] = threadId + global_offset;
    __syncthreads();
    bool odd = dim%2;
    for (size_t i = dim/2; i>0; i >>= 1)
    {
        if(threadId < i)
            if(input[minis[threadId]] > input[minis[threadId + i]])
                minis[threadId] = minis[threadId + i];
        __syncthreads();
        if(odd && threadId == 0)
            if(input[minis[threadId]] > input[minis[threadId + i*2]])
                minis[threadId] = minis[threadId + i*2];
        odd = i%2;
        __syncthreads();
    }
    return minis[0];
}

__global__ void prob_to_death(simulation_state sim_state, configuration_description config, int* min_index)
{   
    unsigned int core_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int walk_id = threadIdx.x + blockIdx.x * blockDim.x;
    int num_of_blocks2D = (config.max_cores+config.block_dim-1)/config.block_dim;
    extern __shared__ int mem_shared[];
    
    if(core_id < config.max_cores && walk_id < config.num_of_tests){
        int global_id = walk_id*config.max_cores + core_id;
        core_state s = sim_state.core_states[global_id];
        
        curandState_t *states = sim_state.rand_states;
        double random = (double)curand_uniform(&states[global_id])* s.curr_r; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
        double alpha = getAlpha(s.temp);
        double t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
        double eqT = alpha * pow(-log(s.curr_r), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value

        sim_state.times[global_id]= (float)(t - eqT);
        
        if(!sim_state.core_states[global_id].alive)
            sim_state.times[global_id] = FLT_MAX;
        __syncthreads();
        

        int* minis = &mem_shared[threadIdx.x*blockDim.y];
        int id_min = accumulate_min(sim_state.times, blockDim.y, config, minis);
        if(threadIdx.y == 0){
            min_index[walk_id*num_of_blocks2D + blockIdx.y] = id_min;
        }
    }
}

__global__ void grid_update_state(simulation_state sim_state, configuration_description config, int* min_index, float* partial_sumTTF, int left_cores){
    unsigned int core_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int walk_id = threadIdx.x + blockIdx.x * blockDim.x;
    int num_of_blocks2D = (config.max_cores+config.block_dim-1)/config.block_dim;
    float stepT = sim_state.times[min_index[walk_id*num_of_blocks2D]];      //every thread in a walk will access the same memory location
    if(core_id < config.max_cores && walk_id < config.num_of_tests){
        int global_id = walk_id*config.max_cores + core_id;
        core_state* s = &sim_state.core_states[global_id];
        
        if(global_id == min_index[walk_id*num_of_blocks2D]){
            s->alive = false;
            s->load = 0.0;
        }
        if(s->alive){
            double alpha = getAlpha(s->temp);
            double eqT = alpha * pow(-log(s->curr_r), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
            s->curr_r = exp(-pow(((double)stepT + eqT) / alpha, BETA));
            float distributedLoad = config.initial_work_load * (float)(config.max_cores / (float)left_cores);
            s->load = distributedLoad;
        }
        __syncthreads();
        if(s->alive)
            tempModel_gpu_grid(sim_state, config, core_id, walk_id);
        
        if(core_id == 0){
            partial_sumTTF[walk_id] += stepT;
        }
    }
}



__global__ void init_grid(simulation_state sim_state, configuration_description config, float* sumTTF){
    unsigned int core_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int walk_id = threadIdx.x + blockIdx.x * blockDim.x;
    if(core_id < config.max_cores && walk_id < config.num_of_tests){ 
    int index = core_id + walk_id*config.max_cores;
    sim_state.core_states[index].load = config.initial_work_load;
    sim_state.core_states[index].curr_r = 1.0;
    sim_state.core_states[index].alive = true;
    tempModel_gpu_grid(sim_state, config, core_id, walk_id);
    }
    if(core_id == 0)
        sumTTF[walk_id] = 0;
}



__global__ void reduce_min(simulation_state sim_state, configuration_description config, int* min_index, int size, float* input){
    extern __shared__ unsigned int localVars[];
    size_t id = blockDim.y*blockIdx.y + threadIdx.y;
    size_t walk_id = threadIdx.x + blockDim.x*blockIdx.x;
    size_t threadId = threadIdx.y;
    int num_of_blocks2D = (config.max_cores+config.block_dim-1)/config.block_dim;
    
    //check for padding
    if(walk_id < config.num_of_tests && id < size){
        localVars[threadIdx.x*blockDim.y + threadIdx.y] = min_index[id + num_of_blocks2D*walk_id];
        int* minis = (int*)&localVars[threadIdx.x*blockDim.y];
        minis[threadId] = min_index[size*walk_id + id];
        __syncthreads();

        size_t dim = (float)((blockDim.y*(blockIdx.y +1))/size) > 1.0 ? size-blockDim.y*blockIdx.y : blockDim.y;
    bool odd = dim%2;
    for (size_t i = dim/2; i>0; i >>= 1)
    {
        if(threadId < i)
            if(input[minis[threadId]] > input[minis[threadId + i]])
                minis[threadId] = minis[threadId + i];
        __syncthreads();    //can be removed 
        if(odd && threadId == 0)
            if(input[minis[threadId]] > input[minis[threadId + i*2]])
                minis[threadId] = minis[threadId + i*2];
        odd = i%2;
        __syncthreads();
    }

    if(threadIdx.y == 0){
   
        min_index[id + num_of_blocks2D*walk_id] = minis[0];
    } }
}

__global__ void accumulate_grid_block_level(simulation_state sim_state, configuration_description config, float* TTF, float* sumTTF_res){
    unsigned int walk_id = threadIdx.x + blockDim.x*blockIdx.x;
    extern __shared__ float partial_sumTTF[];
    if(walk_id < config.num_of_tests)
    partial_sumTTF[threadIdx.x] = TTF[walk_id];
    __syncthreads();
    
    if(((float)(blockIdx.x*blockDim.x+blockDim.x)/(float)config.num_of_tests) > 1.0 )
        accumulate2D(partial_sumTTF,config.num_of_tests%blockDim.x);
    else
        accumulate2D(partial_sumTTF,blockDim.x);
    
    if(threadIdx.x == 0){
        sumTTF_res[blockIdx.x] = partial_sumTTF[0];
    }
    
}


__device__ void update_state(simulation_state sim_state, configuration_description config, int walk_id, int core_id, float stepT){
    int global_id = walk_id*config.max_cores + core_id;
    core_state* s = &sim_state.core_states[global_id];
    if(s->alive){
        double alpha = getAlpha(s->temp);
        double eqT = alpha * pow(-log(s->curr_r), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
        s->curr_r = exp(-pow(((double)stepT + eqT) / alpha, BETA));
    }
}

__device__ int prob_to_death_linearized(simulation_state sim_state, configuration_description config, int walk_id, int core_id, int* minis)
{
    //Identifier of this thread
    int global_id = walk_id*config.max_cores + core_id;

    //Random State
    core_state s = sim_state.core_states[global_id];
    curandState_t *states = sim_state.rand_states;
    double random = (double)curand_uniform(&states[global_id])* s.curr_r; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
   
   // CUDA_DEBUG_MSG("Rand OF  : %d -> %f\n",core_id,random);

    //Simulation Step
    double alpha = getAlpha(s.temp);
    double t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
    double eqT = alpha * pow(-log(s.curr_r), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value

    //Find most probable core to die
    sim_state.times[global_id] = FLT_MAX;       //INITIALIZE TO INFINITE
    if(sim_state.core_states[global_id].alive){
         sim_state.times[global_id]= t - eqT;
    } 

    __syncthreads();
    
    int id_min = accumulate_argMin<float>(sim_state,config,walk_id,core_id,minis);

    __syncthreads();
    if(global_id == id_min) {
        sim_state.core_states[global_id].alive = false;
        sim_state.core_states[global_id].load = 0.0;
    }
    return id_min;
}

__global__ void montecarlo_simulation_cuda_grid_linearized(simulation_state sim_state, configuration_description config ,float * sumTTF_res,float * sumTTFx2_res,unsigned int seed) {

    //TODO ADD THE FOR CYCLE TO GRANT CASE WHERE maxcore > blockdim
    
    unsigned int core_id = threadIdx.x;
    unsigned int walk_id = blockIdx.x;

    unsigned int global_id = (int)core_id + walk_id * config.max_cores; //Checked All care ok,

    sumTTF_res[blockIdx.x] = 0;

    __shared__ float    partial_sumTTF[1024];
    __shared__ int      minis[1024];
    

    //int * minis;
    if(core_id < config.max_cores && walk_id < config.num_of_tests){    //Padding control
        //INITIALIZE RANDOM STATE
        curand_init(seed, global_id, 0, &sim_state.rand_states[global_id]);

        //INITIALIZATION OF CORE STATE
        sim_state.core_states[global_id].load = config.initial_work_load;
        sim_state.core_states[global_id].curr_r = 1;
        sim_state.core_states[global_id].alive = true;
        sim_state.core_states[global_id].temp = 0;
        
        //INITIALIZATION OF UTILS VARIABLES
        int left_cores = config.max_cores;
        partial_sumTTF[threadIdx.x] = 0;
        
        //RANDOM WALK SIMULATION FOR THIS CORE
        while(left_cores >= config.min_cores){
            int min;
            float stepT;
            if(sim_state.core_states[global_id].alive) {
                float distributedLoad = config.initial_work_load * (float)(config.max_cores / (float)left_cores); //Calculate new Load
                sim_state.core_states[global_id].load = distributedLoad;                                              //Update the load
                tempModel_gpu_grid(sim_state, config, core_id, walk_id);                                          //Update the temperature mode

            }
            __syncthreads();

            //FIND THE MIN stepT between cores in this block (work if maxcore < 32)
            minis[core_id] = global_id;
            min = prob_to_death_linearized(sim_state, config, walk_id, core_id, minis);
            stepT = sim_state.times[walk_id*config.max_cores]; //We access the "times[0]" of each walk

            //Update the curr_r of this core
            update_state(sim_state, config, walk_id, core_id,stepT);
            __syncthreads();
            //SUM THE stepT TO THE partialSumTTF
            if(core_id == 0){
                 partial_sumTTF[core_id] += stepT;
            }

            left_cores--;
        }
        
        __syncthreads();

        /*TODO manca accumulate block level !!!!!*/

        if(core_id == 0){
            sumTTF_res[blockIdx.x] = partial_sumTTF[core_id];
        }
        __syncthreads();
    }
}


__global__ void collect_res_gpu_grid(float* input, float* result, int num_of_blocks){
    extern __shared__ float partial_reduction[];   //work on this instead of partil_sumTTF, which lies in global memory 
    size_t threadId = threadIdx.x;
   
    if(threadId < num_of_blocks)
    {   
        if(threadId == 0)
            printf("%d\n", num_of_blocks);
        partial_reduction[threadId] = 0.0; //Initialize Partial reduction
        //TILING
        for(int i = threadId; i<num_of_blocks; i+=blockDim.x){
            partial_reduction[threadId] += input[i];            //Apply Tiling to sum all elements outside "blockSize"
        }
        __syncthreads();
            
        
        int dim = (num_of_blocks/blockDim.x) >= 1 ? blockDim.x : num_of_blocks;

        bool odd = dim%2;
        //REDUCTION
        for (int i = dim / 2; i > 0; i >>= 1)
        {
            if ((threadId  < i))
            {
                partial_reduction[threadId] += partial_reduction[threadId + i];
            }

            if(threadId == 0 && odd)
                partial_reduction[threadId] += partial_reduction[threadId + i+1];
            odd = i%2;
            __syncthreads();
        }
    
        if(threadId == 0)
            *result = partial_reduction[0];
    }
}

__device__ void tempModel_gpu_dynamic(simulation_state sim_state, configuration_description config, int core_id, int walk_id, int left_cores){

    int r = core_id/config.cols;    //Row position into local grid
    int c = core_id%config.cols;    //Col position into local grid

    float temp = 0;
    int k,h;

    if(core_id<left_cores){
        for (k = -1; k < 2; k++){
            for (h = -1; h < 2; h++){
                if ((k != 0 || h != 0) && k != h && k != -h && r + k >= 0 && r + k < config.rows && c + h >= 0 && c + h < config.cols){
                    int real_index = sim_state.core_states[walk_id*config.max_cores+(r + k)*config.cols + (c + h)].real_index;
                    temp += sim_state.core_states[real_index].load * NEIGH_TEMP;
                }
            }
        }
        int real_index = sim_state.core_states[core_id+walk_id*config.max_cores].real_index;
        sim_state.core_states[real_index].temp = ENV_TEMP + sim_state.core_states[real_index].load * SELF_TEMP + temp;
    }
}
   


__device__ int prob_to_death_dynamic(simulation_state sim_state, configuration_description config, int walk_id, int core_id, int* minis)
{
    //Identifier of this thread
    int global_id = walk_id*config.max_cores + core_id;

    //Random State
    core_state s = sim_state.core_states[global_id];
    curandState_t *states = sim_state.rand_states;
    double random = (double)curand_uniform(&states[global_id])* s.curr_r; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)

    //Simulation Step
    double alpha = getAlpha(s.temp);
    double t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
    double eqT = alpha * pow(-log(s.curr_r), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
    //Find most probable core to die
    sim_state.times[global_id]= t - eqT;

    __syncthreads();
    
    int id_min = accumulate_argMin<float>(sim_state,config,walk_id,core_id,minis);

    __syncthreads();
    if(global_id == id_min) {
        //SWAP
        sim_state.core_states[id_min].alive = false;
        sim_state.core_states[id_min].load = 0.0;

        core_state t1 = sim_state.core_states[id_min];
        sim_state.core_states[id_min] = sim_state.core_states[blockDim.x + walk_id*config.max_cores];
        sim_state.core_states[blockDim.x + walk_id*config.max_cores] = t1;
        
        int t2 = sim_state.core_states[id_min].real_index;
        sim_state.core_states[id_min].real_index = sim_state.core_states[blockDim.x + walk_id*config.max_cores].real_index;
        sim_state.core_states[blockDim.x + walk_id*config.max_cores].real_index = t2;

    }
    return id_min;
}


__global__ void montecarlo_dynamic_step(simulation_state sim_state,configuration_description config,int walk_id,int left_cores, float* TTF)
{   
    __shared__ int minis[1024];

    unsigned int core_id = threadIdx.x;
    unsigned int global_id = core_id + config.max_cores*walk_id;

    int min;
    float stepT;
    float distributedLoad = config.initial_work_load * (float)(config.max_cores / (float)left_cores); //Calculate new Load
    sim_state.core_states[global_id].load = distributedLoad;                                              //Update the load
    tempModel_gpu_dynamic(sim_state, config, core_id, walk_id, left_cores);                                          //Update the temperature mode
    __syncthreads();
    
    //FIND THE MIN stepT between cores in this block (work if maxcore < 32)
    minis[core_id] = global_id;
    min = prob_to_death_dynamic(sim_state, config, walk_id, core_id, minis);
    stepT = sim_state.times[walk_id*config.max_cores];
    //Update the curr_r of this core
    update_state(sim_state, config, walk_id, core_id ,stepT);
    __syncthreads();
    //SUM THE stepT TO THE partialSumTTF
    if(core_id == 0){
        TTF[walk_id] += stepT; 
    }
}


__global__ void montecarlo_simulation_cuda_dynamic(simulation_state sim_state,configuration_description config, float* TTF){
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = tid*config.max_cores;
    curandState_t *states = sim_state.rand_states;
    TTF[tid] = 0.0;
    
    if(tid<config.num_of_tests){

        int left_cores = config.max_cores;
        core_state * cores   = sim_state.core_states;
        //Initialization
        for (int j = 0; j < config.max_cores; j++) { //Also parallelizable (dont know if usefull, only for big N)
            int index       = offset + j;
            
            cores[index].curr_r         = 1;
            cores[index].real_index     = index;
            cores[index].load           = config.initial_work_load;
        }
        while (left_cores >= config.min_cores) {
            //---------Launch Child kernel to calculate a random walk step for this thread
            //WHY I CANT RUN A 2D child KERNEL????
            int block_dim = left_cores; //TODO FIND IN EFFICIENT WAY THE NEAREST POW OF 2 NEAR TO BLOCK
            int num_of_blocks = 1;// (left_cores + block_dim - 1) / block_dim;

            //Call simulation step dynamicly by decreasing grid size evry time a core die
            montecarlo_dynamic_step<<<num_of_blocks,block_dim>>>(sim_state,config,tid,left_cores, TTF);
            cudaDeviceSynchronize();
            left_cores--;
        }
        //END SIMULATION-----------------------------

        //Add the partial result of this block to the final result
        __syncthreads();
    }
}


/**
 * -Preparation and launch of Simulation
 * -Allocate Global Memory
 * -Initialize random State
 * -Launch Simulation
 * -Collect results from global mem
*/
void montecarlo_simulation_cuda_launcher(configuration_description* config,double * sumTTF,double * sumTTFx2)
{
    //----------------------------------------------------------------------
    //----CUDA variables:---------------------------------------------------
    //----------------------------------------------------------------------

    float *sumTTF_GPU;   //GPU result sumTTF
    float *sumTTF_GPU_final;   //GPU result sumTTF
    float *sumTTFx2_GPU; //GPU result sumTTFx2
    float res;
    simulation_state sim_state;
    curandState_t *states;

    int num_of_blocks = (config->num_of_tests+config->block_dim-1)/config->block_dim;
    int num_of_blocks2D = (config->max_cores+config->block_dim-1)/config->block_dim;

    //----------------------------------------------------------------------
    //----MEMORY ALLOCATION:------------------------------------------------
    //----------------------------------------------------------------------

    CHECK(cudaMalloc(&sumTTF_GPU    , num_of_blocks*sizeof(float)));    //Allocate Result memory
    CHECK(cudaMalloc(&sumTTFx2_GPU  , sizeof(float)));                  //Allocate Result memory
    CHECK(cudaMalloc(&sumTTF_GPU_final, sizeof(float)));                //Allocate Result memory

    allocate_simulation_state_on_device(&sim_state,*config);            //Allocate Simulation/configuration inside a dedicated datastructure

    if(config->gpu_version == VERSION_2D_GRID || config->gpu_version == VERSION_GRID_LINEARIZED || config->gpu_version == VERSION_DYNAMIC)
    {
        CHECK(cudaMalloc(&states        , config->num_of_tests*config->max_cores*sizeof(curandState_t))); //Random States array
    }
    else
    {
        CHECK(cudaMalloc(&states        , config->num_of_tests*sizeof(curandState_t))); //Random States array
    }

    //----------------------------------------------------------------------
    //-----SIM_STATE INITIALIZATION------------------------------------------
    //----------------------------------------------------------------------

    sim_state.sumTTF   = sumTTF_GPU;
    sim_state.sumTTFx2 = sumTTFx2_GPU;

    dim3 blocksPerGrid(num_of_blocks,1,1);
    dim3 threadsPerBlock(config->block_dim,1,1);

    if(config->gpu_version != VERSION_2D_GRID  && config->gpu_version != VERSION_GRID_LINEARIZED)
    {
        init_random_state<<<blocksPerGrid,threadsPerBlock>>>(time(NULL),config->num_of_tests,states);
        cudaDeviceSynchronize();
        CHECK_KERNELCALL();

    }else if(config->gpu_version == VERSION_2D_GRID)
    {
        blocksPerGrid.y = num_of_blocks2D;
        threadsPerBlock.y = config->block_dim;
        init_random_state2D<<<blocksPerGrid,threadsPerBlock>>>(time(NULL),states, config->max_cores, config->num_of_tests);
        cudaDeviceSynchronize();
        CHECK_KERNELCALL();
    }else if(config->gpu_version == VERSION_DYNAMIC)
    {
        blocksPerGrid.y = 1;
        threadsPerBlock.y = config->block_dim;
        init_random_state2D<<<blocksPerGrid,threadsPerBlock>>>(time(NULL),states, config->max_cores, config->num_of_tests);
        cudaDeviceSynchronize();
        CHECK_KERNELCALL();

    }
    sim_state.rand_states = states;

    //----------------------------------------------------------------------
    //-----KERNEL VERSION SELECTOR------------------------------------------
    //----------------------------------------------------------------------
    //Execute Montecarlo simulation on GPU//,
    //TODO USE C++ TEMPLATE TO AVOID WRITING SAME CODE MULTIPLE TIME (eg at runtime modify the function to change mode)


    //---------------------------------------------------------------------------------
    //-----REDUX VERSION---------------------------------------------------------------
    //-----Basic Parallel reduction with shared memory and global reduction------------
    //---------------------------------------------------------------------------------
    if(config->gpu_version == VERSION_REDUX)
    {
        montecarlo_simulation_cuda_redux<<<blocksPerGrid,threadsPerBlock,config->block_dim*sizeof(float)>>>(sim_state,*config,sumTTF_GPU,sumTTFx2_GPU);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
    }
    //---------------------------------------------------------------------------------
    //-----COALESCED VERSION-----------------------------------------------------------
    //-----Basic Parallel reduction with shared memory and global reduction------------
    //-----Data are distributed in a coalesced fashion---------------------------------
    //---------------------------------------------------------------------------------
    else if(config->gpu_version == VERSION_COALESCED)
    {
        printf("COALESCED\n");//DA FIXARE PROBABILMENTE, RISULTATO VIENE NAN
        montecarlo_simulation_cuda_redux_coalesced<<<blocksPerGrid,threadsPerBlock,config->block_dim*sizeof(float)>>>(sim_state,*config,sumTTF_GPU,sumTTFx2_GPU);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
    }
    //---------------------------------------------------------------------------------
    //-----STRUCT SHARED VERSION-------------------------------------------------------
    //-----Basic Parallel reduction with shared memory and global reduction------------
    //-----Data are contained in an array of structs-----------------------------------
    //---------------------------------------------------------------------------------
    else if(config->gpu_version == VERSION_STRUCT_SHARED)
    {
        printf("STRUCT\n");
        montecarlo_simulation_cuda_redux_struct<<<blocksPerGrid,threadsPerBlock,config->block_dim*sizeof(float)>>>(sim_state,*config,sumTTF_GPU,sumTTFx2_GPU);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
    }
    //---------------------------------------------------------------------------------
    //-----DYNAMIC PROGRAMMING VERSION-------------------------------------------------
    //-----Each Thread simulate a single core of a specific walk (2D grid)-------------
    //-----PROBLEM WITH KERNEL OVERHEAD------------------------------------------------
    //---------------------------------------------------------------------------------
    else if(config->gpu_version == VERSION_DYNAMIC)
    {
        printf("DINAMIC\n");
        float* TTF_GPU;
        CHECK(cudaMalloc(&TTF_GPU, config->num_of_tests*sizeof(float)));
        
        montecarlo_simulation_cuda_dynamic<<<blocksPerGrid,threadsPerBlock,config->block_dim*sizeof(float)>>>(sim_state,*config, TTF_GPU);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();

        //accumulate at block level
        accumulate_grid_block_level<<<num_of_blocks, config->block_dim>>>(sim_state, *config, TTF_GPU, sumTTF_GPU);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
        
    }
    //---------------------------------------------------------------------------------
    //-----2D GRID VERSION-------------------------------------------------
    //-----Each Thread simulate a single core of a specific walk (2D grid)-------------
    //-----PROBLEM WITH GRID LEVEL SYNCH AND OVERHEAD----------------------------------
    //---------------------------------------------------------------------------------
    else if(config->gpu_version == VERSION_2D_GRID){
        printf("GRID\n");

        int* min_indexes;
        CHECK(cudaMalloc(&min_indexes, num_of_blocks2D*config->num_of_tests*sizeof(int)))

        float* TTF_GPU;
        CHECK(cudaMalloc(&TTF_GPU, config->num_of_tests*sizeof(float)));
        dim3 blocksPerGrid1(num_of_blocks,num_of_blocks2D,1);
        dim3 threadsPerBlock1(config->block_dim,config->block_dim,1);
        init_grid<<<blocksPerGrid1, threadsPerBlock1>>>(sim_state, *config, TTF_GPU);

        for(int i = config->max_cores; i >= config->min_cores; i--){
        //random step + partial min
        dim3 blocksPerGrid(num_of_blocks,num_of_blocks2D,1);
        dim3 threadsPerBlock(config->block_dim,config->block_dim,1);
        prob_to_death<<<blocksPerGrid, threadsPerBlock, (config->block_dim*config->block_dim)*sizeof(int)>>>(sim_state, *config, min_indexes);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
        //global min accumulate
        do{ 
            int size = blocksPerGrid.y;
            blocksPerGrid.y = ceil((float)blocksPerGrid.y/(float)config->block_dim);
            reduce_min<<<blocksPerGrid, threadsPerBlock, (config->block_dim*config->block_dim)*sizeof(float)>>>(sim_state, *config, min_indexes, size, sim_state.times);
            CHECK_KERNELCALL();
            cudaDeviceSynchronize();
        }while(blocksPerGrid.y > config->block_dim);
        blocksPerGrid.y = num_of_blocks2D;
        grid_update_state<<<blocksPerGrid, threadsPerBlock>>>(sim_state, *config, min_indexes, TTF_GPU, i-1);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
        }
        //accumulate at block level
        accumulate_grid_block_level<<<num_of_blocks, config->block_dim, config->block_dim*sizeof(float)>>>(sim_state, *config, TTF_GPU, sumTTF_GPU);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
    }
    //---------------------------------------------------------------------------------
    //-----2D GRID VERSION-------------------------------------------------
    //-----Each Thread simulate a single core of a specific walk (2D grid)-------------
    //-----same as dynamic programming but without dynamic prog------------------------
    //---------------------------------------------------------------------------------
    else if(config->gpu_version == VERSION_GRID_LINEARIZED)
    {
        printf("GRID LINEARIZED\n");
        config->block_dim   = 256;
        num_of_blocks       = config->num_of_tests;

        CHECK(cudaMalloc(&sumTTF_GPU    , num_of_blocks*sizeof(float)));   //Allocate Result memory
        
        dim3 blocksPerGrid(num_of_blocks,1,1);
        dim3 threadsPerBlock(config->block_dim,1,1);

        montecarlo_simulation_cuda_grid_linearized<<<blocksPerGrid,threadsPerBlock>>>(sim_state,*config, sumTTF_GPU,sumTTFx2_GPU,time(NULL));
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
    }
    //---------------------------------------------------------------------------------
    //-----2D GRID VERSION-------------------------------------------------
    //-----Each Thread simulate a single core of a specific walk (2D grid)-------------
    //-----same as dynamic programming but without dynamic prog------------------------
    //---------------------------------------------------------------------------------
    else if (config->gpu_version == VERSION_STRUCT_OPTIMIZED){
        printf("STRUCT OPT\n");
        montecarlo_simulation_cuda_redux_struct_optimized<<<blocksPerGrid,threadsPerBlock,config->block_dim*sizeof(float)>>>(sim_state,*config,sumTTF_GPU,sumTTFx2_GPU);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
    }

    //---------------------------------------------------------------------------------
    //-----GLOBAL REDUCTION------------------------------------------------------------
    //-----Sum all the partial SUMTTF of different blocks------------------------------
    //---------------------------------------------------------------------------------
    
    collect_res_gpu_grid<<<1, 1024, 1024*sizeof(float)>>>(sumTTF_GPU, sumTTF_GPU_final, num_of_blocks);     //Global reduction
    CHECK_KERNELCALL(); 
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(&res, sumTTF_GPU_final, sizeof(float), cudaMemcpyDeviceToHost));                       //Copy back results on CPU
    *sumTTF = (double) res;                                                                                 //cast the result
    
    
    
    //---------------------------------------------------------------------------------
    //----FREE CUDA MEMORY-------------------------------------------------------------
    //---------------------------------------------------------------------------------
    CHECK(cudaFree(sumTTF_GPU));
    CHECK(cudaFree(sumTTFx2_GPU));
    CHECK(cudaFree(states));
    free_simulation_state(&sim_state,*config);
    cudaDeviceReset();
}
    /*
    float temp[num_of_blocks];
    CHECK(cudaMemcpy(temp, sumTTF_GPU, (num_of_blocks)*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i< num_of_blocks; i++){
        *sumTTF += temp[i];
    }

    *sumTTFx2 = (*sumTTF)*(*sumTTF);    //TODO sumTTF squared
    */
    
    
    
    
#endif