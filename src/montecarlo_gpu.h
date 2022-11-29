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
        CUDA_DEBUG_MSG("Core index = [%d,%d,%d]\n",23,getIndex(i,config.num_of_tests),33);
        return;

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

__global__ void montecarlo_simulation_cuda_dynamic(simulation_state sim_state,configuration_description config){

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = tid * (config.rows * config.cols);

    extern __shared__ float partial_sumTTF[];

    curandState_t *states = sim_state.rand_states;

    sim_state.sumTTF[blockIdx.x] = 0;
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

        left_cores = config.rows * config.rows;

        totalTime = 0;
        minIndex = 0;

        //Initialization
        for (j = 0; j < config.max_cores; j++) { //Also parallelizable (dont know if usefull, only for big N)
            int index = offset + j;
            currR[index]  = 1;
            indexes[index] = index;
        }

        while (left_cores >= config.min_cores) {
            minIndex = -1;

            //-----------Redistribute Loads among alive cores----------
            double distributedLoad = (double)config.initial_work_load * (double)config.max_cores / (double)left_cores;
            for (j = 0; j < left_cores; j++) {
                int index = indexes[offset + j];    //index of alive cores
                loads[index] = distributedLoad;
            }
            //Set Load of lastly dead core to 0
            loads[indexes[offset + left_cores-1]] = 0;//Potrebbe essere qui lerrore

            //---------Launch Child kernel to calculate a random walk step for this thread
            //WHY I CANT RUN A 2D child KERNEL????
            int block_dim = 1024;
            int num_of_blocks = (left_cores + block_dim - 1) / block_dim;

            //Call simulation step dynamicly by decreasing grid size evry time a core die
            montecarlo_simulation_step<<<num_of_blocks,block_dim>>>(sim_state,config,distributedLoad,left_cores,offset,tid);
            cudaDeviceSynchronize();

            //partial_sumTTF[threadIdx.x] = partial_sumTTF[threadIdx.x] + stepT;
            //TODO SAVE RESULT OF STEPT TO GLOBAL OR SHARED MEMORY 

            //---------Reduce number of alive core in this simulation
            left_cores--;
        }
        //END SIMULATION-----------------------------

        //Sync the threads
        __syncthreads();

        //Acccumulate the results of this block
        accumulate(partial_sumTTF,blockDim.x,config.num_of_tests);

        //Add the partial result of this block to the final result
        __syncthreads();
        if(threadIdx.x == 0){
            sim_state.sumTTF[blockIdx.x] = partial_sumTTF[0]; //Each block save its result in its "ID"
        }

        //TODO ACCUMULATE ALSO SUMTTFx2

    }
}

__device__ int getMatrixIndex(int col,int row,int id,int offset){
    int i;
    return offset;
}




__global__ void montecarlo_simulation_cuda_redux_coalesced(simulation_state sim_state,configuration_description config ,float * sumTTF_res,float * sumTTFx2_res){
    CUDA_DEBUG_MSG("Initialize");
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

        left_cores = config.max_cores;
        totalTime = 0;
        minIndex = 0;

        for (j = 0; j < config.max_cores; j++) { //Also parallelizable (dont know if usefull, only for big N)
            int index = getIndex(j,config.num_of_tests);
            currR[index]    = 1;
            indexes[index]  = j;
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
                int index = getIndex(j,config.num_of_tests);
                loads[index] = distributedLoad;    //Each TH will access contiguous loads cell based on tid
            }

            //DA FIXARE PROBABILMENTE, RISULTATO VIENE NAN
            tempModel_version2(loads, temps,indexes,distributedLoad,left_cores, config.rows, config.cols,config.num_of_tests);
            //Versione non ottimizzata
            //tempModel(loads, temps,indexes,left_cores, rows, cols,offset); 

            //-----------Random Walk Step computation-------------------------
            for (j = 0; j < left_cores; j++) {
                //Random  is in range [0 : currR[j]]
                int index = getIndex(j,config.num_of_tests); //Current alive core
                if(tid == 0){
                    printf("[%d--%f]",j,currR[index]);
                }
                random =(double)curand_uniform(&states[tid])* currR[index]; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                double alpha = getAlpha(temps[index]);
                t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                eqT = alpha * pow(-log(currR[index]), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
                t = t - eqT;

                //the difference between the two values represents the time elapsed from the previous failure to the current failure
                //(we will sum to the total time the minimum of such values)

                if (minIndex == -1 || (minIndex != -1 && t < stepT)) {

                    minIndex = j;//Set new minimum index
                    stepT = t;   //set new minimum time as timeStep
                } //TODO ADD A CHECK ON MULTIPLE FAILURE IN THE SAME INSTANT OF TIME.
            }
            if(tid == 0){
                printf("\n");
            }
            //-------Check if No failed core founded-----------
            if (minIndex == -1) {
                //TODO HOW THROW ERRORS IN CUDA?????
                return;
            }
            if(tid == 0){
                printf("STEP:%f\n",stepT);
                printf("DEAD CORE: %d\n",minIndex);
            }
            //---------UPDATE TOTAL TIME-----------------
            //Current simulation time
            partial_sumTTF[threadIdx.x] = partial_sumTTF[threadIdx.x] + stepT;
            //totalTime = totalTime + stepT;
            //---------UPDATE Configuration-----------------
            if (left_cores > config.min_cores) {
                //Swap all cells 
                swapState(sim_state,minIndex,left_cores,config.max_cores);
                // compute remaining reliability for working cores
                for (j = 0; j < left_cores; j++) {
                    int index = getIndex(j,config.num_of_tests); //Current alive core
                    double alpha = getAlpha(temps[index]);
                    eqT = alpha * pow(-log(currR[index]), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
                    currR[index] = exp(-pow((stepT + eqT) / alpha, BETA));
                }
            }
            //Set Load of lastly dead core to 0
            loads[getIndex(left_cores-1,config.max_cores)] = 0;//Potrebbe essere qui lerrore
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
    CUDA_DEBUG_MSG("Initialize\n");
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

        //Change to struct
        core_state * cores   = sim_state.core_states;

        left_cores = config.max_cores;
        totalTime = 0;
        minIndex = 0;

        for (j = 0; j < config.max_cores; j++) { //Also parallelizable (dont know if usefull, only for big N)
            int index = getIndex(j,config.num_of_tests);
            //Coalesced access to struct
            core_state curr_core = cores[index]; 
            //Local Changes on Register memory (no global mem access!!)
            curr_core.curr_r    = 1;
            curr_core.real_index  = j;
            curr_core.load    = config.initial_work_load;
            //Coalescet write of core state
            cores[index] = curr_core;

            /*
            cores[index].curr_r    = 1;
            cores[index].real_index  = j;
            cores[index].load    = config.initial_work_load;
            */

        }
        CUDA_DEBUG_MSG("Start Simulation\n");
        while (left_cores >= config.min_cores) {
            minIndex = -1;

            //-----------Redistribute Loads among alive cores----------
            double distributedLoad = (double)config.initial_work_load * (double)config.max_cores / (double)left_cores;

            //TODO UPDATE THE TEMPERATURE
            tempModel_gpu_struct(sim_state,config,distributedLoad,left_cores);

            CUDA_DEBUG_MSG("Temp\n");
            //Set Load of alive cores
            for (j = 0; j < left_cores; j++) {
                int index = getIndex(j,config.num_of_tests); //Current alive core
                core_state curr_core = cores[index]; //Coalesced access to struct
                random =(double)curand_uniform(&states[tid])* curr_core.curr_r; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                double alpha = getAlpha(curr_core.temp);
                t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                eqT = alpha * pow(-log(curr_core.curr_r), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
                t = t - eqT;

                //the difference between the two values represents the time elapsed from the previous failure to the current failure
                //(we will sum to the total time the minimum of such values)

                if (minIndex == -1 || (minIndex != -1 && t < stepT)) {

                    minIndex = j;//Set new minimum index
                    stepT = t;   //set new minimum time as timeStep
                } //TODO ADD A CHECK ON MULTIPLE FAILURE IN THE SAME INSTANT OF TIME.
            }

             //---------UPDATE TOTAL TIME-----------------
            //Current simulation time
            partial_sumTTF[threadIdx.x] = partial_sumTTF[threadIdx.x] + stepT;
            //totalTime = totalTime + stepT;
            //---------UPDATE Configuration-----------------
            if (left_cores > config.min_cores) {
                //Swap all cells  (TODO SWAP WITH STRUCT)
                swapStateStruct(sim_state,minIndex,left_cores,config.max_cores);
                // compute remaining reliability for working cores
                for (j = 0; j < left_cores; j++) {
                    int index = getIndex(j,config.num_of_tests); //Current alive core
                    //Coalesced Read of core state struct
                    core_state curr_core = cores[index];
                    //Local changes to the core state (in register memory!!)
                    double alpha = getAlpha(curr_core.temp);
                    eqT = alpha * pow(-log(curr_core.curr_r), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
                    curr_core.curr_r = exp(-pow((stepT + eqT) / alpha, BETA));
                    //Coalesced Write of results in memory
                    cores[index] = curr_core;
                }
            }

            //Set Load of lastly dead core to 0
            int index = getIndex(left_cores-1,config.num_of_tests); //Current alive core
            core_state curr_core = cores[index];
            curr_core.load = 0;
            cores[index] = curr_core; //TODO , IS THIS WORTH IN RESPECT TO cores[index].load = 0???

            //Decrement counter of alive core
            left_cores--; //Reduce number of alive core in this simulation
        }

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

__device__ float accumulate_min(float *input, size_t dim, configuration_description config) //TODO look at what can be done with warp_reduce
//assuming a x_index first organization of the warps, no better improvement can be done
//IDEA can be to flip (x, y) dimensions with (cores, walks)
{
    size_t threadId = threadIdx.y;
    size_t global_offset = (threadIdx.x + blockIdx.x * blockDim.x)*config.max_cores + blockIdx.y * blockDim.y;
    if (dim > 32)
    {
        size_t offset = dim/2;
        for (size_t i = dim / 2; i > 0; i >>= 1)
        {
            if(threadId < i)
                input[threadId + global_offset] = min(input[threadId + global_offset], input[threadId + global_offset + offset]);
            offset >>= 1;
            __syncthreads();
        }
    }

    return input[global_offset];
}

__device__ float prob_to_death(simulation_state sim_state, configuration_description config, int walk_id, int core_id)
{
    int global_id = walk_id*config.max_cores + core_id;
    core_state s = sim_state.core_states[global_id];
    curandState_t *states = sim_state.rand_states;
    double random = (double)curand_uniform(&states[global_id])* s.curr_r; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
    float alpha = getAlpha(s.temp);
    float t = alpha * pow(-log(random), (float) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
    float eqT = alpha * pow(-log(s.curr_r), (float) 1 / BETA); //elapsed time from 0 to obtain the previous R value
    sim_state.times[global_id]= t - eqT;
    if(!sim_state.core_states[global_id].alive)
        sim_state.times[global_id] = FLT_MAX;
    __syncthreads();
    float min = accumulate_min(sim_state.times, blockDim.y, config);
    if(min == t-eqT) {          //what about 2 or more cores have same t-eqT
        sim_state.core_states[global_id].alive = false;
        sim_state.core_states[global_id].load = 0;
    }
    return min;
}

__device__ void update_state(simulation_state sim_state, configuration_description config, int walk_id, int core_id, float stepT){
    int global_id = walk_id*config.max_cores + core_id;
    core_state* s = &sim_state.core_states[global_id];
    if(s->alive){
        float alpha = getAlpha(s->temp);
        float eqT = alpha * pow(-log(s->curr_r), (float) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
        s->curr_r = exp(-pow((stepT + eqT) / alpha, BETA));
    }
}

__global__ void montecarlo_simulation_cuda_grid(simulation_state sim_state, configuration_description config ,float * sumTTF_res,float * sumTTFx2_res) {
    unsigned int core_id = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int walk_id = threadIdx.x + blockIdx.x * blockDim.x;

    int index = core_id + walk_id*config.max_cores;
    sim_state.core_states[index].load = config.initial_work_load;
    sim_state.core_states[index].curr_r = 1;
    sim_state.core_states[index].alive = true;

    int left_cores = config.max_cores;
    extern __shared__ float partial_sumTTF[];

    while(left_cores >= config.min_cores){
        float min;
        if(sim_state.core_states[index].alive) {
            double distributedLoad = config.initial_work_load * config.max_cores / left_cores;
            sim_state.core_states[index].load = distributedLoad;
            tempModel_gpu_grid(sim_state, config, core_id, walk_id);
        }
        min = prob_to_death(sim_state, config, walk_id, core_id);
        update_state(sim_state, config, walk_id, core_id, min);
        if(core_id == 0)
            partial_sumTTF[threadIdx.x] += min;
        left_cores--;
    }
    //TODO global_accumulate parallelized properly
    if(core_id == 0 && threadIdx.x == 0)
        for (size_t i = 0; i<blockDim.x; i++)
            sumTTF_res[blockIdx.x] += partial_sumTTF[i];

    //synch first
    //TODO accumulate sumTTF
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
    //TODO contain all the malloc/necessary code to call the kernel
    //----CUDA variables:------------
    float *sumTTF_GPU;   //GPU result sumTTF
    float *sumTTF_GPU_final;   //GPU result sumTTF
    float *sumTTFx2_GPU; //GPU result sumTTFx2
    float res;
    simulation_state sim_state;
    curandState_t *states;

    int num_of_blocks = (config->num_of_tests+config->block_dim-1)/config->block_dim;
    int num_of_blocks2D = (config->max_cores+config->block_dim-1)/config->block_dim;
    //----MEMORY ALLOCATION:---------
    CHECK(cudaMalloc(&sumTTF_GPU    , num_of_blocks*sizeof(float)));   //Allocate Result memory
    CHECK(cudaMalloc(&sumTTFx2_GPU  , sizeof(float))); //Allocate Result memory
    CHECK(cudaMalloc(&states        , config->num_of_tests*sizeof(curandState_t))); //Random States array
    CHECK(cudaMalloc(&sumTTF_GPU_final, sizeof(float)));   //Allocate Result memory

    allocate_simulation_state_on_device(&sim_state,*config);
    sim_state.sumTTF   = sumTTF_GPU;
    sim_state.sumTTFx2 = sumTTFx2_GPU;

    //----Declare Grid Dimensions:---------
    dim3 blocksPerGrid(num_of_blocks,1,1);
    dim3 threadsPerBlock(config->block_dim,1,1);

    //----KERNELS CALL -------------------
    //Inizialize Random states
    init_random_state<<<blocksPerGrid,threadsPerBlock>>>(time(NULL),states);
    cudaDeviceSynchronize();
    CHECK_KERNELCALL();
    //Associate random states with the sim_state
    sim_state.rand_states = states;
    //Execute Montecarlo simulation on GPU//,
    //TODO USE C++ TEMPLATE TO AVOID WRITING SAME CODE MULTIPLE TIME (eg at runtime modify the function to change mode)
    if(config->gpu_version == VERSION_REDUX){
        montecarlo_simulation_cuda_redux<<<blocksPerGrid,threadsPerBlock,config->block_dim*sizeof(float)>>>(sim_state,*config,sumTTF_GPU,sumTTFx2_GPU);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
    }else if(config->gpu_version == VERSION_COALESCED){
        printf("COALESCED\n");//DA FIXARE PROBABILMENTE, RISULTATO VIENE NAN
        montecarlo_simulation_cuda_redux_coalesced<<<blocksPerGrid,threadsPerBlock,config->block_dim*sizeof(float)>>>(sim_state,*config,sumTTF_GPU,sumTTFx2_GPU);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
    }else if(config->gpu_version == VERSION_STRUCT_SHARED){
        printf("STRUCT\n");
        montecarlo_simulation_cuda_redux_struct<<<blocksPerGrid,threadsPerBlock,config->block_dim*sizeof(float)>>>(sim_state,*config,sumTTF_GPU,sumTTFx2_GPU);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
    }
    else if(config->gpu_version == VERSION_DYNAMIC){
        printf("DINAMIC\n");
        montecarlo_simulation_cuda_dynamic<<<blocksPerGrid,threadsPerBlock,config->block_dim*sizeof(float)>>>(sim_state,*config);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
    }else if(config->gpu_version == VERSION_2D_GRID){
        printf("GRID\n");
        dim3 blocksPerGrid(num_of_blocks,num_of_blocks2D,1);
        dim3 threadsPerBlock(config->block_dim,config->block_dim,1);
        montecarlo_simulation_cuda_grid<<<blocksPerGrid,threadsPerBlock,config->block_dim*sizeof(float)>>>(sim_state,*config, sumTTF_GPU,sumTTFx2_GPU);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
    }
    //MODE 3 -> STRUCT + COALESCED + SHARED MEM
    //MODE 4 -> FIXED 2D GRID OF THREAD INSTEAD OF DYNAMIC PROGRAMMING

    //WARNING!!! DOSNT COUNT PADDING IN ACCUMULATE
    collect_res_gpu<float><<<1,num_of_blocks/2>>>(sumTTF_GPU,sumTTF_GPU_final,num_of_blocks);
    //collect_res_gpu_x<<<1,threadsPerBlock>>>(sumTTF_GPU,sumTTF_GPU_final,blocksPerGrid.x);
    CHECK_KERNELCALL();
    cudaDeviceSynchronize();
    //----Copy back results on CPU-----------
    CHECK(cudaMemcpy(&res, sumTTF_GPU_final, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    *sumTTF = (double) res;
    CHECK(cudaMemcpy(&res, sumTTFx2_GPU, sizeof(float), cudaMemcpyDeviceToHost));
    *sumTTFx2 = (double) res;

    //----FREE CUDA MEMORY------------------
    CHECK(cudaFree(sumTTF_GPU));
    CHECK(cudaFree(states));
    free_simulation_state(&sim_state,*config);
    cudaDeviceReset();
}

#endif