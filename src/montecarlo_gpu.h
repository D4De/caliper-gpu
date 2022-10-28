#ifndef CALIPER_GPU
#define CALIPER_GPU
    #define CUDA
    //CUDA-UTILS
    #include "cuda-utils/cuda_helper.h"
    #include "cuda-utils/parallel_reduction.h"
    #include "cuda-utils/simulation_config.h"
    //UTILS
    #include "utils/benchmark_helper.h"
    #include "utils/utils.h"
    //THERMAL MODEL
    #include "./thermal_model.h"



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
    bool*  alives  = sim_state.alives; //TODO allocate those array on global memory on host side
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
    bool*  alives  = sim_state.alives; //TODO allocate those array on global memory on host side
    int*   indexes = sim_state.indexes;

    //Update Temperature Model
    tempModel_gpu(loads,temps,indexes,left_cores,distributed_wl,config.cols,config.rows,offset);

    __shared__ float stepT[1024];
    int min_index;
    //Determine which core die in this iteration (by calculating TTF)
    montecarlo_random_walk_step(sim_state,config,stepT,&min_index,left_cores,offset); //Work only if leftcores<blockdim
    
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
        bool*  alives  = sim_state.alives; //TODO allocate those array on global memory on host side
        int*   indexes = sim_state.indexes;

        left_cores = config.rows * config.rows;
        
        totalTime = 0;
        minIndex = 0;

        //Initialization
        for (j = 0; j < config.max_cores; j++) { //Also parallelizable (dont know if usefull, only for big N)
            int index = offset + j;
            currR[index]  = 1;
            alives[index] = true;
            indexes[index] = index;
        }

        while (left_cores >= config.min_cores) {
             minIndex = -1;

        //-----------Redistribute Loads among alive cores----------
            double distributedLoad = (double)config.initial_work_load * (double)config.max_cores / (double)left_cores;
            for (j = 0; j < left_cores; j++) {
                int index = indexes[offset + j];
                loads[index] = distributedLoad;
            }
            //Set Load of lastly dead core to 0
            loads[offset + left_cores-1] = 0;

        //---------Launch Child kernel to calculate a random walk step for this thread  
            //WHY I CANT RUN A 2D child KERNEL????
            int block_dim = 1024;
            int num_of_blocks = (left_cores + block_dim - 1) / block_dim;

          //Call simulation step dynamicly by decreasing grid size evry time a core die
            __syncthreads();
            montecarlo_simulation_step<<<num_of_blocks,block_dim>>>(sim_state,config,distributedLoad,left_cores,offset,tid);
            cudaDeviceSynchronize();
            __syncthreads();          

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

__global__ void montecarlo_simulation_cuda_redux(simulation_state sim_state,curandState_t *states, int num_of_tests,int block_dim,int max_cores,int min_cores,int rows,int cols, float wl,float * sumTTF_res,float * sumTTFx2_res){
 
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = tid * (rows * cols);
    //extern __shared__ unsigned int tmp_sumTTF[];
    //extern __shared__ unsigned int tmp_sumTTFx2[]; //??? dont know if necessary
    extern __shared__ float partial_sumTTF[];
    
    sumTTF_res[blockIdx.x] = 0;
    partial_sumTTF[threadIdx.x] = 0;

    if(tid<num_of_tests){
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
        bool*  alives  = sim_state.alives; //TODO allocate those array on global memory on host side
        int*   indexes = sim_state.indexes;

        left_cores = max_cores;
        totalTime = 0;
        minIndex = 0;

        for (j = 0; j < max_cores; j++) { //Also parallelizable (dont know if usefull, only for big N)
            int index = offset + j;
            currR[index]  = 1;
            alives[index] = true;
            indexes[index] = index;
        }

        while (left_cores >= min_cores) {
             minIndex = -1;

        
        //-----------Redistribute Loads among alive cores----------
            double distributedLoad = (double)wl * (double)max_cores / (double)left_cores;

            //To improve both performance and divergence we remove the if(alive[j])
            //by using the indexes of alive cores instead of if(alive[j])

            //Set Load of alive cores
            for (j = 0; j < left_cores; j++) {
                int index = indexes[offset + j];
                loads[index] = distributedLoad;
            }
            //Set Load of lastly dead core to 0
            loads[offset + left_cores-1] = 0;

            int block_dim = 256;
            //TODO optimize tempModel by computing temp only of alive cores
            //CHECK HOW DYNAMICALY ALLOCATE ARRAY INSIDE KERNEL (into registers and not global memory)

            //FUNZIONE OTTIMIZZATA DA FIXARE(??)
            tempModel(loads, temps,indexes,left_cores, rows, cols,offset); 
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
            if (left_cores > min_cores) {
                alives[minIndex] = false;
                swap_core_index(sim_state.indexes,minIndex,left_cores,offset); //move index of dead core to end
                // compute remaining reliability for working cores
                for (j = 0; j < left_cores; j++) {
                        int index = indexes[offset+j]; //Current alive core
                    	double alpha = getAlpha(temps[index]);
                        eqT = alpha * pow(-log(currR[index]), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
                        currR[index] = exp(-pow((stepT + eqT) / alpha, BETA));
                }
            }
            left_cores--; //Reduce number of alive core in this simulation
        }//END SIMULATION-----------------------------
        //Sync the threads
        __syncthreads();
        
        //Acccumulate the results of this block
        accumulate<float>(partial_sumTTF,blockDim.x,num_of_tests);

        //Add the partial result of this block to the final result
        __syncthreads();
        if(threadIdx.x == 0){
            float x;
            //atomicAdd(sumTTF_res,partial_sumTTF[0]); 
            //USING ATOMIC ADD-> SAME RESULT AS CPU, WITH GLOBAL REDUCE.... NOT:.. CHECK WHY
            
            //Each Thread 0 assign to his block cell the result of its accumulate
            sumTTF_res[blockIdx.x] = partial_sumTTF[0]; //Each block save its result in its "ID"
        }
        
        //TODO ACCUMULATE ALSO SUMTTFx2

    }
}

/**
 * -Preparation and launch of Simulation
 * -Allocate Global Memory
 * -Initialize random State
 * -Launch Simulation
 * -Collect results from global mem
*/
void montecarlo_simulation_cuda_launcher(int num_of_tests,int max_cores,int min_cores,int rows,int cols, float wl,double * sumTTF,double * sumTTFx2,int block_dim,int version)
 {
    //TODO contain all the malloc/necessary code to call the kernel
           //----CUDA variables:------------
        float *sumTTF_GPU;   //GPU result sumTTF
        float *sumTTF_GPU_final;   //GPU result sumTTF
        float *sumTTFx2_GPU; //GPU result sumTTFx2
        float res;
        simulation_state sim_state;
        configuration_description config;
        curandState_t *states;

        int num_of_blocks = (num_of_tests+block_dim-1)/block_dim;
        //----MEMORY ALLOCATION:---------
        CHECK(cudaMalloc(&sumTTF_GPU    , num_of_blocks*sizeof(float)));   //Allocate Result memory
        CHECK(cudaMalloc(&sumTTFx2_GPU  , sizeof(float))); //Allocate Result memory
        CHECK(cudaMalloc(&states        , num_of_tests*sizeof(curandState_t))); //Random States array
        CHECK(cudaMalloc(&sumTTF_GPU_final, sizeof(float)));   //Allocate Result memory

        allocate_simulation_state_on_device(&sim_state,rows,cols,num_of_tests);
        sim_state.sumTTF   = sumTTF_GPU;
        sim_state.sumTTFx2 = sumTTFx2_GPU;

        setup_config(&config,num_of_tests,rows*cols,min_cores,wl,rows,cols,block_dim,version);
        //----Declare Grid Dimensions:---------
        dim3 blocksPerGrid(num_of_blocks,1,1);
        dim3 threadsPerBlock(block_dim,1,1);

        //----KERNELS CALL -------------------
        //Inizialize Random states
        init_random_state<<<blocksPerGrid,threadsPerBlock>>>(time(NULL),states);
        cudaDeviceSynchronize();
        CHECK_KERNELCALL();
        //Associate random states with the sim_state
        sim_state.rand_states = states;
        //Execute Montecarlo simulation on GPU//,
        if(version == 0){
            montecarlo_simulation_cuda_redux<<<blocksPerGrid,threadsPerBlock,block_dim*sizeof(float)>>>(sim_state,states,num_of_tests,block_dim,max_cores,min_cores,rows,cols,wl,sumTTF_GPU,sumTTFx2_GPU);
            CHECK_KERNELCALL();
            cudaDeviceSynchronize();
        }else if(version == 1){
            printf("DYNAMIC\n");
            montecarlo_simulation_cuda_dynamic<<<blocksPerGrid,threadsPerBlock,block_dim*sizeof(float)>>>(sim_state,config);
            CHECK_KERNELCALL();
            cudaDeviceSynchronize();
        }
        
         //WARNING!!! DOSNT COUNT PADDING IN ACCUMULATE
        collect_res_gpu<float><<<1,num_of_blocks/2>>>(sumTTF_GPU,sumTTF_GPU_final,num_of_blocks);
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
        free_simulation_state(&sim_state);
        cudaDeviceReset();
 }

#endif