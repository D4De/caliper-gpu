#ifndef CALIPER_GPU
#define CALIPER_GPU
    #define CUDA
    #include "cuda_helper.h"
    #include "./thermal_model.h"
    #include "./benchmark_helper.h"
    #include "./utils.h"
    #include "cuda_helper.h"
//#define debugMSG(msg) if(printf(msg)
struct simulation_state{
    float * currR;
    float * temps;
    float * loads;
    int *   indexes;
    bool *  alives;
};


__global__ void init(unsigned int seed, curandState_t *states){
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &states[tid]);
}

__global__ void pick_next_core_to_die(curandState_t *states,bool alives[],double temps[],double currR,int tid){
    //TODO parallelize the inner for cycle (to find the stepT and minIndex (core to die this turn))

    //IDEA, use the adomicCAS (compare and swap) to find the minimum among the stepT
}

__global__ void update_R_function(bool* alives,double* currR,double* temps,double stepT)
{
    //TODO parallelize the update Configuration for cycle
}

/**
 *  SPLIT the temps into a 2D grid to calculate all temps together
 *  WORK IN PROGRESS
 * https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda
 * https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/
 */
__global__ void tempModel_gpu(float* loads, float* temps,bool* alives,float distributed_wl,int cols, int rows,int offset){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    //TODO calculate loads directly here using alive flag (avoid an O(N) for cycle)

    //Shared memory for loads 0r calculate them real time
    if(i<cols && j<rows){ // CHECK TO BE INSIDE THE GRID
        //Calculate Temp Model for a specific Core
        float temp = 0;
        int k,h;
        for (k = -1; k < 2; k++)
                    for (h = -1; h < 2; h++)
                        if ((k != 0 || h != 0) && k != h && k != -h && i + k >= 0 && i + k < rows && j + h >= 0 && j + h < cols){
                            temp += loads[offset+(i + k)*cols + (j + h)] * NEIGH_TEMP;
                        }
                temps[offset+i*cols+j] = ENV_TEMP + loads[offset+i*cols+j] * SELF_TEMP + temp;
    }

}
__device__ void warpReduce(volatile unsigned int *input,size_t threadId)
{
	input[threadId] += input[threadId + 32];
	input[threadId] += input[threadId + 16];
	input[threadId] += input[threadId + 8];
	input[threadId] += input[threadId + 4];
	input[threadId] += input[threadId + 2];
	input[threadId] += input[threadId + 1];
}
__device__ void warpReduce(volatile float *input,size_t threadId)
{
	input[threadId] += input[threadId + 32];
	input[threadId] += input[threadId + 16];
	input[threadId] += input[threadId + 8];
	input[threadId] += input[threadId + 4];
	input[threadId] += input[threadId + 2];
	input[threadId] += input[threadId + 1];
}
__device__ float accumulate(unsigned int *input, size_t dim)
{
	size_t threadId = threadIdx.x;
	if (dim > 32)
	{
		for (size_t i = dim / 2; i > 32; i >>= 1)
		{
			
            if (threadId  < i)
            {
                input[threadId] += input[threadId + i];
            }
            __syncthreads();
        }
	}
	if (threadId < 32)
		warpReduce(input, threadId);
	__syncthreads();
	return input[0];
}
__global__ void collect_res_gpu(float *input,float* result, int numOfBlocks) // compute the final reduction
{

    //*result = input[0];
    //return;
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned i;
    __shared__ float localVars[1024];   //Max num possible num of blocks
    localVars[tid]=input[tid];//Save input in shared memory
    __syncthreads();
    localVars[tid +  numOfBlocks/2] = input[tid +  numOfBlocks/2];
    __syncthreads();

    if(tid < numOfBlocks/2){
        //IF Remaining cells are more then 32
        for (i = numOfBlocks / 2; i > 32; i >>= 1) // compute the parallel reduction for the collected data
        {
            if (tid < i)//Each cycle use half of the threads
            {
                localVars[tid] += localVars[tid + i];
                // eg: 1->10 || 1-5 || 1-2 || 
                //eg2: 2->11 || 2-6 || 2-3 ||
            }
            __syncthreads();
        }

        if(tid<32)
            warpReduce(localVars,tid);
        __syncthreads();

        if(tid==0)
            *result = localVars[tid];
        __syncthreads();
    }    
}

/**
 * Swap the dead core to the end of the array
 * 
 * This allow to have alives core indexes saved on the first portion of the array
 * [Array of index] = [Alive cores][Dead Cores]
*/
__device__ void swap_core_index(int* cores,int dead_index,int size,int offset){
    int tmp = cores[offset+size-1];

    cores[offset+size-1] = cores[dead_index]; //Swap dead index to end
    cores[dead_index] = tmp;         

}
__global__ void montecarlo_simulation_cuda(simulation_state sim_state,curandState_t *states, int num_of_tests,int block_dim,int max_cores,int min_cores,int rows,int cols, float wl,float * sumTTF_res,float * sumTTFx2_res){
 
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = tid * (rows * cols);
    //extern __shared__ unsigned int tmp_sumTTF[];
    //extern __shared__ unsigned int tmp_sumTTFx2[]; //??? dont know if necessary
    extern __shared__ unsigned int partial_sumTTF[];
    
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

        /*
       currR   = (double*) malloc(rows*cols    * sizeof(double));
        loads   = (double*) malloc(rows*cols    * sizeof(double));
        temps   = (double*) malloc(rows*cols    * sizeof(double));
        alives  = (bool*)   malloc(rows*cols    * sizeof(bool));
        inde xes = (int*)    malloc(rows*cols    * sizeof(int));
        

        double currR[300];
        double loads[300];
        double temps[300];
        bool alives[300]; //TODO allocate those array on global memory on host side
        int indexes[300];
        */

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
           tempModel(loads, temps, rows, cols,offset);

            int row_blocks = (rows+block_dim-1)/block_dim;
            int col_blocks = (cols+block_dim-1)/block_dim;
            dim3 blocksPerGrid(row_blocks,col_blocks,1);
            dim3 threadsPerBlock(block_dim,block_dim,1);
            //tempModel_gpu<<<blocksPerGrid,threadsPerBlock>>>(loads,temps,alives,distributedLoad,cols,rows,offset);
            __syncthreads();

            //-----------Random Walk Step computation-------------------------
            for (j = 0; j < left_cores; j++) {
                //Random  is in range [0 : currR[j]]
                int index = indexes[offset + j]; //Current alive core
                random =(double)curand_uniform(&states[tid])* currR[index]; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                double alpha = getAlpha(temps[index]);
                t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                eqT = alpha * pow(-log(currR[index]), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
                t = t - eqT;

                if(tid==1){
                    //printf("%d -> Death Time: %f ->(%f)(%f) -- [%f][%f][%f]\n",j,t,random,currR[j],alpha_rounded,temps[j],loads[j]);
                }

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
            if(tid==1){
                //double alpha = getAlpha(temps[minIndex]);
                //printf("\nDead core: \t%d (%f)->%f [%f]\n",minIndex,stepT,totalTime,alpha);
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
        accumulate(partial_sumTTF,blockDim.x);

        //Add the partial result of this block to the final result
        __syncthreads();
        if(threadIdx.x == 0){
            float x;
            //atomicAdd(sumTTF_res,partial_sumTTF[0]); 
            //USING ATOMIC ADD-> SAME RESULT AS CPU, WITH GLOBAL REDUCE.... NOT:.. CHECK WHY
            
            //Each Thread 0 assign to his block cell the result of its accumulate
            sumTTF_res[blockIdx.x] = (float)partial_sumTTF[0]; //Each block save its result in its "ID"
        }
        
        //TODO ACCUMULATE ALSO SUMTTFx2

    }
}



/**
 * Allocate Global Memory of gpu to store the simulation state
*/
void allocate_simulation_state_on_device(simulation_state* state,int rows,int cols,int num_of_iteration){
    int cells = rows*cols*num_of_iteration;
    CHECK(cudaMalloc(&state->currR    , cells*sizeof(float)));  //CurrR
    CHECK(cudaMalloc(&state->temps    , cells*sizeof(float)));  //temps
    CHECK(cudaMalloc(&state->loads    , cells*sizeof(float)));  //loads
    CHECK(cudaMalloc(&state->indexes  , cells*sizeof(int)));    //indexes
    CHECK(cudaMalloc(&state->alives   , cells*sizeof(bool)));   //alives
}

void montecarlo_simulation_cuda_launcher(int num_of_tests,int max_cores,int min_cores,int rows,int cols, float wl,double * sumTTF,double * sumTTFx2,int block_dim)
 {
    //TODO contain all the malloc/necessary code to call the kernel
           //----CUDA variables:------------
        float *sumTTF_GPU;   //GPU result sumTTF
        float *sumTTF_GPU_final;   //GPU result sumTTF
        float *sumTTFx2_GPU; //GPU result sumTTFx2
        float res;
        simulation_state sim_state;

        curandState_t *states;
        int num_of_blocks = (num_of_tests+block_dim-1)/block_dim;
        //----MEMORY ALLOCATION:---------
        CHECK(cudaMalloc(&sumTTF_GPU    , num_of_blocks*sizeof(float)));   //Allocate Result memory
        CHECK(cudaMalloc(&sumTTFx2_GPU  , sizeof(float))); //Allocate Result memory
        CHECK(cudaMalloc(&states        , num_of_tests*sizeof(curandState_t))); //Random States array
        CHECK(cudaMalloc(&sumTTF_GPU_final, sizeof(float)));   //Allocate Result memory

        allocate_simulation_state_on_device(&sim_state,rows,cols,num_of_tests);

        //----Declare Grid Dimensions:---------
        dim3 blocksPerGrid(num_of_blocks,1,1);
        dim3 threadsPerBlock(block_dim,1,1);

        //----KERNELS CALL -------------------
        //Inizialize Random states
        init<<<blocksPerGrid,threadsPerBlock>>>(time(NULL),states);
        cudaDeviceSynchronize();
        CHECK_KERNELCALL();
        //Execute Montecarlo simulation on GPU//,
        montecarlo_simulation_cuda<<<blocksPerGrid,threadsPerBlock,block_dim*sizeof(float)>>>(sim_state,states,num_of_tests,block_dim,max_cores,min_cores,rows,cols,wl,sumTTF_GPU,sumTTFx2_GPU);
        CHECK_KERNELCALL();
        cudaDeviceSynchronize();
         //WARNING!!! DOSNT COUNT PADDING IN ACCUMULATE
        collect_res_gpu<<<1,num_of_blocks/2>>>(sumTTF_GPU,sumTTF_GPU_final,num_of_blocks);
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
        cudaDeviceReset();
 }

#endif