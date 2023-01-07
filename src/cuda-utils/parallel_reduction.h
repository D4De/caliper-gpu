#ifndef PARALLEL_REDUX
#define PARALLEL_REDUX

#define CUDA //To enambe cuda_helper functions
#include "cuda_helper.h"

//-----------------------------------------------------------------------
//-------------WARP REDUCTION ALL VERSIONS-------------------------------
//-----------------------------------------------------------------------

//"volatile" tag explanation https://stackoverflow.com/questions/15331009/when-to-use-volatile-with-shared-cuda-memory
template<class T>
__device__ void warpReduce(volatile T *input,size_t threadId)
{
	input[threadId] += input[threadId + 32];
	input[threadId] += input[threadId + 16];
	input[threadId] += input[threadId + 8];
	input[threadId] += input[threadId + 4];
	input[threadId] += input[threadId + 2];
	input[threadId] += input[threadId + 1];
}

template<class T>
__device__ void warpReduce_min(volatile T *input,size_t threadId)
{
    input[threadId] = min(input[threadId],input[threadId + 32]);
    input[threadId] = min(input[threadId],input[threadId + 16]);
    input[threadId] = min(input[threadId],input[threadId + 8]);
    input[threadId] = min(input[threadId],input[threadId + 4]);
    input[threadId] = min(input[threadId],input[threadId + 2]);
    input[threadId] = min(input[threadId],input[threadId + 1]);
}

//-----------------------------------------------------------------------
//-------------ACCUMULATE FUNCTION VERSIONS------------------------------
//-----------------------------------------------------------------------
__device__ float accumulate2D(float *input, size_t dim)
{
    size_t threadId = threadIdx.x;
    bool odd = dim%2;
    for (int i = dim / 2; i > 0; i >>= 1)
    {

        if ((threadId  < i))
        {
            input[threadId] += input[threadId + i];
        }

        if(threadId == 0 && odd)
            input[threadId] += input[threadId + 2*i];
        odd = i%2;
        __syncthreads();
    }

    return input[0];
}

template<class T>
__device__ float accumulate(T *input, size_t dim,int num_of_elem)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	size_t threadId = threadIdx.x;
	//if (dim > 32)
	//{
		for (size_t i = dim / 2; i > 0; i >>= 1)
		{

            if ((threadId  < i))//&& (tid + i < num_of_elem) -> those elements are initialized to 0 so not relevant
            {
                input[threadId] += input[threadId + i];
            }
            __syncthreads();
        }
	//}
    /*
    WHY WARP REDUCE DOES NOT CHECK IF WE HAVE LESS THEN threadId + 32 elements?
	if (threadId < 32)
		warpReduce<T>(input, threadId);
	__syncthreads();
    */

	return input[0];
}



template<class T>
__device__ float accumulate_min(T *input, size_t dim,int num_of_elem)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	size_t threadId = threadIdx.x;
	if (dim > 32)
	{
		for (size_t i = dim / 2; i > 32; i >>= 1)
		{

            if ((threadId  < i )&& (tid + i < num_of_elem))// -> those elements are initialized to 0 so not relevant
            {
                input[threadId] = min(input[threadId],input[threadId + i]);
            }
            __syncthreads();
        }
	}
	if (threadId < 32)
		warpReduce_min<T>(input, threadId);
	__syncthreads();

    if (threadId == 0){
        //printf("Partial Result = %f\n",input[0]);
    }
	return input[0];
}




template<class T>
__device__ void warpReduce_argMin(volatile T *input,int* minis,size_t threadId)
{
    minis[threadId] = argmin(input,threadId,threadId + 32);
    minis[threadId] = argmin(input,threadId,threadId + 16);
    minis[threadId] = argmin(input,threadId,threadId + 8);
    minis[threadId] = argmin(input,threadId,threadId + 4);
    minis[threadId] = argmin(input,threadId,threadId + 2);
    minis[threadId] = argmin(input,threadId,threadId + 1);
}

template<class T>
__device__ T argmin(T* array,int index_a,int index_b,int walk_id,int* minis){
    int index = array[index_a]<array[index_b] ? index_a : index_b;
    //CUDA_DEBUG_MSG("COMPARE [%d ::> %f] with [%d ::> %f] selected -> [%d ::> %f]\n",index_a,array[index_a],index_b,array[index_b],index,array[index]);
    array[index_a] = array[index];//MOVE MIN INTO index_a position

    return minis[index];

}

template<class T>
__device__ T accumulate_argMin(simulation_state sim_state, configuration_description config, int walk_id, int core_id, int* minis,int dim)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	size_t threadId = threadIdx.x;

    int global_id = core_id + walk_id * config.max_cores;
    bool odd = dim%2;
	//if (config.max_cores > 32)
	//{
		for (size_t i = dim / 2; i > 0; i >>= 1)
		{  
            if ((core_id  < i ))// -> those elements are initialized to 0 so not relevant
            {   
                int index_a = global_id;        //A
                int index_b = global_id + i;    //B

                int index = sim_state.times[index_a]<sim_state.times[index_b] ? index_a : index_b; // A < B ?? Which is minimum?

                //CUDA_DEBUG_MSG("COMPARE [%d ::> %f] with [%d ::> %f] selected -> [%d ::> %f]\n",index_a,sim_state.times[index_a],index_b,sim_state.times[index_b],index,sim_state.times[index]);
                sim_state.times[index_a] = sim_state.times[index];//Move minimum to coreID
                
                index   = index     - walk_id*config.max_cores;         //Convert Global index to local block index
                index_a = index_a   - walk_id*config.max_cores;         //Convert Global index to local block index

                minis[index_a] =  minis[index];                   //Change the min Index of this local block minis
            }
            if(odd && core_id == 0){
                int index_a = global_id;        //A
                int index_b = global_id + i*2;    //B

                int index = sim_state.times[index_a]<sim_state.times[index_b] ? index_a : index_b; // A < B ?? Which is minimum?

                //CUDA_DEBUG_MSG("COMPARE [%d ::> %f] with [%d ::> %f] selected -> [%d ::> %f]\n",index_a,sim_state.times[index_a],index_b,sim_state.times[index_b],index,sim_state.times[index]);
                sim_state.times[index_a] = sim_state.times[index];//Move minimum to coreID
                
                index   = index     - walk_id*config.max_cores;         //Convert Global index to local block index
                index_a = index_a   - walk_id*config.max_cores;         //Convert Global index to local block index

                minis[index_a] =  minis[index];   //Change the min Index of this local block minis

            }
            odd = i%2;
            __syncthreads();
        }
	//}
    /*
	if (threadId < 32)
		warpReduce_argMin<T>(sim_state.times,minis, threadId);
	__syncthreads();
    */
   
	return minis[0];
}

//-----------------------------------------------------------------------
//-------------GLOBAL COLLECTOR ALL VERSIONS-----------------------------
//-----------------------------------------------------------------------
template<class T>
__global__ void collect_res_gpu(T *input,float* result, int numOfBlocks) // compute the final reduction
{

    //*result = input[0];
    //return;
    unsigned int tid = threadIdx.x;// + blockDim.x*blockIdx.x;
    unsigned i;
    __shared__ T localVars[1024];   //Max num possible num of blocks
    //Each thread Save in shared mem-> element in first and second half

    if(tid < numOfBlocks/ 2){
        localVars[tid] = input[tid] + input[tid+ numOfBlocks/2];
        __syncthreads();
    }

    if(tid <= numOfBlocks/2){
        //IF Remaining cells are more then 32
        for (i = numOfBlocks / 2; i > 32; i >>= 1) // compute the parallel reduction for the collected data
        {
            if (tid < i)//Each cycle use half of the threads
            {
                localVars[tid] += localVars[tid + i]; //first half elem + second half elem
            }
            __syncthreads();
        }

        if(tid<32)
            warpReduce<T>(localVars,tid);
        __syncthreads();

        if(tid==0){
            *result = localVars[tid]; //Thread 0 has result
        }
            
        __syncthreads();
    }    
}

#endif //PARALLEL_REDUX