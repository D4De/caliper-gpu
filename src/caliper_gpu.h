#ifndef CALIPER_GPU
#define CALIPER_GPU
    #include "cuda_helper.h"
    #include "./thermal_model.h"
    #include "./benchmark_helper.h"
    #include "./utils.h"
#endif

//#define debugMSG(msg) if(printf(msg)

#if defined(CUDA) //To avoid errors if compiling with GCC


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
__global__ void tempModel_gpu(double* loads, double* temps,bool* alives,float distributed_wl,int cols, int rows){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    //TODO calculate loads directly here using alive flag (avoid an O(N) for cycle)

    //Shared memory for loads 0r calculate them real time
    if(i<cols && j<rows){ // CHECK TO BE INSIDE THE GRID

        //Calculate Load for a specific core
        if (alives[i*cols+j] == true) {
            loads[i*cols+j] = distributed_wl;
        } else {
            loads[i*cols+j] = 0;
        }

        //Calculate Temp Model for a specific Core
        int temp = 0;
        int k,h;
        for (k = -1; k < 2; k++)
                    for (h = -1; h < 2; h++)
                        if ((k != 0 || h != 0) && k != h && k != -h && i + k >= 0 && i + k < rows && j + h >= 0 && j + h < cols){
                            temp += loads[(i + k)*cols + (j + h)] * NEIGH_TEMP;
                        }
                temps[i*cols+j] = ENV_TEMP + loads[i*cols+j] * SELF_TEMP + temp;
    }


}

__global__ void montecarlo_simulation_cuda(curandState_t *states, int num_of_tests,int max_cores,int min_cores,int rows,int cols, float wl,float * sumTTF_res,float * sumTTFx2_res){
 
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        
    if(tid<num_of_tests){
        double random;
        int left_cores;
        double currR[ROWS*COLS];
        double stepT;
        int minIndex;
        double totalTime;
        bool alives[ROWS*COLS];
        int j;
        double t, eqT;
        double loads[ROWS*COLS];
        double temps[ROWS*COLS];

        left_cores = max_cores;
        totalTime = 0;
        minIndex = 0;

        for (j = 0; j < max_cores; j++) { //Also parallelizable (dont know if usefull, only for big N)
            currR[j]  = 1;
            alives[j] = true;
        }

        while (left_cores >= min_cores) {
             minIndex = -1;

        
        //-----------Redistribute Loads among alive cores----------
            double distributedLoad = (double)wl * (double)max_cores / (double)left_cores;

            //Potentialy use dynamic programming (merge tempModel and loads into a 2D kernel)
            for (int i = 0, k = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (alives[i*cols+j] == true) {
                        loads[i*cols+j] = distributedLoad;
                    } else {
                        loads[i*cols+j] = 0;
                    }
                    k++;
                }
            }

            //CHECK HOW DYNAMICALY ALLOCATE ARRAY INSIDE KERNEL (into registers and not global memory)
           tempModel(loads, temps, rows, cols);

            //-----------Random Walk Step computation-------------------------
            for (j = 0; j < max_cores; j++) {
                if (alives[j] == true) {
                    //Random  is in range [0 : currR[j]]
                    random =(double)curand_uniform(&states[tid])* currR[j]; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                    double alpha = getAlpha(temps[j]);
                    double alpha_rounded = round1(alpha);
                    t = alpha_rounded * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                    eqT = alpha_rounded * pow(-log(currR[j]), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
                    t = t - eqT;

                    //EQT > t some times... WHY??? (Overflow? Error? Random function?)
                    //Strange CurrR behaviour (on GPU remain near 1 on CPU it decrease correctly toward 0)
                    //ON GPU The random number seem generally higher than cpu rand ...

                    //THE PROBLEMS IS IN tempModel()... (PUT 863361.000 to all alphas!!!!!!!! -> is due to pointers incompatible with cuda?)
                    if(tid==1){
                        //printf("%d -> Death Time: %f ->(%f)(%f) -- [%f][%f][%f]\n",j,t,random,currR[j],alpha_rounded,temps[j],loads[j]);
                    }

                    //the difference between the two values represents the time elapsed from the previous failure to the current failure
                    //(we will sum to the total time the minimum of such values)
                    
                    if (minIndex == -1 || (minIndex != -1 && t < stepT)) {
                        minIndex = j;//Set new minimum index
                        stepT = t;   //set new minimum time as timeStep
                    } //TODO ADD A CHECK ON MULTIPLE FAILURE IN THE SAME INSTANT OF TIME.
                }
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
            totalTime = totalTime + stepT; //Current simulation time
        //---------UPDATE Configuration----------------- 
            if (left_cores > min_cores) {
                alives[minIndex] = false;
                // compute remaining reliability for working cores
                for (j = 0; j < max_cores; j++) {
                    if (alives[j]) {
                    		double alpha = getAlpha(temps[j]);
                    		double alpha_rounded = round1(alpha);
                            eqT = alpha_rounded * pow(-log(currR[j]), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
                            currR[j] = exp(-pow((stepT + eqT) / alpha_rounded, BETA));
                    }
                }
            }
            left_cores--; //Reduce number of alive core in this simulation
        }//END SIMULATION-----------------------------
        __syncthreads();
        atomicAdd(sumTTF_res,(float)totalTime); //sumTTF += totaltime
        atomicAdd(sumTTFx2_res,(float)totalTime *  (float)totalTime); //sumTTFx2 += totaltime * totaltime
        
        //TODO 
        //SOL1:--------------------------------------------
        //USE SHARED MEMORY TO STORE  temporary SumTTF_Res to each block
        //  and then Apply Global Parallel Reduction
        //SOL2:---------------------------------------------
        //Or Shared Mem to store An array of temp SumTTF 
        // Apply local parallel reduction
        //Then global parallel reduction on the accumulated results

    }
}

void montecarlo_simulation_cuda_host(curandState_t *states, int num_of_tests,int max_cores,int min_cores,int rows,int cols, float wl,float * sumTTF_res,float * sumTTFx2_res){
 {
    
 }

#endif