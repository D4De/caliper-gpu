#include "cuda_helper.h"
#include "./thermal_model.h"
#include "./benchmark_helper.h"
#include "./utils.h"

#if defined(CUDA)
__global__ void init(unsigned int seed, curandState_t *states){
    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__ void montecarlo_simulation_cuda(curandState_t *states, int num_of_tests,int max_cores,int min_cores,int rows,int cols, int wl,float * sumTTF_res){
 
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        
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

        for (j = 0; j < max_cores; j++) {
            currR[j]  = 1;
            alives[j] = true;
        }

        while (left_cores >= min_cores) {
             minIndex = -1;

        //-----------Redistribute Loads among alive cores----------
            double distributedLoad = wl * max_cores / left_cores;

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

            tempModel(loads, temps, rows, cols);
            
            //-----------Random Walk Step computation-------------------------
            for (j = 0; j < max_cores; j++) {
                if (alives[j] == true) {
                    random = curand_uniform(&states[tid]); //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                    double alpha = getAlpha(temps[j]);
                    double alpha_rounded = round1(alpha);
                    t = alpha_rounded * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                    eqT = alpha_rounded * pow(-log(currR[j]), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
                    t = t - eqT;
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
            left_cores--;
        }
        __syncthreads();
        atomicAdd(sumTTF_res,(float)totalTime);
}

#endif