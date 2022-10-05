#ifndef CALIPER_GPU
#define CALIPER_GPU
    #include "cuda_helper.h"
    #include "./thermal_model.h"
    #include "./benchmark_helper.h"
    #include "./utils.h"
#endif

#define debugMSG(msg) if(printf(msg)

#if defined(CUDA)
__global__ void init(unsigned int seed, curandState_t *states){
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &states[tid]);
    //curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__ void montecarlo_simulation_cuda(curandState_t *states, int num_of_tests,int max_cores,int min_cores,int rows,int cols, float wl,float * sumTTF_res){
 
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

        for (j = 0; j < max_cores; j++) {
            currR[j]  = 1;
            alives[j] = true;
        }

        while (left_cores >= min_cores) {
             minIndex = -1;

        
        //-----------Redistribute Loads among alive cores----------
            double distributedLoad = (double)wl * (double)max_cores / (double)left_cores;

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
           tempModel(loads, temps, rows, cols); //CHECK HOW PASS CUDA ARRAY TO FUNCTION BY REFERENCE
            
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
            left_cores--;
        }
        __syncthreads();
        atomicAdd(sumTTF_res,(float)totalTime); //sumTTF += totaltime
        //printf("ss=%f \n", totalTime);
        }
}

#endif