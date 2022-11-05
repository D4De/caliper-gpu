#ifndef THERMAL_MODEL
#define THERMAL_MODEL
#include <iostream>
#include "simulation_config.h"

// Electro-Migration related parameters
#define BETA 2
#define ACTIVATIONENERGY 0.48
#define BOLTZMANCONSTANT 8.6173324*0.00001
#define CONST_JMJCRIT 1500000
#define CONST_N 1.1
#define CONST_ERRF 0.88623
#define CONST_A0 30000 //cross section = 1um^2  material constant = 3*10^13

// Thermal model parameters
#define ENV_TEMP 295 //room temperature
#define SELF_TEMP 40 //self contribution
#define NEIGH_TEMP 5 //neighbor contribution

#define getAlpha(temp) ((CONST_A0 * (pow(CONST_JMJCRIT,(-CONST_N))) * exp(ACTIVATIONENERGY / (BOLTZMANCONSTANT * temp))) / CONST_ERRF)

#define NTESTS 100000 //With 100k more the result converge
#define BETA 2
#define MIN_NUM_OF_TRIALS 100

#define RANDOMSEED_STR "RANDOM"

// Support function to allow arbitrary confidence intervals
#define INV_ERF_ACCURACY 10e-6
//__device__ __host__ 

#ifdef CUDA
__device__ __host__ 
#endif
void tempModel(double *loads, double* temps, int rows, int cols) {
    double temp;
    int i, j, k, h;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++) {
            for (k = -1, temp = 0; k < 2; k++)
                for (h = -1; h < 2; h++)
                    if ((k != 0 || h != 0) && k != h && k != -h && i + k >= 0 && i + k < rows && j + h >= 0 && j + h < cols){
                        temp += loads[(i + k)*cols + (j + h)] * NEIGH_TEMP;
                    }
            temps[i*cols+j] = ENV_TEMP + loads[i*cols+j] * SELF_TEMP + temp;
        }
}

#ifdef CUDA
__device__ __host__ 
#endif
void tempModel(float *loads, float* temps, int rows, int cols,int offset) {

    float temp;
    int i, j, k, h;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++) {
            for (k = -1, temp = 0; k < 2; k++)
                for (h = -1; h < 2; h++)
                    if ((k != 0 || h != 0) && k != h && k != -h && i + k >= 0 && i + k < rows && j + h >= 0 && j + h < cols){
                        temp += loads[offset + (i + k)*cols + (j + h)] * NEIGH_TEMP;
                    }
            temps[offset + i*cols+j] = ENV_TEMP + loads[offset + i*cols+j] * SELF_TEMP + temp;
        }
}

//TODO MERGE ALL THOSE FUNCTIONS TOGETHER USING TEMPLATES!!!!! (to avoid code repetition)
#ifdef CUDA
__device__ __host__ 
#endif
void tempModel(float *loads, float* temps,int* indexes,int left_alive, int rows, int cols,int offset) {

    for(int i=0;i<left_alive;i++){
        int absolute_index = indexes[i + offset];     //contain position of this core in the original grid
        int relative_index = absolute_index - offset; //local position (usefull only on gpu global memory,for cpu is same as absolute)

        int r = relative_index/cols;    //Row position into local grid
        int c = relative_index%cols;    //Col position into local grid

        float temp = 0;
        int k,h;

        for (k = -1; k < 2; k++)
        {
            for (h = -1; h < 2; h++)
            {
                if ((k != 0 || h != 0) && k != h && k != -h && r + k >= 0 && r + k < rows && c + h >= 0 && c + h < cols){
                    temp += loads[offset+(r + k)*cols + (c + h)] * NEIGH_TEMP;
                    //int ld = (r + k)*cols + (c + h) < left_alive ? distributed_load : 0;
                }
            }
        }

        temps[absolute_index] = ENV_TEMP + loads[absolute_index] * SELF_TEMP + temp;
    }
}

#ifdef CUDA
__device__  

void tempModel_version2(float *loads, float* temps,int* indexes,float distributed_load,int left_alive, int rows, int cols,int num_of_tests) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int max_cores = rows*cols;
    for(int i=0;i<left_alive;i++){
        int absolute_index = getIndex(i,num_of_tests);     //contain position of this core in the original grid
        int relative_index = indexes[absolute_index];   //local position (usefull only on gpu global memory,for cpu is same as absolute)
        
        if(tid == 0){
            printf("(%d-%d-%f)",relative_index,absolute_index,temps[absolute_index]);
        }
        int r = relative_index/cols;    //Row position into local grid
        int c = relative_index%cols;    //Col position into local grid

        float temp = 0;
        int k,h;

        for (k = -1; k < 2; k++)
        {
            for (h = -1; h < 2; h++)
            {
                if ((k != 0 || h != 0) && k != h && k != -h && r + k >= 0 && r + k < rows && c + h >= 0 && c + h < cols){
                    int idx = getIndex((r + k)*cols + (c + h),num_of_tests);
                    //int ld = (indexes[idx] < left_alive) ? distributed_load : 0;
                    temp += loads[idx] * NEIGH_TEMP;
                }
            }
        }

        temps[absolute_index] = ENV_TEMP + loads[absolute_index] * SELF_TEMP + temp;
    }
    if(tid == 0){
            printf("\n");
    }
}
#endif

#endif //THERMAL_MODEL