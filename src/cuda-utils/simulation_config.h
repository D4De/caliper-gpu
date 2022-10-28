#ifndef SIMULATION_CONFIG
#define SIMULATION_CONFIG
//#define CUDA //To enambe cuda_helper functions
#ifdef CUDA
    #include "cuda_helper.h"
#endif
struct simulation_state{
    float * currR;  
    float * temps;   //TODO merge temps and loads array 
    float * loads;
    int   * indexes;
    bool  * alives;  //TODO remove alives array

    float   current_workload;
    int     left_cores;
    #ifdef CUDA
        curandState_t * rand_states;
    #endif

    float * sumTTF;
    float * sumTTFx2;
};

struct configuration_description{
    int     rows;           //Rows of cores in the grid
    int     cols;           //Cols of cores in the grid
    int     min_cores;        //Min num of cores alive
    int     max_cores;
    int     num_of_tests;    //Num of iteration in montecarlo
    int     gpu_version;    //GPU algorithm version selector (0-> redux, 1-> dynamic,2...)  
    int     block_dim;      //Blocks dimension chosen

    float   initial_work_load;  //Initial cores workload
};

#ifdef CUDA
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

/**
 * Free all the Global memory allocated for the simulation state
*/
void free_simulation_state(simulation_state* state){
    CHECK(cudaFree(state->currR));
    CHECK(cudaFree(state->temps));
    CHECK(cudaFree(state->loads));
    CHECK(cudaFree(state->indexes));
    CHECK(cudaFree(state->alives));
}

#endif //CUDA

void setup_config(configuration_description* config,int num_of_tests,int max_cores,int min_cores,int wl,int rows,int cols,int block_dim,int gpu_version){
    
    config->num_of_tests = num_of_tests;

    config->cols        = cols;
    config->rows        = rows;
    config->min_cores     = min_cores;

    config->block_dim   = block_dim;
    config->gpu_version = gpu_version;
    config->initial_work_load = wl;
    config->max_cores = max_cores;
}

/**
 * Swap the dead core to the end of the array
 * 
 * This allow to have alives core indexes saved on the first portion of the array
 * [Array of index] = [Alive cores][Dead Cores]
*/
#ifdef CUDA
__device__ __host__ 
#endif
void swap_core_index(int* cores,int dead_index,int size,int offset){
    int tmp = cores[offset+size-1];

    cores[offset+size-1] = cores[dead_index]; //Swap dead index to end
    cores[dead_index] = tmp;         

}


#endif //SIMULATION_CONFIG