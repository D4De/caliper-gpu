#ifndef SIMULATION_CONFIG
#define SIMULATION_CONFIG
//#define CUDA //To enambe cuda_helper functions
#ifdef CUDA
    #include "../cuda-utils/cuda_helper.h"
#endif

#include "default_values.h"
/**
 * Allow to store the current state of a single core
*/
struct core_state{
    float curr_r;   //Current Probability to die
    float temp;     //Temperature of the core
    float load;     //Work load of this core
    float voltage;  //We can add more Death types like Chinese paper
    int real_index; //Real position in the grid
    bool alive;
};

/**
 * Contain all usefull information about the simulation State
 * All the Working variables used by threads
 * All the results (sumTTF...)
 * All the parameters that evolve during time (left_cores, current_workload)
*/
struct simulation_state{
    float * currR;  
    float * temps;   //TODO merge temps and loads array 
    float * loads;
    int   * indexes;
    int   * real_pos;
    bool  * alives;  //TODO remove alives array

    core_state * core_states;
    float* times;   //times to death

    float   current_workload;
    int     left_cores;
    #ifdef CUDA
        curandState_t * rand_states;
    #endif

    float * sumTTF;
    float * sumTTFx2;
};

/**
 * Contain all the static configuration values initialized by user using args or initialized by default
*/
struct configuration_description{
    int     rows;           //Rows of cores in the grid
    int     cols;           //Cols of cores in the grid
    int     min_cores;      //Min num of cores alive
    int     max_cores;
    int     num_of_tests;   //Num of iteration in montecarlo
    //GPU
    int     gpu_version;    //GPU algorithm version selector (0-> redux, 1-> dynamic,2...)  
    int     block_dim;      //Blocks dimension chosen
    bool    isGPU;

    //Confidence intervall
    bool    useNumOfTest;   //True use confidence, False use numOfTest
    float   threshold;      //Threshold
    float   confInt;        //Intervallo confidenza
    
    
    float   initial_work_load;  //Initial cores workload
};

#ifdef CUDA
/**
 * Allocate Global Memory of gpu to store the simulation state
*/

void allocate_simulation_state_on_device(simulation_state* state,configuration_description config){
    int cells = config.rows*config.cols*config.num_of_tests;

    //STRUCT VERSION
    if(config.gpu_version == VERSION_STRUCT_SHARED){
        CHECK(cudaMalloc(&state->core_states    , cells*sizeof(core_state)));
        CHECK(cudaMalloc(&state->real_pos  , cells*sizeof(int)));    //real positions
        //return;
    }

    //ALL OTHER VERSIONS
    if(config.gpu_version != VERSION_STRUCT_SHARED){
        //ALL OTHER VERSION
        CHECK(cudaMalloc(&state->currR    , cells*sizeof(float)));  //CurrR
        CHECK(cudaMalloc(&state->temps    , cells*sizeof(float)));  //temps
        CHECK(cudaMalloc(&state->loads    , cells*sizeof(float)));  //loads       
        CHECK(cudaMalloc(&state->alives   , cells*sizeof(bool)));   //alives
        CHECK(cudaMalloc(&state->times    , cells*sizeof(float)));  //times
    }

    CHECK(cudaMalloc(&state->indexes  , cells*sizeof(int)));    //indexes
    
}

/**
 * Free all the Global memory allocated for the simulation state
*/
void free_simulation_state(simulation_state* state,configuration_description config){

    //STRUCT VERSION
    if(config.gpu_version == VERSION_STRUCT_SHARED){

        CHECK(cudaFree(state->core_states));
        CHECK(cudaFree(state->real_pos));
        //return;
    }

    //ALL OTHER VERSIONS
    if(config.gpu_version != VERSION_STRUCT_SHARED){
         
        CHECK(cudaFree(state->currR));
        CHECK(cudaFree(state->temps));
        CHECK(cudaFree(state->loads));
        CHECK(cudaFree(state->alives));
        CHECK(cudaFree(state->times));
        //return;
    }

   
    CHECK(cudaFree(state->indexes));
    
    
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
    config->useNumOfTest = true;
    config->isGPU  = false;
}
/**
 * Data structure is composed like
 * [All data 0][All data 1][..][All data N] N is Num of thread/iteration
 * 
 * i represent the data position we want to access
*/
#ifdef CUDA
__device__  int getIndex(int i, int N){

    int tid = threadIdx.x + blockDim.x*blockIdx.x; //Identify 
    int index = tid + N*i;
    //CUDA_DEBUG_MSG("Get Index [%d,%d]%\n",tid,index);
    return index; //Get the position of this thread inside the array for "CORE i in the grid"
}
#endif
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

#ifdef CUDA
__device__ 

void swapState(simulation_state sim_state,int dead_index,int left_cores,int max_cores){

    //Get absolute indexes from relative one
    int last_elem   = getIndex(left_cores-1,max_cores);
    int idxToSwap   = getIndex(dead_index,max_cores);

    //Save the Alive core at the end of the list into a tmp val  (COALESCED)
    int tmp_currR = sim_state.currR[last_elem];
    int tmp_loads = sim_state.loads[last_elem];
    int tmp_temps = sim_state.temps[last_elem];

    //Swap dead index to end      (COALESCED WRITE, NON COALESCED READ)
    sim_state.currR[last_elem] = sim_state.currR[idxToSwap];
    sim_state.loads[last_elem] = sim_state.loads[idxToSwap];
    sim_state.temps[last_elem] = sim_state.temps[idxToSwap];
    sim_state.indexes[last_elem] = dead_index; //Save his original position of alive core swapped

    //Put Alive core at dead position (NOT COALESCED)
    sim_state.currR[dead_index] = tmp_currR;
    sim_state.loads[dead_index] = tmp_loads;
    sim_state.temps[dead_index] = tmp_temps;
    sim_state.indexes[dead_index] = left_cores-1; //Save original position of dead core swapped
}

__device__
void swapStateStruct(simulation_state sim_state,int dead_index,int left_cores,int num_of_test){
    int* index = sim_state.indexes;
    int* value = sim_state.real_pos;
    core_state* cores = sim_state.core_states;

    //Get some indexes
    int last_elem      = getIndex(left_cores-1,num_of_test); // Last elem alive
    int death_i        = getIndex(dead_index,num_of_test);   // current core to die

    //CUDA_DEBUG_MSG("Swap [%d,%d]%\n",last_elem,death_i);
    
    int temp = value[last_elem];
    value[last_elem] = value[death_i];
    value[death_i] = temp;

    core_state t_core = cores[last_elem];
    cores[last_elem] = cores[death_i];
    cores[death_i] = t_core;
    
    int lc = getIndex(value[last_elem],num_of_test);//value[last_elem];
    int dc = getIndex(value[death_i],num_of_test);//value[death_i];

    temp = index[lc];
    index[lc] = index[dc];
    index[dc] = temp;

        
    //CUDA_DEBUG_MSG("Swap [%d.%d]\n",dead_index,left_cores);
    //CUDA_PRINT_ARRAY(value,16,getIndex);
    //CUDA_PRINT_ARRAY(index,16,getIndex);
    
}

__device__ 
void swapStateStruct1(simulation_state sim_state,int dead_index,int left_cores,int max_cores){
    int* indexes = sim_state.indexes;

    //Get some indexes
    int last_elem        = getIndex(left_cores-1,max_cores); // Last elem alive
    int old_last_elem    = getIndex(left_cores,max_cores);   // elem died last cycle
    int idxToSwap        = getIndex(dead_index,max_cores);   // current core to die

    //Update the indexes
    indexes[last_elem] = dead_index; //coalesced

    //Uncoalesced + divergent  access (depend this index where already used)
    if(indexes[dead_index] == -1){
        indexes[idxToSwap] = left_cores-1;      //First time this index host a dead core  (uncoalesced)
    }else{
        indexes[last_elem+1] = left_cores-1;    //Second time this index host a dead core (coalesced)
    }
}
#endif


#endif //SIMULATION_CONFIG