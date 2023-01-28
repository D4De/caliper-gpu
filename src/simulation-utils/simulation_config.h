#ifndef SIMULATION_CONFIG
#define SIMULATION_CONFIG
//#define CUDA //To enambe cuda_helper functions
#ifdef CUDA
#include "../cuda-utils/cuda_helper.h"
#endif

#include "default_values.h"

#define ALIVE true
#define DEAD  false


//----------------------------------------------------------------------------------
//---------------SIMULATION DATA STRUCTURES-----------------------------------------
//----------------------------------------------------------------------------------

class Core_neighbourhood;


/**
 * Allow to store the current state of a single core
*/
struct core_state{
    float curr_r;   //Current Probability to die
    float temp;     //Temperature of the core
    float load;     //Work load of this core
    //float voltage;  //We can add more Death types like Chinese paper
    int real_index; //Real position in the grid
    bool alive;

    bool* top_core;
    bool* bot_core;
    bool* left_core;
    bool* right_core;
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

    Core_neighbourhood *neighbour_state;
    
    bool  * false_register;
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

#define __Host_Device__  __host__ __device__

class Core_neighbourhood{
    public:

    int my_position;
    int offset;
    bool me;
    bool top_core;
    bool bot_core;
    bool left_core;
    bool right_core;

    __Host_Device__ Core_neighbourhood(){
        me = ALIVE;
    }

    __Host_Device__ void initialize(int my_pos,int off,configuration_description config){
        my_position = my_pos;
        offset      = off;

        int r = my_position / config.cols;
        int c = my_position % config.cols;

        bool out_of_range;

        //Top
        out_of_range = ((r - 1) < 0);
        bot_core = out_of_range ? DEAD : ALIVE; 
        //Bot
        out_of_range = ((r + 1) > config.rows);
        top_core = out_of_range ? DEAD : ALIVE; 
        //Left
        out_of_range = ((c - 1) < 0);
        left_core = out_of_range ? DEAD : ALIVE; 
        //Right
        out_of_range = ((c + 1) > config.cols);
        right_core = out_of_range ? DEAD : ALIVE; 
    }

    __Host_Device__ void set_top_dead(){
        top_core = DEAD;
    }

    __Host_Device__ void set_bot_dead(){
        bot_core = DEAD;
    }

    __Host_Device__ void set_left_dead(){
        left_core = DEAD;
    }

    __Host_Device__ void set_right_dead(){
        right_core = DEAD;
    }

    //COMUNICATE TO THIS CORE NEIGHBOURS THAT IT IS DEAD
    __Host_Device__ void update_state_after_core_die(Core_neighbourhood* neighbours,configuration_description config,simulation_state sim_state){
        
        int r = my_position / config.cols;
        int c = my_position % config.cols;

        //CUDA_DEBUG_MSG("CORE IN [%d][%d] is dead\n",r,c);
        //Border check is not necessary since if "top/bot/left/right" is false then or core is dead or does not exist
        if(top_core){
            neighbours[((r+1)*config.cols) + (c)].set_bot_dead();   //I say to my top neighbour to set my dead
        }
        if(bot_core){
            neighbours[((r-1)*config.cols) + (c)].set_top_dead();   //I say to my bot neighbour to set my dead
        }
        if(left_core){
            neighbours[((r)*config.cols) + (c-1)].set_right_dead(); //I say to my left neighbour to set my dead
        }
        if(right_core){
            neighbours[((r)*config.cols) + (c+1)].set_left_dead();  //I say to my right neighbour to set my dead
        }
    }

    __device__ float calculate_temperature(configuration_description config,simulation_state sim_state){

    }
};

struct n{
    bool top_core;
    bool bot_core;
    bool right_core;
    bool left_core;
};

class Core_neighbourhood_v2;
class Core_neighbourhood_v3;
__global__  void initialize_neighbourhood(bool * t, bool * b, bool * l, bool * r, Core_neighbourhood_v2 * neighbours);
__global__  void initialize_neighbourhood_v3(n* ne, Core_neighbourhood_v3 * neighbours);




//template <bool mode> //0 no struct 1 struct
class Core_neighbourhood_v2{

    public:
    bool * top_cores;
    bool * bot_cores;
    bool * right_cores;
    bool * left_cores;


    Core_neighbourhood_v2(){}



    //Build a Core_neighbourhood on CPU but copy it on GPU
    static Core_neighbourhood_v2* factory_neighbours(int cells){
        Core_neighbourhood_v2* cpu_copy = new Core_neighbourhood_v2(); 

        Core_neighbourhood_v2* cuda_version; 

        CHECK(cudaMalloc(&cuda_version    , sizeof(Core_neighbourhood_v2)));

            CHECK(cudaMalloc(&cpu_copy->top_cores    , cells*sizeof(float)));  //Top Cores
            CHECK(cudaMalloc(&cpu_copy->bot_cores    , cells*sizeof(float)));  //Bot Cores
            CHECK(cudaMalloc(&cpu_copy->right_cores   , cells*sizeof(float)));  //Right Cores
            CHECK(cudaMalloc(&cpu_copy->left_cores    , cells*sizeof(float)));  //Left Cores

        
        initialize_neighbourhood<<<1,1>>>(cpu_copy->top_cores,cpu_copy->bot_cores,cpu_copy->right_cores,cpu_copy->left_cores,cuda_version);
        CHECK_KERNELCALL(); 
        cudaDeviceSynchronize();
        //CHECK(cudaMemcpy(cuda_version, cpu_copy, sizeof(Core_neighbourhood_v2), cudaMemcpyHostToDevice));  

        return cuda_version;
    }

    __device__ void initialize_core(configuration_description config,int index,int offset){
        int r = index / config.cols;
        int c = index % config.cols;

        int idx = index + offset;
        bool out_of_range = false;

        //bot_cores[index] = DEAD;
        //Top
        out_of_range = ((r - 1) < 0);
        top_cores[idx] = out_of_range ? DEAD : ALIVE; 
        //Bot
        out_of_range = ((r + 1) >= config.rows);
        bot_cores[idx] = out_of_range ? DEAD : ALIVE; 
        //Left
        out_of_range = ((c - 1) < 0);
        left_cores[idx] = out_of_range ? DEAD : ALIVE; 
        //Right
        out_of_range = ((c + 1) >= config.cols);
        right_cores[idx] = out_of_range ? DEAD : ALIVE; 
    }

    __device__ void notify_core_death(configuration_description config,int index,int offset){
        int r = index / config.cols;
        int c = index % config.cols;

        int idx = index + offset;
        //(me)in example is the core dead
        if(top_cores[idx]){//If my top is alive, set his bot (me) as dead
            bot_cores[offset + ((r-1)*config.cols) + (c)] = DEAD;
        }

        if(bot_cores[idx]){
            top_cores[offset + ((r+1)*config.cols) + (c)] = DEAD;
        }


        if(left_cores[idx]){
            right_cores[offset + ((r)*config.cols) + (c-1)] = DEAD;
        }

        if(right_cores[idx]){
            left_cores[offset + ((r)*config.cols) + (c+1)] = DEAD;
        }

    }

    __device__ float computeTemp(int idx,float NEIGH_TEMP,float distributed_load){

        //TODO, use a struct instead of 4 arrays for more coalesed load accesses
        float temp = 0;
        temp +=  top_cores[idx] * distributed_load * NEIGH_TEMP; 
         //BOTTOM
        temp +=  bot_cores[idx] * distributed_load * NEIGH_TEMP; 
         //LEFT
        temp +=  left_cores[idx] * distributed_load * NEIGH_TEMP; 
         //RIGHT
        temp +=  right_cores[idx] * distributed_load * NEIGH_TEMP; 

        return temp;
    }


};



class Core_neighbourhood_v3{

    public:
    n* neighbours_state;


    Core_neighbourhood_v3(){}



    //Build a Core_neighbourhood on CPU but copy it on GPU
    static Core_neighbourhood_v3* factory_neighbours(int cells){
        Core_neighbourhood_v3* cpu_copy = new Core_neighbourhood_v3(); 

        Core_neighbourhood_v3* cuda_version; 

        CHECK(cudaMalloc(&cuda_version    , sizeof(Core_neighbourhood_v3)));

        CHECK(cudaMalloc(&cpu_copy->neighbours_state    , cells*sizeof(n)));  //Top Cores

        
        initialize_neighbourhood_v3<<<1,1>>>(cpu_copy->neighbours_state,cuda_version);
        CHECK_KERNELCALL(); 
        cudaDeviceSynchronize();
        //CHECK(cudaMemcpy(cuda_version, cpu_copy, sizeof(Core_neighbourhood_v2), cudaMemcpyHostToDevice));  

        return cuda_version;
    }

    __device__ void initialize_core(configuration_description config,int index,int offset){
        int r = index / config.cols;
        int c = index % config.cols;

        int idx = index + offset;
        bool out_of_range = false;

        //bot_cores[index] = DEAD;
        //Top

        n tmp;

        out_of_range = ((r - 1) < 0);
        tmp.top_core = out_of_range ? DEAD : ALIVE; 
        //Bot
        out_of_range = ((r + 1) >= config.rows);
        tmp.bot_core = out_of_range ? DEAD : ALIVE; 
        //Left
        out_of_range = ((c - 1) < 0);
        tmp.left_core = out_of_range ? DEAD : ALIVE; 
        //Right
        out_of_range = ((c + 1) >= config.cols);
        tmp.right_core = out_of_range ? DEAD : ALIVE; 

        neighbours_state[idx] = tmp;

    }

    __device__ void notify_core_death(configuration_description config,int index,int offset){
        int r = index / config.cols;
        int c = index % config.cols;

        int idx = index + offset;

        n tmp = neighbours_state[idx];
        //(me)in example is the core dead
        if(tmp.top_core){//If my top is alive, set his bot (me) as dead
            neighbours_state[offset + ((r-1)*config.cols) + (c)].bot_core = DEAD;
        }

        if(tmp.bot_core){
            neighbours_state[offset + ((r+1)*config.cols) + (c)].top_core = DEAD;
        }

        if(tmp.left_core){
            neighbours_state[offset + ((r)*config.cols) + (c-1)].right_core = DEAD;
        }

        if(tmp.right_core){
            neighbours_state[offset + ((r)*config.cols) + (c+1)].left_core = DEAD;
        }

    }

    __device__ float computeTemp(int idx,float NEIGH_TEMP,float distributed_load){

        //TODO, use a struct instead of 4 arrays for more coalesed load accesses

        n tmp = neighbours_state[idx];

        float temp = 0;
        temp +=  tmp.top_core * distributed_load * NEIGH_TEMP; 
         //BOTTOM
        temp +=  tmp.bot_core * distributed_load * NEIGH_TEMP; 
         //LEFT
        temp +=  tmp.left_core * distributed_load * NEIGH_TEMP; 
         //RIGHT
        temp +=  tmp.right_core * distributed_load * NEIGH_TEMP; 

        return temp;
    }


};

//template <bool mode> //0 no struct 1 struct
__global__  void initialize_neighbourhood(bool * t, bool * b, bool * l, bool * r, Core_neighbourhood_v2 * neighbours){
    printf("INITIALIZED NEIGHBOURS\n");

        neighbours->top_cores = t;
        neighbours->bot_cores = b;
        neighbours->right_cores = r;
        neighbours->left_cores = l;


}

__global__  void initialize_neighbourhood_v3(n* ne, Core_neighbourhood_v3 * neighbours){
    printf("INITIALIZED NEIGHBOURS\n");
    neighbours->neighbours_state = ne;
}

/**
 * Allocate Global Memory of gpu to store the simulation state
*/

void allocate_simulation_state_on_device(simulation_state* state,configuration_description config){
    int cells = config.rows*config.cols*config.num_of_tests;

    //STRUCT VERSION
    if(config.gpu_version == VERSION_STRUCT_SHARED || config.gpu_version == VERSION_DYNAMIC || config.gpu_version == VERSION_STRUCT_OPTIMIZED || config.gpu_version == VERSION_2D_GRID || config.gpu_version == VERSION_GRID_LINEARIZED){
        CHECK(cudaMalloc(&state->core_states    , cells*sizeof(core_state)));
    }

    //ALL OTHER VERSION
    if(config.gpu_version != VERSION_STRUCT_SHARED){
        CHECK(cudaMalloc(&state->currR    , cells*sizeof(float)));  //CurrR
        CHECK(cudaMalloc(&state->temps    , cells*sizeof(float)));  //temps
        CHECK(cudaMalloc(&state->loads    , cells*sizeof(float)));  //loads
    }
    
    
    CHECK(cudaMalloc(&state->alives   , cells*sizeof(bool)));   //alives
    CHECK(cudaMalloc(&state->times    , cells*sizeof(float)));  //times

    CHECK(cudaMalloc(&state->false_register, sizeof(bool)));
    
    CHECK(cudaMalloc(&state->indexes  , cells*sizeof(int)));    //indexes
    CHECK(cudaMalloc(&state->real_pos  , cells*sizeof(int)));   //real positions

    if(config.gpu_version == VERSION_COALESCED_OPTIMIZED){
        CHECK(cudaMalloc(&state->neighbour_state, cells*sizeof(Core_neighbourhood)));
    }
}

/**
 * Free all the Global memory allocated for the simulation state
*/
void free_simulation_state(simulation_state* state,configuration_description config){

    //STRUCT VERSION
    if(config.gpu_version == VERSION_STRUCT_SHARED || config.gpu_version == VERSION_DYNAMIC || config.gpu_version == VERSION_STRUCT_OPTIMIZED || config.gpu_version == VERSION_2D_GRID || config.gpu_version == VERSION_GRID_LINEARIZED){
        CHECK(cudaFree(state->core_states));
    }

    //ALL OTHER VERSIONS
    if(config.gpu_version != VERSION_STRUCT_SHARED){
        CHECK(cudaFree(state->currR));
        CHECK(cudaFree(state->temps));
        CHECK(cudaFree(state->loads));
    }
    CHECK(cudaFree(state->indexes));
    CHECK(cudaFree(state->alives));
    CHECK(cudaFree(state->times));
    CHECK(cudaFree(state->false_register));

    if(config.gpu_version == VERSION_COALESCED_OPTIMIZED){
        CHECK(cudaFree(state->neighbour_state));
    }
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

    return tid + N*i; //Get the position of this thread inside the array for "CORE i in the grid"
}
#define GETINDEX(i,tid,N)  (tid + N*(i))

#endif

//----------------------------------------------------------------------------------------------
//----------------------SWAP STATE FUNCTIONS----------------------------------------------------
//-Allow to swap the dead core state with the last alive to optimize simulation cycles----------
//----------------------------------------------------------------------------------------------

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
void swapState(simulation_state sim_state,int dead_index,int left_cores,int num_of_tests)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int* index = sim_state.indexes;
    int* value = sim_state.real_pos;
    core_state* cores = sim_state.core_states;

    //Get some indexes
    int last_elem   = GETINDEX((left_cores-1),tid,num_of_tests);
    int death_i     = GETINDEX(dead_index,tid,num_of_tests);
    

    int temp = value[last_elem];
    value[last_elem] = value[death_i];
    value[death_i] = temp;

    //Save the Alive core at the end of the list into a tmp val  (COALESCED)
    float tmp_currR = sim_state.currR[last_elem];
    float tmp_loads = sim_state.loads[last_elem];
    float tmp_temps = sim_state.temps[last_elem];

    //Swap dead index to end      (COALESCED WRITE, NON COALESCED READ)
    sim_state.currR[last_elem] = sim_state.currR[death_i];
    sim_state.loads[last_elem] = sim_state.loads[death_i];
    sim_state.temps[last_elem] = sim_state.temps[death_i];
    

    //Put Alive core at dead position (NOT COALESCED)
    sim_state.currR[dead_index] = tmp_currR;
    sim_state.loads[dead_index] = tmp_loads;
    sim_state.temps[dead_index] = tmp_temps;

    int temp2 = index[value[last_elem]];
    index[value[last_elem]] = index[value[death_i]];
    index[value[death_i]] = temp2;
    //CUDA_DEBUG_MSG("Swappato core: %d con core %d \n",left_cores-1, dead_index)
}



__device__ 
void swapStateOptimized (simulation_state sim_state,int dead_index,int left_cores,int num_of_tests)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int* index = sim_state.indexes;
    int* value = sim_state.real_pos;
    core_state* cores = sim_state.core_states;

    //Get some indexes
    int last_elem   = GETINDEX((left_cores-1),tid,num_of_tests);
    int death_i     = GETINDEX(dead_index,tid,num_of_tests);

    //Swap dead index to end      (COALESCED WRITE, NON COALESCED READ)
    sim_state.currR[last_elem]      = sim_state.currR[death_i];
    sim_state.loads[last_elem]      = sim_state.loads[death_i];
    sim_state.temps[last_elem]      = sim_state.temps[death_i];
    sim_state.indexes[last_elem]    = sim_state.indexes[death_i];

}
template<bool optimized>
__device__
void swapStateStruct(simulation_state sim_state,int dead_index,int left_cores,int num_of_tests){
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int* index = sim_state.indexes;
    int* value = sim_state.real_pos;
    core_state* cores = sim_state.core_states;

    //Get some indexes
    int last_elem   = GETINDEX((left_cores-1),tid,num_of_tests);
    int death_i     = GETINDEX(dead_index,tid,num_of_tests);

    if(optimized){
        cores[death_i] = cores[last_elem];
        return;
    }

    int temp = value[last_elem];
    value[last_elem] = value[death_i];
    value[death_i] = temp;

    core_state t_core = cores[last_elem];
    cores[last_elem] = cores[death_i];
    cores[death_i] = t_core;
    
    temp = index[value[last_elem]];
    index[value[last_elem]] = index[value[death_i]];
    index[value[death_i]] = temp;

    //CUDA_DEBUG_MSG("Swappato core: %d con core %d -> check %d\n",left_cores-1, dead_index,cores[death_i].real_index)
}

template<bool optimized>
__device__
void swapStateDynamic(simulation_state sim_state,int dead_index,int left_cores,int offset){

    int* index = sim_state.indexes;
    int* value = sim_state.real_pos;
    core_state* cores = sim_state.core_states;

    //Get some indexes
    int last_elem   = offset + (left_cores-1);
    int death_i     = offset + (dead_index);

    if(optimized){
        cores[death_i] = cores[last_elem];
        return;
    }

    int temp = value[last_elem];
    value[last_elem] = value[death_i];
    value[death_i] = temp;

    core_state t_core = cores[last_elem];
    cores[last_elem] = cores[death_i];
    cores[death_i] = t_core;
    
    temp = index[value[last_elem]];
    index[value[last_elem]] = index[value[death_i]];
    index[value[death_i]] = temp;

    //CUDA_DEBUG_MSG("Swappato core: %d con core %d -> check %d\n",left_cores-1, dead_index,cores[death_i].real_index)
}

__device__
void swapStateStructOptimized(simulation_state sim_state,int dead_index,int left_cores,int num_of_tests){
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int* index = sim_state.indexes;
    int* value = sim_state.real_pos;
    core_state* cores = sim_state.core_states;


    int last_elem   = GETINDEX((left_cores-1),tid,num_of_tests);
    int death_i     = GETINDEX(dead_index,tid,num_of_tests);

    cores[death_i] = cores[last_elem];
}

#endif


#endif //SIMULATION_CONFIG