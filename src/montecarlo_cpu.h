#ifndef CALIPER_CPU
#define CALIPER_CPU
    #include "./simulation-utils/thermal_model.h"
    #include "./utils/utils.h"
    #include "./simulation-utils/simulation_config.h"
#endif

#define FIXED_NUM_TEST true
#define USE_CONFIDENCE false


typedef struct core_state_p{
    float temp;
    float curr_r;
    float time;
    int real_index;
    bool alive;
    bool* right;
    bool* left;
    bool* up;
    bool* down;
} core_state_p;

void tempModel_pointer(core_state_p* cores, int left_cores, float load){
    for(int i = 0; i<left_cores; i++) {
        float temp = 0.0;
        temp += *cores[i].left*load*NEIGH_TEMP;
        temp += *cores[i].right*load*NEIGH_TEMP;
        temp += *cores[i].up*load*NEIGH_TEMP;
        temp += *cores[i].down*load*NEIGH_TEMP;

        cores[i].temp = ENV_TEMP + load * SELF_TEMP + temp;
    }

}

void tempModel_optimized(core_state_p* cores, bool* alives, int left_cores, float load,int rows, int cols){
    for(int i = 0; i<left_cores; i++) {
        float temp = 0.0;
        int k, h;
        int index = cores[i].real_index;
        int r = index/cols;
        int c = index%cols;
        for (k = -1, temp = 0; k < 2; k++)
            for (h = -1; h < 2; h++)
                if ((k != 0 || h != 0) && k != h && k != -h && r + k >= 0 && r + k < rows && c + h >= 0 && c + h < cols){
                    bool alive = alives[(r+k)*cols + c+h];
                    temp += load * alive * NEIGH_TEMP;
                }
        cores[i].temp = ENV_TEMP + load * SELF_TEMP + temp;
    }
}

void swap_core_struct(core_state_p* cores, int min_index, int left_cores) {
    if (min_index > left_cores){
        std::cerr << "Invalid index for swap" << std::endl;
        return;
    }
    core_state_p temp = cores[min_index];
    cores[min_index] = cores[left_cores];
    cores[left_cores] = temp;
}

void montecarlo_simulation_cpu_optimized(configuration_description* config,double * sumTTF,double * sumTTFx2){
    core_state_p* cores;
    bool* alives;

    int i;
    int left_cores;
    double ciSize,mean,var,Zinv,th;

    int rows = config->rows;
    int cols = config->cols;

    cores = (core_state_p*) malloc(sizeof(core_state_p)* rows * cols);
    alives = (bool*) malloc(sizeof(bool)*rows*cols);

    Zinv = invErf(0.5 + config->confInt / 100.0 / 2);
    config->threshold = config->threshold / 100.0 / 2; // half of the threshold

    for (i = 0;(config->useNumOfTest && (i < config->num_of_tests)) || (!config->useNumOfTest && ((i < MIN_NUM_OF_TRIALS) || (ciSize / mean > config->threshold))); i++) {
        double random;
        double stepT;
        int minIndex;
        double totalTime;
        double t, eqT;
        int j;

        left_cores = config->max_cores;
        totalTime = 0;
        minIndex = 0;

        //Arrays Initialization:
        for (j = 0; j < config->max_cores; j++) {
            cores[j].curr_r = 1;
            cores[j].real_index = j;
            alives[j] = true;
        }

        while (left_cores >= config->min_cores) {
            //Initialize Minimum index
            minIndex = -1;
            //-----------Redistribute Loads among alive cores----------
            double distributedLoad = config->initial_work_load * config->max_cores / left_cores;
            //-----------Compute Temperatures of each core based on loads----
            tempModel_optimized(cores, alives, left_cores, distributedLoad, rows, cols);

            for (j = 0; j < left_cores; j++) {
                random = (double) drand48() * cores[j].curr_r; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                double alpha = getAlpha(cores[j].temp);
                t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                eqT = alpha * pow(-log(cores[j].curr_r), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
                t = t - eqT;

                //the difference between the two values represents the time elapsed from the previous failure to the current failure
                //(we will sum to the total time the minimum of such values)

                if (minIndex == -1 || (minIndex != -1 && t < stepT)) {
                    minIndex = j;//Set new minimum index
                    stepT = t;   //set new minimum time as timeStep
                } //TODO ADD A CHECK ON MULTIPLE FAILURE IN THE SAME INSTANT OF TIME.

            }
            //-------Check if No failed core founded-----------
            if (minIndex == -1) {
                std::cerr << "Failing cores not found" << std::endl;
                return;
            }
            //---------UPDATE TOTAL TIME-----------------
            // update total time by using equivalent time according to the R for the core that is dying
            //stepT is the time starting from 0 to obtain the R value when the core is dead with the current load
            //eqT is the time starting from 0 to obtain the previous R value with the current load
            //thus the absolute totalTime when the core is dead is equal to the previous totalTime + the difference between stepT and eqT
            //geometrically we translate the R given the current load to right in order to intersect the previous R curve in the previous totalTime
            totalTime = totalTime + stepT; //Current simulation time

            if (left_cores > config->min_cores) {
                alives[cores[minIndex].real_index] = false;
                swap_core_struct(cores,minIndex,left_cores-1);
                // compute remaining reliability for working cores
                for (j = 0; j < left_cores-1; j++) {
                    double alpha = getAlpha(cores[j].temp);
                    eqT = alpha * pow(-log(cores[j].curr_r), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
                    cores[j].curr_r = exp(-pow((stepT + eqT) / alpha, BETA));
                }
            }
            left_cores--;
        }
        //---------UPDATE Stats-----------------
        *sumTTF     += totalTime;
        *sumTTFx2   += totalTime * totalTime;
        mean = *sumTTF / (double) (i + 1); //do consider that i is incremented later
        var = *sumTTFx2 / (double) (i) - mean * mean;
        //Maybe more correct calculate var as   sqrt((totalTime - CurrMean)^2/n)
        //                          instead of  sqrt( (totalTime^2) / mean^2) / n)
        ciSize = Zinv * sqrt(var / (double) (i + 1));
        //printf("CiSize[%d]: %f\n",i,ciSize);
        //printf("%f\n",ciSize);
        //https://www.omnicalculator.com/statistics/confidence-interval#:~:text=Compute%20the%20standard%20error%20as,to%20obtain%20the%20confidence%20interval.
    }
    config->num_of_tests = i;//Final num of test
    i=0;

}

void montecarlo_simulation_cpu_pointer(configuration_description* config,double * sumTTF,double * sumTTFx2){
    core_state_p* cores;

    int i;
    int left_cores;
    double ciSize,mean,var,Zinv,th;

    int rows = config->rows;
    int cols = config->cols;

    cores = (core_state_p*) malloc(sizeof(core_state_p)* rows * cols);

    Zinv = invErf(0.5 + config->confInt / 100.0 / 2);
    config->threshold = config->threshold / 100.0 / 2; // half of the threshold

    for (i = 0;(config->useNumOfTest && (i < config->num_of_tests)) || (!config->useNumOfTest && ((i < MIN_NUM_OF_TRIALS) || (ciSize / mean > config->threshold))); i++) {
        double random;
        double stepT;
        int minIndex;
        double totalTime;
        double t, eqT;
        int j;

        left_cores = config->max_cores;
        totalTime = 0;
        minIndex = 0;

        bool false_ref = false;
        //Arrays Initialization:
        for (j = 0; j < config->max_cores; j++) {
            cores[j].curr_r = 1;
            cores[j].alive = true;

            int core_id = j;
            int r = core_id/config->cols;
            int c = core_id%config->cols;

            cores[j].right = c+1<config->cols ? &cores[r*config->max_cores + c + 1].alive : &false_ref;
            cores[j].left = c-1 >= 0 ? &cores[r*config->max_cores + c - 1].alive : &false_ref;
            cores[j].down = r+1<config->rows ? &cores[(r+1)*config->max_cores + c].alive : &false_ref;
            cores[j].up = r-1 >= 0 ? &cores[(r-1)*config->max_cores + c].alive : &false_ref;
        }

        while (left_cores >= config->min_cores) {
            //Initialize Minimum index
            minIndex = -1;
            //-----------Redistribute Loads among alive cores----------
            double distributedLoad = config->initial_work_load * config->max_cores / left_cores;
            //-----------Compute Temperatures of each core based on loads----
            tempModel_pointer(cores, left_cores, distributedLoad);

            for (j = 0; j < left_cores; j++) {
                random = (double) drand48() * cores[j].curr_r; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                double alpha = getAlpha(cores[j].temp);
                t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                eqT = alpha * pow(-log(cores[j].curr_r), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
                t = t - eqT;

                //the difference between the two values represents the time elapsed from the previous failure to the current failure
                //(we will sum to the total time the minimum of such values)

                if (minIndex == -1 || (minIndex != -1 && t < stepT)) {
                    minIndex = j;//Set new minimum index
                    stepT = t;   //set new minimum time as timeStep
                } //TODO ADD A CHECK ON MULTIPLE FAILURE IN THE SAME INSTANT OF TIME.

            }
            //-------Check if No failed core founded-----------
            if (minIndex == -1) {
                std::cerr << "Failing cores not found" << std::endl;
                return;
            }
            //---------UPDATE TOTAL TIME-----------------
            // update total time by using equivalent time according to the R for the core that is dying
            //stepT is the time starting from 0 to obtain the R value when the core is dead with the current load
            //eqT is the time starting from 0 to obtain the previous R value with the current load
            //thus the absolute totalTime when the core is dead is equal to the previous totalTime + the difference between stepT and eqT
            //geometrically we translate the R given the current load to right in order to intersect the previous R curve in the previous totalTime
            totalTime = totalTime + stepT; //Current simulation time

            if (left_cores > config->min_cores) {
                cores[minIndex].alive = false;
                swap_core_struct(cores,minIndex,left_cores-1);
                // compute remaining reliability for working cores
                for (j = 0; j < left_cores-1; j++) {
                    double alpha = getAlpha(cores[j].temp);
                    eqT = alpha * pow(-log(cores[j].curr_r), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
                    cores[j].curr_r = exp(-pow((stepT + eqT) / alpha, BETA));
                }
            }
            left_cores--;
        }
        //---------UPDATE Stats-----------------
        *sumTTF     += totalTime;
        *sumTTFx2   += totalTime * totalTime;
        mean = *sumTTF / (double) (i + 1); //do consider that i is incremented later
        var = *sumTTFx2 / (double) (i) - mean * mean;
        //Maybe more correct calculate var as   sqrt((totalTime - CurrMean)^2/n)
        //                          instead of  sqrt( (totalTime^2) / mean^2) / n)
        ciSize = Zinv * sqrt(var / (double) (i + 1));
        //printf("CiSize[%d]: %f\n",i,ciSize);
        //printf("%f\n",ciSize);
        //https://www.omnicalculator.com/statistics/confidence-interval#:~:text=Compute%20the%20standard%20error%20as,to%20obtain%20the%20confidence%20interval.
    }
    config->num_of_tests = i;//Final num of test
    i=0;
}


void montecarlo_simulation_cpu(configuration_description* config,double * sumTTF,double * sumTTFx2){
   
    //-------------------------------------------------
    //----Variables Declaration------------------------
    //-------------------------------------------------
    float * loads;
    float * temps;
    float * currR;
    bool   * alives;
    int    * indexes;
    
    int i;
    int left_cores;
    double ciSize,mean,var,Zinv,th;

    int rows = config->rows;
    int cols = config->cols;

    //-------------------------------------------------
    //----Allocate Memory------------------------------
    //-------------------------------------------------
    loads   = (float*) malloc(sizeof(float)* rows * cols);
    temps   = (float*) malloc(sizeof(float)* rows * cols);
    currR   = (float*) malloc(sizeof(float)* rows * cols);
    alives  = (bool*)  malloc(sizeof(bool)* rows * cols);
    //indexes = (int*)   malloc(sizeof(int)* rows * cols);

    //--------------------------------------------------
    //----Confidence Intervall Setup--------------------
    //--------------------------------------------------
    Zinv = invErf(0.5 + config->confInt / 100.0 / 2);
    config->threshold = config->threshold / 100.0 / 2; // half of the threshold

    //--------------------------------------------------
    //---Montecarlo Simulation--------------------------
    //--------------------------------------------------
    //TODO PUT again THE "level of confidence as possible stop condition"
    for (i = 0;(config->useNumOfTest && (i < config->num_of_tests)) || (!config->useNumOfTest && ((i < MIN_NUM_OF_TRIALS) || (ciSize / mean > config->threshold))); i++) {
        double random;
        double stepT;
        int minIndex;
        double totalTime;
        double t, eqT;
        int j;
    //-----------EXPERIMENT INITIALIZATION----------
        left_cores = config->max_cores;
        totalTime = 0;
        minIndex = 0;

        //Arrays Initialization:
        for (j = 0; j < config->max_cores; j++) {
            currR[j]  = 1;
            alives[j] = true;
            //indexes[j] = j;
        }

    //-----------RUN CURRENT EXPERIMENT----------
    //THIS CYCLE generate failure times for alive cores and find the shortest one  
        while (left_cores >= config->min_cores) {
            //Initialize Minimum index
            minIndex = -1;
        //-----------Redistribute Loads among alive cores----------
            double distributedLoad = config->initial_work_load * config->max_cores / left_cores;

            for (j = 0; j < config->max_cores; j++) {
                //int index = indexes[j];
                if(alives[j])loads[j] = distributedLoad;
            }

        //-----------Compute Temperatures of each core based on loads----
            tempModel_cpu(loads, temps, rows, cols);
	        //tempModel(loads, temps, rows, cols);
        //-----------Random Walk Step computation-------------------------
            for (j = 0; j < config->max_cores; j++) {
                if(alives[j]){
                    //int index = indexes[j]; //Current alive core
                    random = (double) drand48() * currR[j]; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                    double alpha = getAlpha(temps[j]);
                    t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                    eqT = alpha * pow(-log(currR[j]), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
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
                std::cerr << "Failing cores not found" << std::endl;
                return;
            }

        //---------UPDATE TOTAL TIME----------------- 
            // update total time by using equivalent time according to the R for the core that is dying
            //stepT is the time starting from 0 to obtain the R value when the core is dead with the current load
            //eqT is the time starting from 0 to obtain the previous R value with the current load
            //thus the absolute totalTime when the core is dead is equal to the previous totalTime + the difference between stepT and eqT
            //geometrically we translate the R given the current load to right in order to intersect the previous R curve in the previous totalTime
            totalTime = totalTime + stepT; //Current simulation time

        //---------UPDATE Configuration----------------- 
            if (left_cores > config->min_cores) {
                alives[minIndex] = false;
                loads[minIndex]  = 0;
                //swap_core_index(indexes,minIndex,left_cores,0);
                // compute remaining reliability for working cores
                
                for (j = 0; j < config->max_cores; j++) {
                    if(alives[j]){
                        //int index = indexes[j];
                        double alpha = getAlpha(temps[j]);
                        eqT = alpha * pow(-log(currR[j]), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
                        currR[j] = exp(-pow((stepT + eqT) / alpha, BETA));
                    }
                }
            }
            left_cores--;
            //Set Load of lastly dead core to 0
            //loads[indexes[left_cores-1]] = 0;
            
        }
    //---------UPDATE Stats----------------- 
    *sumTTF     += totalTime;
    *sumTTFx2   += totalTime * totalTime;
    mean = *sumTTF / (double) (i + 1); //do consider that i is incremented later
    var = *sumTTFx2 / (double) (i) - mean * mean;
    //Maybe more correct calculate var as   sqrt((totalTime - CurrMean)^2/n) 
    //                          instead of  sqrt( (totalTime^2) / mean^2) / n)
    ciSize = Zinv * sqrt(var / (double) (i + 1));
    //printf("CiSize[%d]: %f\n",i,ciSize);
    //printf("%f\n",ciSize);
    //https://www.omnicalculator.com/statistics/confidence-interval#:~:text=Compute%20the%20standard%20error%20as,to%20obtain%20the%20confidence%20interval.
    }
    config->num_of_tests = i;//Final num of test
    i=0;

    
}