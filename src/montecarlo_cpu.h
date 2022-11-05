#ifndef CALIPER_CPU
#define CALIPER_CPU
    #include "./simulation-utils/thermal_model.h"
    #include "./utils/utils.h"
    #include "./simulation-utils/simulation_config.h"
#endif

#define FIXED_NUM_TEST true
#define USE_CONFIDENCE false


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
    indexes = (int*)   malloc(sizeof(int)* rows * cols);

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
            indexes[j] = j;
        }

    //-----------RUN CURRENT EXPERIMENT----------
    //THIS CYCLE generate failure times for alive cores and find the shortest one  
        while (left_cores >= config->min_cores) {
            //Initialize Minimum index
            minIndex = -1;
        //-----------Redistribute Loads among alive cores----------
            double distributedLoad = config->initial_work_load * config->max_cores / left_cores;

            for (j = 0; j < left_cores; j++) {
                int index = indexes[j];
                loads[index] = distributedLoad;
            }
            //Set Load of lastly dead core to 0
            loads[indexes[left_cores-1]] = 0;

        //-----------Compute Temperatures of each core based on loads----
            tempModel(loads, temps,indexes,left_cores, rows, cols,0);
	        //tempModel(loads, temps, rows, cols);
        //-----------Random Walk Step computation-------------------------
            for (j = 0; j < left_cores; j++) {
                    int index = indexes[j]; //Current alive core
                    random = (double) drand48() * currR[index]; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
                    double alpha = getAlpha(temps[index]);
                    t = alpha * pow(-log(random), (double) 1 / BETA); //elapsed time from 0 to obtain the new R value equal to random
                    eqT = alpha * pow(-log(currR[index]), (double) 1 / BETA); //elapsed time from 0 to obtain the previous R value
                    t = t - eqT;
                    if(i==1){
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
                swap_core_index(indexes,minIndex,left_cores,0);
                // compute remaining reliability for working cores
                for (j = 0; j < left_cores; j++) {
                    int index = indexes[j];
                    		double alpha = getAlpha(temps[index]);
                            eqT = alpha * pow(-log(currR[index]), (double) 1 / BETA); //TODO: fixed a buf. we have to use the eqT of the current unit and not the one of the failed unit
                            currR[index] = exp(-pow((stepT + eqT) / alpha, BETA));
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