#ifndef CALIPER_CPU
#define CALIPER_CPU
    #include "./thermal_model.h"
    #include "./utils.h"
#endif
double montecarlo_simulation_cpu(int num_of_tests,int max_cores,int min_cores,int rows,int cols, int wl){
     //TODO INSERT ALSO THE "CONFIDENCE LEVEL STOP CONDITION"
    
    int left_cores;
    int sumTTF = 0;
    double *loads;
    double *temps;
    loads = (double*) malloc(sizeof(double)* rows * cols);
    temps = (double*) malloc(sizeof(double)* rows * cols);


     for (int i = 0;i < num_of_tests; i++) {
        double random;
        double* currR;
        double stepT;
        std::string currConf;
        int minIndex;
        double totalTime;
        bool* alives;
        int j;
        double t, eqT;

    //-----------EXPERIMENT INITIALIZATION----------
        left_cores = max_cores;
        totalTime = 0;
        minIndex = 0;
        currConf = EMPTY_SET;
        //Arrays Allocation:
        currR  = (double*) malloc(sizeof(double)*max_cores);
        alives = (bool*)   malloc(sizeof(bool)*max_cores);
        //Arrays Initialization:
        for (j = 0; j < max_cores; j++) {
            currR[j]  = 1;
            alives[j] = true;
        }

    //-----------RUN CURRENT EXPERIMENT----------
    //THIS CYCLE generate failure times for alive cores and find the shortest one  
        while (left_cores >= min_cores) {
        //Initialize Minimum index
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

        //-----------Compute Temperatures of each core based on loads----
            
	        tempModel(loads, temps, rows, cols);
            
        //-----------Random Walk Step computation-------------------------
            for (j = 0; j < max_cores; j++) {
                if (alives[j] == true) {
                    random = (double) drand48() * currR[j]; //current core will potentially die when its R will be equal to random. drand48() generates a number in the interval [0.0;1.0)
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
                std::cerr << "Failing cores not found" << std::endl;
                return 1;
            }

        //---------UPDATE TOTAL TIME----------------- 
            // update total time by using equivalent time according to the R for the core that is dying
            //stepT is the time starting from 0 to obtain the R value when the core is dead with the current load
            //eqT is the time starting from 0 to obtain the previous R value with the current load
            //thus the absolute totalTime when the core is dead is equal to the previous totalTime + the difference between stepT and eqT
            //geometrically we translate the R given the current load to right in order to intersect the previous R curve in the previous totalTime
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
                if (currConf == EMPTY_SET)
                    currConf = MAKE_STRING(minIndex);
                else
                    currConf = MAKE_STRING(currConf << "," << minIndex);
            }
            left_cores--;
            
        }
    //---------UPDATE Stats----------------- 
    sumTTF += totalTime;
    }  
    return sumTTF

}