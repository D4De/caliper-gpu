/*
****************************** Copyright & License *****************************
CALIPER v1.0 is a software framework for the reliability lifetime evaluation of Multicore architectures. Copyright (C) 2014 Politecnico di Milano.

This framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details (http://www.gnu.org/licenses/).

Neither the name of Politecnico di Milano nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
********************************************************************************

 This code implements the R/MTTF computation for a K-out-of-N system by means of Monte Carlo simulations, for the CALIPER framework
 */


//CUSTOM HEADERS:

#include "thermal_model.h"
#include "utils.h"
#include "benchmark_helper.h"

#define BLOCK_DIM 128

int main(int argc, char* argv[]) {
    int left_cores, min_cores = 0, tmp_min_cores, max_cores;
    long num_of_tests = NTESTS;
    std::map<double, double> results;
    int i;

    bool isGPU = false;
    char* outputfilename = NULL;
    bool numTest = false;

    unsigned short randomSeed[3] = { 0, 0, 0 };
    double confInt = 0, thr = 0;
    
    int cols,rows;
    double wl;
    double *loads;
    double *temps;

    //TODO Inizialize Loads using params from user

    //-------------------------------------------------
    //----parsing input arguments----------------------
    //-------------------------------------------------
    rows            = atoi(argv[1]);
    cols            = atoi(argv[2]);
    min_cores       = atoi(argv[3]);
    max_cores       = rows * cols;
    wl              = atof(argv[4]);
    //outputfilename  = argv[5];
    
    //-------------------------------------------------
    //----Allocate Memory------------------------------
    //-------------------------------------------------
    loads = (double*) malloc(sizeof(double)* rows * cols);
    temps = (double*) malloc(sizeof(double)* rows * cols);

    benchmark_results benchmark(rows,cols,min_cores,wl);
    benchmark_timer timer = benchmark_timer();

	//-------------------------------------------------
    //-----Check QUOS constraint-----------------------
    //-------------------------------------------------

    //check both stopping threshold and confidence interval are set if 1 is false
    if (!numTest && (confInt == 0 || thr == 0)) {
        if (!(confInt == 0 && thr == 0)) {
            std::cerr << "Confidence Interval / Stopping threshold missing!" << std::endl;
            exit(1);
        } else {
            numTest = true;
        }
    }

    //-------------------------------------------------
    //-----set up environment--------------------------
    //-------------------------------------------------
    //set seed
    seed48(randomSeed);
    //set output precision
    std::cout << std::setprecision(10);
    std::cerr << std::setprecision(10);

    //if not specified in the arguments, set the minimum number of cores read from the file
   	if (min_cores > max_cores) {
        std::cerr << "The minimum number of cores " << min_cores << " cannot be greater than the initial number of cores " << max_cores << std::endl;
        return 1;
    }

    // confidence interval set up
    double Zinv = invErf(0.5 + confInt / 100.0 / 2);
    double ht = thr / 100.0 / 2; // half of the threshold
    double sumTTF = 0, sumTTFX2 = 0; //sum of times to failure and sum of squared times to failure
    double ciSize = 0; // current size of the confidence interval
    double mean;   // current mean of the distribution
    double var;	   // current variance of the distribution


    //------------------------------------------------------------------------------
    //----------Run Monte Carlo Simulation------------------------------------------
    //------------------------------------------------------------------------------
    //when using the confidence interval, we want to execute at least MIN_NUM_OF_TRIALS
    timer.start();
    for (i = 0; (numTest && (i < num_of_tests)) || (!numTest && ((i < MIN_NUM_OF_TRIALS) || (ciSize / mean > ht))); i++) {
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
            if (distributedLoad > 1) {
                std::cerr << "QoS not satisfied with less than " << (left_cores + 1) << " cores" << std::endl;
                return 1;
            }
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
        results[totalTime]++; //
        sumTTF += totalTime;
        sumTTFX2 += totalTime * totalTime;
        mean = sumTTF / (double) (i + 1); //do consider that i is incremented later
        var = sumTTFX2 / (double) (i) - mean * mean;
        ciSize = Zinv * sqrt(var / (double) (i + 1));
    }
     timer.stop();

   
    //------------------------------------------------------------------------------
    //---------Display Results------------------------------------------------------
    //------------------------------------------------------------------------------
    if (!numTest)
        num_of_tests = i;
    double curr_alives = num_of_tests;
    double prec_time = 0;
    double mttf_int = (sumTTF / num_of_tests);
    double mttf_int1 = 0;
    if (outputfilename) {
        std::ofstream outfile(outputfilename);
        if (results.count(0) == 0) {
            results[0] = 0;
        }
        //TODO understand if it is important
        for (std::map<double, double>::iterator mapIt = results.begin(); mapIt != results.end(); mapIt++) {
            curr_alives = curr_alives - mapIt->second;
            mttf_int1 = mttf_int1 + curr_alives / num_of_tests * (mapIt->first - prec_time);
            prec_time = mapIt->first;
            outfile << mapIt->first << " " << (curr_alives / num_of_tests) << std::endl;
        }
        outfile.close();
    }

    std::cout << "SumTTF: " <<sumTTF<< std::endl;
    std::cout << "MTTF: " << mttf_int << " (years: " << (mttf_int / (24 * 365)) << ") " << mttf_int1 << std::endl;
    std::cout << "Exec time: " << ((double) timer.getTime())<< std::endl;
    std::cout << "Number of tests performed: " << num_of_tests << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Variance: " << var << std::endl;
    std::cout << "Standard Deviation: " << sqrt(var) << std::endl;
    std::cout << "Coefficient of variation: " << (sqrt(var) / mean) << std::endl;
    std::cout << "Confidence interval: " << mean - ciSize << " " << mean + ciSize << std::endl;

    
    benchmark.set_results(mttf_int,timer.getTime(),mean,var,ciSize);
    benchmark.save_results("benchmark.txt");
    return 0;
}