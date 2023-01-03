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
#define CUDA
#include "montecarlo_cpu.h"
#include "montecarlo_gpu.h"
#include "utils/utils.h"
#include "utils/benchmark_helper.h"
#include "utils/args_parsing.h"


int main(int argc, char* argv[]) {

    
    std::map<double, double> results;
    int i;

    bool isGPU = false;
    char* outputfilename = NULL;
    bool numTest = true;

    unsigned short randomSeed[3] = { 0, 0, 0 };
    double confInt = 0, thr = 0;
    
    //Static Configs
    int cols,rows;
    double wl;
    long num_of_tests = NTESTS;
    int min_cores = 0, tmp_min_cores, max_cores;
    int left_cores;
    
    //GPU configs
    int gpu_version = 0;
    int block_dim = BLOCK_DIM;


    //SETUP BENCHMARK
    benchmark_results benchmark(rows,cols,min_cores,wl);
    benchmark_timer timer = benchmark_timer();

    //SETUP CONFIG STRUCT (TODO)
    //Set default values and then check for arguments setup
    configuration_description config;
    setup_config(&config,NUM_OF_TESTS,ROWS*COLS,MIN_CORES,INITIAL_WORKLOAD,ROWS,COLS,BLOCK_DIM,GPU_VERSION);
    parse_args(&config,argc,argv);

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
    //printf("ZINV: %f\n",Zinv);

    //------------------------------------------------------------------------------
    //----------Run Monte Carlo Simulation------------------------------------------
    //------------------------------------------------------------------------------
    //when using the confidence interval, we want to execute at least MIN_NUM_OF_TRIALS
    if(!config.isGPU){
        timer.start();
        montecarlo_simulation_cpu(&config,&sumTTF,&sumTTFX2);
        timer.stop();
    }else{
        //printDeviceInfo();
        timer.start();
        montecarlo_simulation_cuda_launcher(&config,&sumTTF,&sumTTFX2);
        timer.stop();
    }
    

    //----CALCULATE OTHER RESULTS-----------
    //std::cout<<"SumTTF : \t"<<sumTTF<<"\n";
    mean = sumTTF / (double) (config.num_of_tests); //do consider that num_of_tests is equal to i at end cycle 
    var = sumTTFX2 / (double) (config.num_of_tests-1) - mean * mean;
    ciSize = Zinv * sqrt(var / (double) (config.num_of_tests));
    //printf("CiSize[%d]: %f\n",num_of_tests,ciSize);

    //------------------------------------------------------------------------------
    //---------Display Results------------------------------------------------------
    //------------------------------------------------------------------------------

    saveOnFile(&config,results,outputfilename);
    double mttf_int = (sumTTF / config.num_of_tests);

    std::cout<<config.rows<<","<<config.cols<<","<<config.rows*config.cols<<","<<config.min_cores<<","<<config.initial_work_load<<","<<config.num_of_tests<<","<<ciSize<<","<<((double) timer.getTime())<<","<<mttf_int<<","<<(mttf_int / (24 * 365))<<"\n";
    benchmark.set_results(mttf_int,timer.getTime(),mean,var,ciSize);
    benchmark.save_results("benchmark.txt");
    return 0;
}
