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
#include "caliper_gpu.h"
#include "caliper_cpu.h"
#define BLOCK_DIM 256


int main(int argc, char* argv[]) {
    int left_cores, min_cores = 0, tmp_min_cores, max_cores;
    long num_of_tests = NTESTS;
    std::map<double, double> results;
    int i;

    bool isGPU = false;
    char* outputfilename = NULL;
    bool numTest = false;

    unsigned short randomSeed[3] = { 23, 132, 222 };
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

    if(argv[5][1] == 'g'){
        isGPU = true;
    }

    if(strcmp(argv[5],"-o")){
        outputfilename  = argv[6];
    }
    
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
   
    if(isGPU){

        //TODO: Put all this Code inside a host function
        std::cout<<"EXECUTED ON GPU:"<<std::endl;

        //----CUDA variables:------------
        float *sumTTF_GPU;   //GPU result sumTTF
        float *sumTTFx2_GPU; //GPU result sumTTFx2
        float res;
        curandState_t *states;
        
        //----MEMORY ALLOCATION:---------
        CHECK(cudaMalloc(&sumTTF_GPU  , sizeof(float)));   //Allocate Result memory
        CHECK(cudaMalloc(&sumTTFx2_GPU, sizeof(float))); //Allocate Result memory
        CHECK(cudaMalloc(&states, num_of_tests*sizeof(curandState_t))); //Random States array

        //----Declare Grid Dimensions:---------
        dim3 blocksPerGrid((num_of_tests+BLOCK_DIM-1)/BLOCK_DIM,1,1);
        dim3 threadsPerBlock(BLOCK_DIM,1,1);

        //----KERNELS CALL -------------------
        //Inizialize Random states
        init<<<blocksPerGrid,threadsPerBlock>>>(time(NULL),states);
        cudaDeviceSynchronize();
        CHECK_KERNELCALL();
        timer.start();
        //Execute Montecarlo simulation on GPU
        montecarlo_simulation_cuda<<<blocksPerGrid,threadsPerBlock>>>(states,num_of_tests,max_cores,min_cores,rows,cols,wl,sumTTF_GPU,sumTTFx2_GPU);
        cudaDeviceSynchronize();
        CHECK_KERNELCALL();
        timer.stop();
        //----Copy back results on CPU-----------
        CHECK(cudaMemcpy(&res, sumTTF_GPU, sizeof(float), cudaMemcpyDeviceToHost));
        sumTTF = (double) res;
        CHECK(cudaMemcpy(&res, sumTTFx2_GPU, sizeof(float), cudaMemcpyDeviceToHost));
        sumTTFX2 = (double) res;
        
        //----FREE CUDA MEMORY------------------
        CHECK(cudaFree(sumTTF_GPU));
        CHECK(cudaFree(states));
        
    }else{
        //when using the confidence interval, we want to execute at least MIN_NUM_OF_TRIALS
        timer.start();
        montecarlo_simulation_cpu(num_of_tests,max_cores,min_cores,rows,cols,wl,&sumTTF,&sumTTFX2);
        timer.stop();
    }

    //----CALCULATE OTHER RESULTS-----------
    std::cout<<"SumTTF : \t"<<sumTTF<<"\n";
    mean = sumTTF / (double) (num_of_tests + 1); //do consider that i is incremented later
    var = sumTTFX2 / (double) (num_of_tests) - mean * mean;
    ciSize = Zinv * sqrt(var / (double) (num_of_tests + 1));
   
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