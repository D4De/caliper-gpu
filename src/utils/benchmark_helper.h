#ifndef BENCHMARK
#define BENCHMARK
#include <sys/time.h>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>

/**
 * Get current time
 * @return
 */
double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/**
 * Allow to simply calculate time of functions
 * TODO add GPU timing function
 */
class benchmark_timer{
    
    double start_time;
    double end_time;

    public:

    benchmark_timer(){

    }

    void start(){
        start_time = get_time();
    }

    void stop(){
        end_time = get_time();
    }

    double getTime(){
        return end_time - start_time;
    }
};

class benchmark_results{

public:
    std::string configuration; //Eg : [ROW][COL][MINCORE][]
    double mttf_int;
    double execution_time;
    double mean;
    double variance;
    double confidence_interval;

    benchmark_results(){}

    benchmark_results(int row,int col, int mincore,float workload){
        //TODO compact configuration into a string
        std::ostringstream ss;

        ss<<"["<<row<<"]"<< "["<<col<<"]"<< + "["<<mincore<<"]"<< + "["<<workload<<"]";
        configuration = std::string(ss.str());
    }

    void set_results(double mttf,double exe_time,double m,double v,double confidence){
        mttf_int = mttf;
        execution_time = exe_time;
        mean = m;
        variance = v;
        confidence_interval = confidence;
    }

    void save_results(std::string output_file_name){
        std::ofstream outfile(output_file_name.c_str());
        //Save configuration On file
        outfile<<configuration<<std::endl;
        outfile<<execution_time<<std::endl;
        outfile<<mttf_int<<std::endl;
        outfile<<mean<<std::endl;
        outfile<<variance<<std::endl;
        outfile<<sqrt(variance)<<std::endl;
        outfile<<sqrt(variance) / mean<<std::endl;
        outfile<<confidence_interval<<std::endl;
    }   

    static bool compare_results(std::string file1, std::string file2){

        benchmark_results f1 = load_results(file1);
        benchmark_results f2 = load_results(file1);

        //PRINT THE EXECUTION TIMES COMPARISON
        return compare_results(f1,f2);
    }

    static benchmark_results load_results(std::string file){
        benchmark_results out;

        std::ifstream benchmark_file(file.c_str());
        std::string line;   
        if (benchmark_file.is_open()) {
            getline(benchmark_file, line);
            out.configuration = line;
            //MTTF
            getline(benchmark_file, line);
            out.mttf_int = std::stod(line);
            //MEAN
            getline(benchmark_file, line);
            out.mean = std::stod(line);
            //VARIANCE
            getline(benchmark_file, line);
            out.variance = std::stod(line);
            //SKIP LINES
            getline(benchmark_file, line);
            getline(benchmark_file, line);
            //Confidence Intervall
            getline(benchmark_file, line);
            out.confidence_interval = std::stod(line);
        }
        return out;
    }

    static bool compare_results(benchmark_results a, benchmark_results b){
        return a.mttf_int == b.mttf_int && 
               a.mean == b.mean &&
               a.variance == b.variance &&
               a.confidence_interval == b.confidence_interval;
    }

    
};

#endif //BENCHMARK