#include "../simulation-utils/simulation_config.h"


//TODO CHECK ARGS HERE (with some if statement or with original caliper techniques)
void parse_args(configuration_description* config,int argc, char* argv[]){

    //TODO REWRITE THIS IN A CLEANER WAY
    //-------------------------------------------------
    //----parsing input arguments----------------------
    //-------------------------------------------------
    config->rows            = atoi(argv[1]);
    config->cols            = atoi(argv[2]);
    config->min_cores       = atoi(argv[3]);
    config->max_cores       = config->rows * config->cols;
    config->initial_work_load = atof(argv[4]);
    
    int index_offset = 0;   //This allow to have multiple arguments simply in priority order
    
    //Use confidence intervall

    
    //TODO put args parsing inside a function dedicated and save it in config
    if(argc > 5 && argv[5][1] == 'n')           //NUM OF ITERATION
    {
        config->num_of_tests = atoi(argv[6]);
        config->useNumOfTest = true;
        index_offset += 2;
    }else if(argc > 5 && argv[5][1] == 'c')     //CONFIDENCE INTERVALL
    {
        config->confInt = atof(argv[6]);
        config->threshold = atof(argv[7]);
        config->useNumOfTest = false;
        index_offset += 3;
    }

    
    //Use GPU                                   //USE GPU
    if(argc > 5+index_offset &&  argv[5+index_offset][1] == 'g'){
        config->isGPU = true;
        

        if(argc > 6+index_offset ){
            config->gpu_version = atoi(argv[6+index_offset]);
            index_offset++;
        }

        if(argc > 6+index_offset ){
            config->block_dim = atoi(argv[6+index_offset]);
            index_offset++;
        }
        
        index_offset++;
        //TODO SETUP NUM OF BLOCKS
    }

    //check both stopping threshold and confidence interval are set if 1 is false
    if (!config->useNumOfTest && (config->confInt == 0 || config->threshold == 0)) {
        if (!(config->confInt == 0 && config->threshold == 0)) {
            std::cerr << "Confidence Interval / Stopping threshold missing!" << std::endl;
            exit(1);
        } else {
            config->useNumOfTest = true;
        }
    }
}