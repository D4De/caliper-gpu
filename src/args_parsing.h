
void check_gpu_args(){

}

void check_confidence_args(){

}

void check_config_args(){

}


//TODO CHECK ARGS HERE (with some if statement or with original caliper techniques)
void parse_args(int argc, char* argv[]){

    int curr_index = 0;

    while(curr_index < argc){
        
        //-n (num of test)
        //Num of Iteration

        //-g (grid)
        //Num of Rows
        //Num of Columns

        //-w (workload)
        //Initial Workload

        //-a (alive)
        //Minimum Cores Alive
        //Check if user config a grid or workload or mincore

        //-c [threshold][conf](confidence)
        //Check wanted confidence intervall

        //-g [BlockDim] (gpu)
        //Check if user want gpu (GPU falg or BLOCK dimension)
    }
}