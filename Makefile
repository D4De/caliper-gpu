#------COMPILER SETTINGS-----------------
CXX=g++
CFLAGS=-c -O3
LDFLAGS=-lm

#------SOURCES FILES----------------------
CALIPER_SRC=src/caliper_cpu.cpp
CALIPER_GPU_SRC=src/caliper_gpu.cu
#------BUILD PATH DEFINITION--------------
BUILD_PATH=./build
CPU_BUILD_PATH=$(BUILD_PATH)/cpu
GPU_BUILD_PATH=$(BUILD_PATH)/cpu
#------OBJECT FILES-----------------------
CALIPER_OBJ=$(CALIPER_SRC:src/%.cpp=$(CPU_BUILD_PATH)/%.o)
CALIPER_GPU_OBJ=$(CALIPER_GPU_SRC:src/%.cpp=$(GPU_BUILD_PATH)/%.o)
#------EXECUTABLES------------------------
CALIPER_CPU_EX=caliper
CALIPER_GPU_EX=caliper_gpu

#PATH TO CUDA INSTALLATION FOLDER
CUDA_PATH = /usr/local/cuda-11.6
#CUDA STANDARD LIBRARY PATH
CUDA_INC = $(CUDA_PATH)/include
#If using on terminal NVCC can be replaced with nvcc directly without path
NVCC = $(CUDA_PATH)/bin/nvcc
NVPROF = $(CUDA_PATH)/bin/nvprof

#C or C++ compiler
GCC = g++
#All compilation flag must be placed here
FLAGS =
#GPU CAPABILITY: the capability is a version number composed like X.X (eg 5.0 or 6.1 )
#You can find capability of device on https://developer.nvidia.com/cuda-gpus#compute
#IN THE FOLLOWING FIELD REPLACE 50 with your GPU capability without the "." so X.X become XX
#eg: 5.0 -> 50 , 6.1->60 , 7.5 -> 75
CAPABILITY = 37
#THE CODE IS EQUAL TO THE ARCHITECTURE CAPABILITY SO MY IS 5.0 so sm_50 and compute_50
#GENERATING ARCHITECTURE FLAGS--------------- TODO AUTOMATIZE
CODE = code=sm_$(CAPABILITY)
ARCH = arch=compute_$(CAPABILITY)
ARCHITECTURE_FLAG = $(ARCH),$(CODE)

#STATISTICS FOLDERS
STATISTICS_PATH=./statistics
STATISTICS_CPU_FILE=$(STATISTICS_PATH)/cpu-times.csv
BENCHMARK_SCRIPT=./run-benchmark.sh
#----------------------------------------
#--------MAKEFILE COMMANDS---------------
#----------------------------------------

all: clean cpu gpu
#gpu: $(CALIPER_GPU_EX)
cpu: $(CALIPER_CPU_EX)
gpu:
	nvcc -g -m64 -gencode $(ARCHITECTURE_FLAG) -rdc=true -lcudadevrt  ./src/caliper_gpu.cu -o gpu_exec 

gpu-debug:
	export CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1
	nvcc -g -m64 -gencode $(ARCHITECTURE_FLAG) -rdc=true -lcudadevrt  ./src/caliper_gpu.cu -o gpu_exec
	

statistics:clean-statistics $(STATISTICS_CPU_FILE)
	#Statistics ended computation

clean-statistics:
	rm -f $(STATISTICS_CPU_FILE)
	
$(STATISTICS_CPU_FILE):
	$(BENCHMARK_SCRIPT) >> $(STATISTICS_CPU_FILE)

buid_folders:
	mkdir -p $(CPU_BUILD_PATH)
$(CALIPER_CPU_EX): buid_folders $(CALIPER_OBJ) 
	#------START Caliper CPU build------
	$(CXX) $(CALIPER_OBJ) $(LDFLAGS) -o $@
	#------END   Caliper CPU build------

$(CALIPER_GPU_EX): buid_folders $(CALIPER_GPU_OBJ)  
	#------START Caliper GPU build------
	$(NVCC) $(CALIPER_GPU_OBJ) -m64  -gencode $(ARCHITECTURE_FLAG)  -o $@
	#------END   Caliper GPU build------

$(CALIPER_OBJ) : $(CALIPER_SRC)
	#OBJ
	$(CXX) $(CFLAGS) $< -o $@

.cpp .o:
	$(CXX) $(CFLAGS) $< -o $@

clean:
	rm -rf caliper $(CALIPER_EX)
	rm -rf build/cpu/*.o $(CALIPER_EX) $(THERMAL_EX) 