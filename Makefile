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

#----------------------------------------
#--------MAKEFILE COMMANDS---------------
#----------------------------------------

all: $(CALIPER_CPU_EX) 
gpu: $(CALIPER_GPU_EX)
cpu: $(CALIPER_EX)

buid_folders:
	mkdir -p $(CPU_BUILD_PATH)

$(CALIPER_CPU_EX): buid_folders $(CALIPER_OBJ) 
	#------START Caliper CPU build------
	$(CXX) $(CALIPER_OBJ) $(LDFLAGS) -o $@
	#------END   Caliper CPU build------

$(CALIPER_GPU_EX): $(CALIPER_GPU_OBJ)  
	#------START Caliper GPU build------
	$(CXX) $(THERMAL_OBJ) $(LDFLAGS) -o $@
	#------END   Caliper GPU build------

$(CALIPER_OBJ) : $(CALIPER_SRC)
	#OBJ
	$(CXX) $(CFLAGS) $< -o $@

.cpp .o:
	$(CXX) $(CFLAGS) $< -o $@
	
clean:
	rm -rf src/*.o $(CALIPER_EX) $(THERMAL_EX) 