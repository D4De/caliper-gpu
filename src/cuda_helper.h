#ifdef CUDA
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>


#include <sys/time.h>
#include <unordered_map>


//DEFINE DEBUG MACRO
#define CHECK(call)\
{\
    const cudaError_t err = call;\
        if (err != cudaSuccess) {\
        printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__, __LINE__);\
        exit(EXIT_FAILURE);\
    }\
}\

#define CHECK_KERNELCALL()\
{ \
    const cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}\

#ifndef PrintInfo
#define PrintInfo
void printDeviceInfo(){
    std::printf("Query Available devices\n");

    int devCount;
    cudaDeviceProp devProp;
    cudaGetDeviceCount(&devCount);
    cudaGetDeviceProperties(&devProp, 0);

    printf("Major revision number: %d\n", devProp.major);
    printf("Minor revision number: %d\n", devProp.minor);
    printf("Name: %s\n", devProp.name);
    printf("Total global memory: %lu bytes\n",
           devProp.totalGlobalMem);
    printf("Total registers per block: %d\n",
           devProp.regsPerBlock);
    printf("Maximum threads per block: %d\n",
           devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block: %d \n", i,
               devProp.maxThreadsDim[i]);
    printf("Maximum Shared Memory per block: %zu bytes\n",
           devProp.sharedMemPerBlock);
    printf("Maximum Shared Memory per multiprocessor: %zu bytes\n",
           devProp.sharedMemPerMultiprocessor);
    /*... output data visualization ...*/
}
#endif
#endif
