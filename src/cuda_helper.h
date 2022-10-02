
#if defined(CUDA)
    #include <cuda_runtime.h>
    #include <curand.h>
    #include <curand_kernel.h>
#endif

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
}  \

