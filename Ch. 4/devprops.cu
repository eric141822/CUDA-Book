#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main() {
    int devCount;
    cudaGetDeviceCount(&devCount);

    printf("There %s %d CUDA device(s).\n", devCount <= 1 ? "is" : "are", devCount);

    cudaDeviceProp devProp;
    for (unsigned int i = 0; i < devCount; ++i) {
        cudaGetDeviceProperties(&devProp, i);
        printf("Device %d has %d SMs (streaming multiprocessors).\n", i, devProp.multiProcessorCount);
        printf("Device %d has compute capability %d.%d.\n", i, devProp.major, devProp.minor);
        printf("Device %d has a maximum of %d threads per block.\n", i, devProp.maxThreadsPerBlock);
        printf("Device %d has a maximum of %d threads per multiprocessor.\n", i, devProp.maxThreadsPerMultiProcessor);
        printf("Device %d has a maximum of %d blocks per multiprocessor.\n", i, devProp.maxBlocksPerMultiProcessor);
        printf("Device %d has a maximum of %d warps per multiprocessor.\n", i, devProp.maxThreadsPerMultiProcessor / devProp.warpSize);
        printf("Device %d has a maximum of %d warps per block.\n", i, devProp.maxThreadsPerBlock / devProp.warpSize);
        printf("Device %d has a maximum of %d threads per warp.\n", i, devProp.warpSize);
        printf("Device %d has a maximum of %d registers per block.\n", i, devProp.regsPerBlock);
        printf("Device %d has a maximum of %d registers per multiprocessor.\n", i, devProp.regsPerMultiprocessor);
        // printf("Device %d has a maximum of %d shared memory per block.\n", i, devProp.sharedMemPerBlock);
        // printf("Device %d has a maximum of %d shared memory per multiprocessor.\n", i, devProp.sharedMemPerMultiprocessor);
        // printf("Device %d has a maximum of %d bytes of constant memory.\n", i, devProp.totalConstMem);
        // printf("Device %d has a maximum of %d bytes of global memory.\n", i, devProp.totalGlobalMem);
        // printf("Device %d has a maximum of %d bytes of memory pitch.\n", i, devProp.memPitch);
    }

    return 0;
}