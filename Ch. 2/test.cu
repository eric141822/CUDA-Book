#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>


__global__
void vecAddKernel(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vecAdd(float* a, float* b, float* c, int n) {
    int size = n * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_b, size);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_c, size);
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_a, d_b, d_c, n);
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    float *a, *b, *c;
    
    int n = 100000;
    a = (float*)malloc(n * sizeof(float));
    b = (float*)malloc(n * sizeof(float));
    c = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }
    // performance test
    // start timer
    clock_t start, end;
    start = clock();
    vecAdd(a, b, c, n);
    end = clock();

    printf("CUDA Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    // end timer

    // regular for-loop

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }
    float *c2 = (float*)malloc(n * sizeof(float));   
    start = clock();
    for (int i = 0; i < n; i++) {
        c2[i] = a[i] + b[i];
    }
    end = clock();
    for (int i = 0; i < n; i++) {
        if (c[i] != c2[i]) {
            printf("Error\n");
            break;
        }
    }
    printf("CPU Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}