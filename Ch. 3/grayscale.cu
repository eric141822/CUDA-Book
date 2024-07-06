#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHANNELS (3)
#define BLUR_SIZE (1)

__global__
void colortoGrayscaleConvertion(unsigned char *Pout, unsigned char *Pin, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int grayOffset = row*width + col;
        int rgbOffset = grayOffset*CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

__global__
void blurKernel(unsigned char * Pout, unsigned char * Pin, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int pixVal = 0;
        int pixels = 0;
        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    pixVal += Pin[curRow*width + curCol];
                    pixels++;
                }
            }
        }
        Pout[row*width + col] = (unsigned char)(pixVal / pixels);
    }
}

__global__
void matMulKernel(float *M, float *N, float *P, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        float Pvalue = 0;
        for (int k = 0; k < width; ++k) {
            Pvalue += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = Pvalue;
    }
}

void grayscale(const char* fname) {
    int width, height, channels;
    unsigned char *h_Pin = stbi_load(fname, &width, &height, &channels, CHANNELS);
    if (!h_Pin) {
        printf("Error: Unable to load image %s\n", fname);
        return;
    }

    // Allocate host memory for the output image
    unsigned char *h_Pout = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    if (!h_Pout) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        stbi_image_free(h_Pin);
        return;
    }

    // Allocate device memory
    unsigned char *d_Pin, *d_Pout;
    cudaMalloc((void**)&d_Pin, width * height * CHANNELS * sizeof(unsigned char));
    cudaMalloc((void**)&d_Pout, width * height * sizeof(unsigned char));

    // Copy image data to device
    cudaMemcpy(d_Pin, h_Pin, width * height * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Launch kernel
    colortoGrayscaleConvertion<<<grid, block>>>(d_Pout, d_Pin, width, height);

    // Copy result back to host
    cudaMemcpy(h_Pout, d_Pout, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save the grayscale image using stb_image_write
    stbi_write_png("grayscale_image.png", width, height, 1, h_Pout, width);

    // Free device and host memory
    cudaFree(d_Pin);
    cudaFree(d_Pout);
    free(h_Pout);
    stbi_image_free(h_Pin);
}

void blur(const char* fname) {
        int width, height, channels;
    unsigned char *h_Pin = stbi_load(fname, &width, &height, &channels, CHANNELS);
    if (!h_Pin) {
        printf("Error: Unable to load image %s\n", fname);
        return;
    }

    // Allocate host memory for the output image
    unsigned char *h_Pout = (unsigned char*)malloc(width * height * CHANNELS * sizeof(unsigned char));
    if (!h_Pout) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        stbi_image_free(h_Pin);
        return ;
    }

    // Allocate device memory
    unsigned char *d_Pin, *d_Pout;
    cudaMalloc((void**)&d_Pin, width * height * CHANNELS * sizeof(unsigned char));
    cudaMalloc((void**)&d_Pout, width * height * CHANNELS *sizeof(unsigned char));

    // Copy image data to device
    cudaMemcpy(d_Pin, h_Pin, width * height * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Launch kernel
    blurKernel<<<grid, block>>>(d_Pout, d_Pin, width, height);

    // Copy result back to host
    cudaMemcpy(h_Pout, d_Pout, width * height * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save the grayscale image using stb_image_write
    stbi_write_png("blur_image.png", width, height, 1, h_Pout, width);

    // Free device and host memory
    cudaFree(d_Pin);
    cudaFree(d_Pout);
    free(h_Pout);
    stbi_image_free(h_Pin);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <image path>\n", argv[0]);
        return -1;
    }

    grayscale(argv[1]);
    blur(argv[1]);

    printf("Grayscale image saved as grayscale_image.png\n");
    printf("Blurred image saved as blur_image.png\n");
    return 0;
}


