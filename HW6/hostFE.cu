#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "hostFE.h"
}

#define BLOCK_SIZE 5

__global__ void convolution(float *inputImage, float *outputImage, 
                            const float *filter, int filterWidth, 
                            int imageHeight, int imageWidth)  
{
    // Iterate over the rows of the source image
    int halffilterSize = filterWidth / 2;
    float sum;
    int k, l;
    //column
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    //row
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    sum = 0;
    for (k = -halffilterSize; k <= halffilterSize; k++)
    {
        for (l = -halffilterSize; l <= halffilterSize; l++)
        {
            if (idx_y + k >= 0 && idx_y + k < imageHeight &&
                idx_x + l >= 0 && idx_x + l < imageWidth)
            {
                sum += inputImage[(idx_y + k) * imageWidth + idx_x + l] *
                        filter[(k + halffilterSize) * filterWidth +
                                l + halffilterSize];
            }
        }
    }
    // int idx = ix + iy * imageWidth;
    outputImage[idx_y * imageWidth + idx_x] = sum;
}

// Host front-end function that allocates the memory and launches the GPU kernel
extern "C"
void hostFE (int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);

    // Allocate GPU memory
    float *inputImageCuda;
    cudaMalloc( (void **) &inputImageCuda, imageSize);
    float *outputImageCuda;
    cudaMalloc( (void **) &outputImageCuda, imageSize);
    float *flitereCuda;
    cudaMalloc( (void **) &flitereCuda, filterSize);


    cudaMemcpy(inputImageCuda, inputImage, imageSize, cudaMemcpyHostToDevice); 
    cudaMemcpy(flitereCuda, filter, filterSize, cudaMemcpyHostToDevice); 



    // 100 blocks, 10 threads per block Function launch
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(imageWidth / threadsPerBlock.x, imageHeight / threadsPerBlock.y);
    convolution<<< numBlocks, threadsPerBlock >>>(inputImageCuda, outputImageCuda, flitereCuda, filterWidth, imageHeight, imageWidth);

    // shift result from cuda to host
    cudaMemcpy(outputImage, outputImageCuda, imageSize, cudaMemcpyDeviceToHost); 
    cudaFree(inputImageCuda);
    cudaFree(outputImageCuda);
    cudaFree(flitereCuda);

}
