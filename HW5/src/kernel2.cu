#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 8

__global__ void mandelKernel(float lowerX, float lowerY,
                             float stepX, float stepY,
                             int resX, int resY,
                             int maxInteration, int* dataCuda) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int Index_i = blockIdx.x * blockDim.x + threadIdx.x;
    int Index_j = blockIdx.y * blockDim.y + threadIdx.y;

    float c_re = lowerX + Index_i * stepX;
    float c_im = lowerY + Index_j * stepY;

    //from Mandel
    float z_re = c_re, z_im = c_im;

    int i;
    for (i = 0; i < maxInteration; ++i)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    dataCuda[Index_j * resX + Index_i] = i;
    // return i;
}
// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // Allocate GPU memory
    int size = resX * resY * sizeof(int);
    size_t pitch;
    int *dataHost;
    int *dataCuda;
    cudaHostAlloc((void **) &dataHost, size, cudaHostAllocDefault);
    cudaMallocPitch( (void **) &dataCuda, &pitch,  sizeof(int) * resX, resY);


    // 100 blocks, 10 threads per block Function launch
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);
    mandelKernel<<< numBlocks, threadsPerBlock >>>(lowerX, lowerY, stepX, stepY, resX, resY, maxIterations, dataCuda);

    // shift result from cuda to host
    cudaMemcpy(dataHost, dataCuda, size, cudaMemcpyDeviceToHost); 
    cudaFree(dataCuda);

    //shift result to img pointer
    memcpy(img, dataHost, size);
    cudaFreeHost(dataHost);
}
