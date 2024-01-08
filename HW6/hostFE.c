#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);
    // Create a command queue
    cl_command_queue commandQueue = clCreateCommandQueue(*context, *device, 0, &status);

    //create memory buffer
    cl_mem inputImageBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, imageSize , inputImage, &status);
    CHECK(status, "clCreateBuffer Input");
    cl_mem outputImageBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, imageSize, outputImage, &status);
    CHECK(status, "clCreateBuffer Output");
    cl_mem fliterImageBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, filterSize, filter, &status);
    CHECK(status, "clCreateBuffe Fliter");

    //Write host Data to device Buffer
    clEnqueueWriteBuffer(commandQueue, inputImageBuffer, CL_TRUE, 0, imageSize, (void *)inputImage, 0, NULL, NULL);
    clEnqueueWriteBuffer(commandQueue, fliterImageBuffer, CL_TRUE, 0, filterSize, (void *)filter, 0, NULL, NULL);
    
    //create kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    CHECK(status, "clCreateKernel");

    //setting the kernel arguments
    clSetKernelArg(kernel, 0 , sizeof(cl_mem), (void *)&inputImageBuffer);
    clSetKernelArg(kernel, 1 , sizeof(cl_mem), (void *)&outputImageBuffer);
    clSetKernelArg(kernel, 2 , sizeof(cl_mem), (void *)&fliterImageBuffer);
    clSetKernelArg(kernel, 3 , sizeof(cl_int), (void *)&filterWidth);
    clSetKernelArg(kernel, 4 , sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernel, 5 , sizeof(cl_int), (void *)&imageWidth);

    //
    size_t globalWorkSize[2] = {imageWidth, imageHeight};
    size_t localWorkSize[2] = {10, 10};
    clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    clEnqueueReadBuffer(commandQueue, outputImageBuffer, CL_TRUE, 0, imageSize, (void *)outputImage, NULL, NULL, NULL);

}
