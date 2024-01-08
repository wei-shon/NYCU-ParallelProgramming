__kernel void convolution(__global float *inputImage, __global float *outputImage, 
                          __global float *filter, int filterWidth, 
                         int imageHeight, int imageWidth) 
{

    // Iterate over the rows of the source image
    int halffilterSize = filterWidth / 2;
    float sum;
    int k, l;
    //column
    int idx_x = get_global_id(0);
    //row
    int idx_y = get_global_id(1);

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
    outputImage[idx_y * imageWidth + idx_x] = sum;
}
