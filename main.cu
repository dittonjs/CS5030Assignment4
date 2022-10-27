#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>

__global__
void RGBToGrayscale(unsigned char * grayImage, unsigned char * rgbImage, int width, int height)
{
    int CHANNELS = 3;
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height) {
        // get 1D coordinate for the grayscale image
        int grayOffset = Row*width + Col;
        // one can think of the RGB image having
        // CHANNEL times columns of the gray scale image
        int rgbOffset = grayOffset*CHANNELS;
        unsigned char r = rgbImage[rgbOffset ]; // red value for pixel
        unsigned char g = rgbImage[rgbOffset + 1]; // green value for pixel
        unsigned char b = rgbImage[rgbOffset + 2]; // blue value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

int main() {

    // READ IMAGE
    int image_size = 1024 * 1024;
    unsigned char * image_h = (unsigned char *)malloc((image_size * 3));
    FILE *file = fopen("gc_conv_1024x1024.raw", "rb");
    fread(image_h, (image_size * 3), 1, file);
    fclose(file);

    // ALLOCATE MEMORY
    unsigned char * gray_image_h = (unsigned char *)calloc(image_size, 1);
    unsigned char * image_d, * gray_image_d;
    cudaMalloc((void **) &image_d, image_size * 3);
    cudaMalloc((void **) &gray_image_d, image_size);
    cudaDeviceSynchronize();

    // COPY IMAGE TO DEVICE
    cudaMemcpy( image_d, image_h, image_size * 3, cudaMemcpyHostToDevice );
    cudaDeviceSynchronize();

    // INITIALIZE BLOCK DIMENTIONS
    dim3 DimGrid(ceil(1024/8), ceil(1024/8), 1);
    dim3 DimBlock(8, 8, 1);

    // RUN DEVICE CODE
    RGBToGrayscale<<< DimGrid, DimBlock >>>(gray_image_d, image_d, 1024, 1024);

    cudaDeviceSynchronize();

    // COPY GRAYSCALE BACK INTO HOST MEMORY
    cudaMemcpy(gray_image_h, gray_image_d, image_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // WRITE THE FILE!
    FILE *outfile = fopen("gc.raw", "wb");
    fwrite(gray_image_h, 1, image_size, outfile);
    fclose(outfile);

    // UN ALLOCATE MEMORY
    free(image_h);
    free(gray_image_h);
    cudaFree(image_d);
    cudaFree(gray_image_d);
    return 0;
}