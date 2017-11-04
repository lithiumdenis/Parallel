#include <cstdlib>
#include <cuda.h>
#include "cs43805351.h"

static const int ThreadsPerBlock = 512;

static const double Delta = 0.005491;
static const double xMid = 0.745796;
static const double yMid = 0.105089;

static __global__
void FractalKernel(const int width, unsigned char pic[])
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    

    if (idx < width * width) {        
        const int col = idx % width;
        const int row = (idx / width) % width;


        const double xMin = xMid - Delta;
        const double yMin = yMid - Delta;
        const double dw = 2.0 * Delta / width;
        //todo: compute a single pixel here
        if (row < width) { // bounds checking, ensures no wasted calc
            const double cy = -yMin - row * dw;
            if (col < width) { // bounds checking
                const double cx = -xMin - col * dw;
                double x = cx;
                double y = cy;
                int depth = 256;
                double x2, y2;
                do {
                    x2 = x * x;
                    y2 = y * y;
                    y = 2 * x * y + cy;
                    x = x2 - y2 + cx;
                    depth--;
                } while ((depth > 0) && ((x2 + y2) < 5.0));
                pic[row * width + col] = (unsigned char)depth;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int width = 1000;

    // allocate picture array
    unsigned char* pic = new unsigned char[width * width];
    unsigned char* pic_d;
    if (cudaSuccess != cudaMalloc((void **)&pic_d, width * width * sizeof(unsigned char))) 
    {
        fprintf(stderr, "could not allocate memory\n"); 
        exit(-1);
    }

    // compute
    FractalKernel<<<(width * width + (ThreadsPerBlock - 1)) / ThreadsPerBlock, ThreadsPerBlock>>>(width, pic_d);
    if (cudaSuccess != cudaMemcpy(pic, pic_d, width * width * sizeof(unsigned char), cudaMemcpyDeviceToHost)) 
    {
        fprintf(stderr, "copying from device failed\n"); 
        exit(-1);
    }

    // verify result by writing
    char name[32];
    sprintf(name, "result.bmp");
    writeBMP(width, width, &pic[0], name);

    delete [] pic;
    cudaFree(pic_d);
    return 0;
}

// nvcc mandelbrot.cu -o mandel.out
// ./mandel.out