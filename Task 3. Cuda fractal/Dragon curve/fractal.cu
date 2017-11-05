#include <cstdlib>
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "stdbool.h"
#include <cuda.h>
#include "lodepng.h"

#define iters 20
#define PURPLE 0xFFCC00CC //В rgb фиолетовый 0xCC00CC, а в abgr это
#define BLACK 0xFF000000

#define ThreadsPerBlock 512

enum TURN
{
  LEFT,
  RIGHT
};

enum DIRECTION
{
  D_LEFT,
  D_UP,
  D_RIGHT,
  D_DOWN
};

//precondition: turnBuf is populated
void createImage(int maxTurns, unsigned char* turnBuf, unsigned* pic, int x, int y, int iw)
{
  int dir = D_UP;

  for(int i = 0; i <= maxTurns; i++)
  {
    //move in current direction, writing pixels on segment to color
    pic[x + y * iw] = PURPLE;
    switch(dir)
    {
      case D_LEFT:
        pic[(x - 1) + y * iw] = PURPLE;
        x -= (false ? 1 : 2);
        break;
      case D_UP:
        pic[x + (y - 1) * iw] = PURPLE;
        y -= (false ? 1 : 2);
        break;
      case D_RIGHT:
        pic[(x + 1) + y * iw] = PURPLE;
        x += (false ? 1 : 2);
        break;
      case D_DOWN:
        pic[x + (y + 1) * iw] = PURPLE;
        y += (false ? 1 : 2);
        break;
      default:;
    }
    //turn (except after last position update)
    if(i != maxTurns)
    {
      if(turnBuf[i] == LEFT)
      {
        dir = (dir + 1) % 4;
      }
      else
      {
        dir = (dir + 3) % 4;
      }
    }
    else
    {
        pic[x + y * iw] = PURPLE;
    }
  }
  
}












/*static const double Delta = 0.005491;
static const double xMid = 0.745796;
static const double yMid = 0.105089;


static __global__
void FractalKernel(const int width, unsigned char pic[])
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    

    if (idx < width * width) { 
        
        



        

        const int col = idx % width;  //столбец
        const int row = (idx / width) % width; //строка


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
*/

int main(int argc, char *argv[])
{
    //Получение путей
    int numTurns = 0;
    int maxTurns = (1 << iters) - 1;
    unsigned char* turnBuf = (unsigned char*)calloc(maxTurns, sizeof(unsigned char));
    
    for(int i = 0; i < iters; i++)
    {
      //append RIGHT to the existing sequence
      turnBuf[numTurns] = RIGHT;
      //append inverse transpose of first numTurns turns
      int src = numTurns - 1;
      int dst = numTurns + 1;
      while(src >= 0)
      {
        turnBuf[dst] = (turnBuf[src] == LEFT) ? RIGHT : LEFT;
        src--;
        dst++;
      }
      //update numTurns
      numTurns = numTurns * 2 + 1;
    }

    //Вычисление размера выходного изображения путём прохода всех итераций
    //И получения минимума и максимума того, где они заканчиваются 
    int x = 0;
    int y = 0;
    int minx = 0;
    int miny = 0;
    int maxx = 0;
    int maxy = 0;
    int dir = D_UP;
    
    for(int i = 0; i <= maxTurns; i++)
    {
      //move in current direction
      switch(dir)
      {
        case D_LEFT:
          x--;
          break;
        case D_UP:
          y--;
          break;
        case D_RIGHT:
          x++;
          break;
        case D_DOWN:
          y++;
          break;
        default:;
      }
      //update bounding box if outside
      minx = x < minx ? x : minx;
      miny = y < miny ? y : miny;
      maxx = x > maxx ? x : maxx;
      maxy = y > maxy ? y : maxy;
      //turn (except after last position update)
      if(i != maxTurns)
      {
        if(turnBuf[i] == LEFT)
        {
          dir = (dir + 1) % 4;
        }
        else
        {
          dir = (dir + 3) % 4;
        }
      }
    }
  
    //set starting point to stay in [0, iw) x [0, ih)
    x = (-minx) * (false ? 1 : 2) + 1;
    y = (-miny) * (false ? 1 : 2) + 1;

    //using the bounding box, compute final image size
    //note: in real image, each position update is 2 pixels
    //also add one pixel border on all 4 edges
    int iw = (maxx - minx) * (false ? 1 : 2) + 3;
    int ih = (maxy - miny) * (false ? 1 : 2) + 3;


    unsigned* pic = (unsigned*)malloc(iw * ih * sizeof(unsigned));
    //fill image with BLACK (most pixels will be in final product)
    for(int i = 0; i < iw * ih; i++)
      pic[i] = BLACK;



    createImage(maxTurns, turnBuf, pic, x, y, iw);



    //get image filename
    char fname[64];
    sprintf(fname, "dragon.png");
    lodepng_encode32_file(fname, (unsigned char*) pic, iw, ih);
    free(pic);



    free(turnBuf);
    return 0;



    /*// allocate picture array
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
    return 0;*/
}

// nvcc fractal.cu -o cutey.out
// ./cutey.out
