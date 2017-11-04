//#include <cstdlib>
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "stdbool.h"
#include <cuda.h>
//#include "cs43805351.h"


#include "lodepng.h"

static const int ThreadsPerBlock = 512;

#define SET_PIXEL(x, y) {pixels[(x) + (y) * iw] = lerp(blend1, blend2, (double) i / maxTurns);} 

typedef unsigned char Pixel;
typedef unsigned char Turn;

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

Pixel* buf;
Turn* turnBuf;
unsigned BLACK = 0x000000;
unsigned WHITE = 0xFFFFFF;
unsigned blend1;
unsigned blend2;
bool dense;
int iters;
int maxTurns;
double partial;
int w;    //image width
int h;    //image height

int cx;   //center x (axis of rotation)
int cy;   //center y

unsigned lerp(unsigned c1, unsigned c2, double k)
{
  //rgba -> abgr, a omitted from a, b
  unsigned r = (c1 & 0xFF) * (1 - k) + (c2 & 0xFF) * k;
  unsigned g = ((c1 & 0xFF00) >> 8) * (1 - k) + ((c2 & 0xFF00) >> 8) * k;
  unsigned b = ((c1 & 0xFF0000) >> 16) * (1 - k) + ((c2 & 0xFF0000) >> 16) * k;
  return 0xFF000000 | (b << 16) | (g << 8) | r;
}

//precondition: turnBuf is populated
void createImage()
{
  //filename format: dragonN.png, where N is iteration count
  //calculate bounding box of resulting image by walking through the turns
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
  //using the bounding box, compute final image size
  //note: in real image, each position update is 2 pixels
  //also add one pixel border on all 4 edges
  int iw = (maxx - minx) * (dense ? 1 : 2) + 3;
  int ih = (maxy - miny) * (dense ? 1 : 2) + 3;
  //unsigned char* pixels = new unsigned char[iw * ih];     /////////////////////////
  unsigned* pixels = (unsigned*)malloc(iw * ih * sizeof(unsigned));
  //fill image with BLACK (most pixels will be in final product)
  for(int i = 0; i < iw * ih; i++)
    pixels[i] = (0xFF000000 | BLACK);
  //set starting point to stay in [0, iw) x [0, ih)
  x = (-minx) * (dense ? 1 : 2) + 1;
  y = (-miny) * (dense ? 1 : 2) + 1;
  dir = D_UP;

  for(int i = 0; i <= maxTurns * partial; i++)
  {
    //move in current direction, writing pixels on segment to WHITE
    SET_PIXEL(x, y);
    switch(dir)
    {
      case D_LEFT:
        SET_PIXEL(x - 1, y);
        x -= (dense ? 1 : 2);
        break;
      case D_UP:
        SET_PIXEL(x, y - 1);
        y -= (dense ? 1 : 2);
        break;
      case D_RIGHT:
        SET_PIXEL(x + 1, y);
        x += (dense ? 1 : 2);
        break;
      case D_DOWN:
        SET_PIXEL(x, y + 1);
        y += (dense ? 1 : 2);
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
      SET_PIXEL(x, y);
    }
  }
  //get image filename
  char fname[64];
  sprintf(fname, "dragon%i.png", iters);
  lodepng_encode32_file(fname, (unsigned char*) pixels, iw, ih);
  free(pixels);
}

//precondition: turnBuf is allocated to exact size required
void getPath()
{
  int numTurns = 0;
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
    //Разрешение квадратного изображения
    //int width = 1000;
    //printf("sex\n");
    //set default option values
    iters = 20;
    dense = false;
    blend1 = WHITE;
    blend2 = WHITE;
    partial = 1.0;

    //use 1 byte per pixel, 1 = white, 0 = black
    //allocate initial buffer 
    w = 1000;
    h = 1000;
    maxTurns = (1 << iters) - 1;
    //turnBuf = new unsigned char[maxTurns];
    turnBuf = (Turn*)calloc(maxTurns, sizeof(Turn));
    getPath();
    createImage();
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
