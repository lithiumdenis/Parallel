#include <cstdlib>
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "stdbool.h"
#include <cuda.h>
#include "lodepng.h"
#include <vector>

#include <iostream>

using namespace std;

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

int *CopyArrayToGPU(int *HostArray, int NumElements)
{
    int bytes = sizeof(int) * NumElements;
    int *DeviceArray;

    // Allocate memory on the GPU for array
    if (cudaMalloc(&DeviceArray, bytes) != cudaSuccess)
    {
        printf("CopyArrayToGPU(): Couldn't allocate mem for array on GPU.");
        return NULL;
    }

    // Copy the contents of the host array to the GPU
    if (cudaMemcpy(DeviceArray, HostArray, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("CopyArrayToGPU(): Couldn't copy host array to GPU.");
        cudaFree(DeviceArray);
        return NULL;
    }

    return DeviceArray;
}

/*
//precondition: turnBuf is populated
void createImage(int maxTurns, unsigned char* turnBuf, unsigned* pic, int x, int y, int iw, int ih)
{
  int dir = D_UP;
  
  //fill image with BLACK (most pixels will be in final product)
  for(int i = 0; i < iw * ih; i++)
  {
	  pic[i] = BLACK;
  }

  for(int i = 0; i <= 30; i++)   //maxTurns
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
	



	//cout << i << " iter,dir  is " << dir << endl;
  }


  


  
}




*/










static __global__
void FractalKernel(unsigned char *turnBuf, unsigned *pic, int maxTurns, int x, int y, int iw, int ih, int dir, int *d_dirs, int *d_x_arr, int *d_y_arr)
{
	/*for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < maxTurns; idx += blockDim.x * gridDim.x) 
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
		if(idx != maxTurns)
		{
		  if(turnBuf[idx] == LEFT)
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
		
		__syncthreads();
		
		
		
		
		
		
		
		
		//y[i] = a * x[i] + y[i];
	}

	*/
	
















    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	//printf("ih = %d", ih);

    if (idx < maxTurns) { 

		//const int col = idx % width;  //столбец
        //const int row = (idx / width) % width; //строка

		/*if(idx % 2 == 0)
		{
			pic[(d_x_arr[idx]) + d_y_arr[idx] * iw] = PURPLE;
		}*/

		
			//pic[idx] = PURPLE;
		

		//int dir = D_UP;
		
		
	  
		//for(int i = 0; i <= maxTurns; i++)
		//{
		  //move in current direction, writing pixels on segment to color
		  //pic[x + y * iw] = PURPLE;
		  /*for(int idx = 0; idx < maxTurns; idx++)
		  {
			switch(d_dirs[idx])
		  	{
			case D_LEFT:
			  pic[(d_x_arr[idx] - 1) + d_y_arr[idx] * iw] = PURPLE;
			  //x -= (false ? 1 : 2);
			  break;
			case D_UP:
			  //pic[x + (y - 1) * iw] = PURPLE;
			  pic[(d_x_arr[idx]) + (d_y_arr[idx] - 1) * iw] = PURPLE;
			  //y -= (false ? 1 : 2);
			  break;
			case D_RIGHT:
			  //pic[(x + 1) + y * iw] = PURPLE;
			  pic[(d_x_arr[idx] + 1) + d_y_arr[idx] * iw] = PURPLE;
			  //x += (false ? 1 : 2);
			  break;
			case D_DOWN:
			  pic[(d_x_arr[idx]) + (d_y_arr[idx] + 1) * iw] = PURPLE;
			  //pic[x + (y + 1) * iw] = PURPLE;
			  //y += (false ? 1 : 2);
			  break;
			default:;
		  	}
		  }*/

		  //for (int idx = 0; idx < maxTurns; idx++)
		  //{
			  switch((int)d_dirs[idx])
			  {
					case D_LEFT:
					  pic[((int)d_x_arr[idx] - 1) + (int)d_y_arr[idx] * iw] = PURPLE;
					  //x -= (false ? 1 : 2);
					  break;
					case D_UP:
					  //pic[x + (y - 1) * iw] = PURPLE;
					  pic[((int)d_x_arr[idx]) + ((int)d_y_arr[idx] - 1) * iw] = PURPLE;
					  //y -= (false ? 1 : 2);
					  break;
					case D_RIGHT:
					  //pic[(x + 1) + y * iw] = PURPLE;
					  pic[((int)d_x_arr[idx] + 1) + (int)d_y_arr[idx] * iw] = PURPLE;
					  //x += (false ? 1 : 2);
					  break;
					case D_DOWN:
					  pic[((int)d_x_arr[idx]) + ((int)d_y_arr[idx] + 1) * iw] = PURPLE;
					  //pic[x + (y + 1) * iw] = PURPLE;
					  //y += (false ? 1 : 2);
					  break;
					default:;
			  }
		  //}

		  /*
		  if((int)d_dirs[idx] == D_LEFT)
		  {
			pic[((int)d_x_arr[idx] - 1) + (int)d_y_arr[idx] * iw] = PURPLE;
		  }
		  else if ((int)d_dirs[idx] == D_UP)
		  {
			pic[((int)d_x_arr[idx]) + ((int)d_y_arr[idx] - 1) * iw] = PURPLE;
		  }
		  else if ((int)d_dirs[idx] == D_RIGHT)
		  {
			pic[((int)d_x_arr[idx] + 1) + (int)d_y_arr[idx] * iw] = PURPLE;
		  }
		  else if((int)d_dirs[idx] == D_DOWN)
		  {
			pic[((int)d_x_arr[idx]) + ((int)d_y_arr[idx] + 1) * iw] = PURPLE;
		  }
		  
*/





		  
		  /*{
			if(turnBuf[i] == LEFT)
			{
			  dir = (dir + 1) % 4;
			}
			else
			{
			  dir = (dir + 3) % 4;
			}
		  }
		  else*/
		  
		//}
        
        



		
		/*

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
		
		*/


        

    }
}


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
	dir = D_UP;


	//create arrays of dots
	int *h_dirs = new int[maxTurns];
	int *x_arr = new int[maxTurns];
	int *y_arr = new int[maxTurns];
	//h_dirs[0] = D_UP;
	//x_arr[0] = x;
	//y_arr[0] = y;


	//vector aaa = new vector[maxTurns];

	dir = D_UP;
	
	for(int i = 0; i < maxTurns; i++)
	{
		y_arr[i] = y;
		x_arr[i] = x; 
		h_dirs[i] = dir; 




	  //move in current direction, writing pixels on segment to color
	  //pic[x + y * iw] = PURPLE;
	  switch(dir)
	  {
		case D_LEFT:
		  //pic[(x - 1) + y * iw] = PURPLE;
		  x -= (false ? 1 : 2);
		  //x_arr[i + 1] = x; 
		  break;
		case D_UP:
		  //pic[x + (y - 1) * iw] = PURPLE;
		  y -= (false ? 1 : 2);
		  //y_arr[i + 1] = y; 
		  break;
		case D_RIGHT:
		  //pic[(x + 1) + y * iw] = PURPLE;
		  x += (false ? 1 : 2);
		  //x_arr[i + 1] = x; 
		  break;
		case D_DOWN:
		  //pic[x + (y + 1) * iw] = PURPLE;
		  y += (false ? 1 : 2);
		  //y_arr[i + 1] = y; 
		  break;
		default:;
	  }
	  //turn (except after last position update)

	  if(i != maxTurns)
	  {
		if(turnBuf[i] == LEFT)
		{
			dir = (dir + 1) % 4;
			//h_dirs[i + 1] = dir;
		}
		else
		{
			dir = (dir + 3) % 4;
			//h_dirs[i + 1] = dir;
		}
	  }
	  else
	  {
		  //pic[x + y * iw] = PURPLE;
	  }


	  //cout << "x " << i << " is " << h_dirs[i] << endl;
	}

	/*cout << "maxTurns is " << maxTurns << endl;
	for(int i = 0; i < maxTurns; i++)
	{
		cout << "x " << i << " is " << x << endl;
	}*/

	/////////////////////////////////////////

	//cout << "dddvev" << endl;




	int *d_dirs = CopyArrayToGPU(h_dirs, maxTurns);
	int *d_x_arr = CopyArrayToGPU(x_arr, maxTurns);
	int *d_y_arr = CopyArrayToGPU(y_arr, maxTurns);














	/*
	//allocate memory for d_dirs on device
	unsigned *d_dirs = NULL;
    if (cudaSuccess != cudaMalloc((void **)&d_dirs, maxTurns))
    {
        fprintf(stderr, "Failed to allocate device vector d_dirs\n");
        exit(EXIT_FAILURE);
	}
	 
	//Копирование h_dirs из хоста на девайс
	cudaError_t err = cudaMemcpy(d_dirs, h_dirs, maxTurns, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		 fprintf(stderr, "Failed to copy vector h_dirs from host to device (error code %s)!\n", cudaGetErrorString(err));
		 exit(EXIT_FAILURE);
	}
	 
	//allocate memory for d_x_arr on device
	unsigned *d_x_arr = NULL;
    if (cudaSuccess != cudaMalloc((void **)&d_x_arr, maxTurns))
    {
        fprintf(stderr, "Failed to allocate device vector d_x_arr\n");
        exit(EXIT_FAILURE);
	}
	 
	//Копирование x_arr из хоста на девайс
	err = cudaMemcpy(d_x_arr, x_arr, maxTurns, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		 fprintf(stderr, "Failed to copy vector x_arr from host to device (error code %s)!\n", cudaGetErrorString(err));
		 exit(EXIT_FAILURE);
	}
	 
	//allocate memory for d_y_arr on device
	unsigned *d_y_arr = NULL;
    if (cudaSuccess != cudaMalloc((void **)&d_y_arr, maxTurns))
    {
        fprintf(stderr, "Failed to allocate device vector d_y_arr\n");
        exit(EXIT_FAILURE);
	}
	 
	//Копирование y_arr из хоста на девайс
	err = cudaMemcpy(d_y_arr, y_arr, maxTurns, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		 fprintf(stderr, "Failed to copy vector y_arr from host to device (error code %s)!\n", cudaGetErrorString(err));
		 exit(EXIT_FAILURE);
 	}

*/

	/////////////////////////////////////////

	






















	//pic on host
	unsigned* pic = (unsigned*)malloc(iw * ih * sizeof(unsigned));
	
	//fill image with BLACK (most pixels will be in final product)
	for(int i = 0; i < iw * ih; i++)
	{
		pic[i] = BLACK;
	}

	//allocate memory for cuda pic on device
	unsigned* pic_d;
    if (cudaSuccess != cudaMalloc((void **)&pic_d, iw * ih * sizeof(unsigned))) 
    {
        fprintf(stderr, "could not allocate memory\n"); 
        exit(-1);
	}

	//Копирование pic из хоста на девайс, т.е. из pic в pic_d
	cudaError_t err = cudaMemcpy(pic_d, pic, iw * ih * sizeof(unsigned), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	 {
		 fprintf(stderr, "Failed to copy pic from host to device (error code %s)!\n", cudaGetErrorString(err));
		 exit(EXIT_FAILURE);
 	}


	//allocate memory for turnBuf on device
	unsigned char *d_turnbuf = NULL;
    if (cudaSuccess != cudaMalloc((void **)&d_turnbuf, maxTurns))
    {
        fprintf(stderr, "Failed to allocate device vector d_turnbuf\n");
        exit(EXIT_FAILURE);
	}
	 
	//Копирование turnBuf из хоста на девайс, т.е. из turnBuf в d_turnBuf
	//printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_turnbuf, turnBuf, maxTurns, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	 {
		 fprintf(stderr, "Failed to copy vector turnBuf from host to device (error code %s)!\n", cudaGetErrorString(err));
		 exit(EXIT_FAILURE);
 	}


	
	// compute
    FractalKernel<<<(iw * ih + (ThreadsPerBlock - 1)) / ThreadsPerBlock, ThreadsPerBlock>>>(d_turnbuf, pic_d, maxTurns, x, y, iw, ih, dir, d_dirs, d_x_arr, d_y_arr);

	//Return from cuda
	if (cudaSuccess != cudaMemcpy(pic, pic_d, iw * ih * sizeof(unsigned), cudaMemcpyDeviceToHost)) 
    {
        fprintf(stderr, "copying from device failed\n"); 
        exit(-1);
	}
	

	/*
	 //it WORKS!!!!!!!!!!!!!!
	for (int idx = 0; idx < maxTurns; idx++)
	{
		switch((int)h_dirs[idx])
		{
	  		case D_LEFT:
				pic[((int)x_arr[idx] - 1) + (int)y_arr[idx] * iw] = PURPLE;
				//x -= (false ? 1 : 2);
				break;
	  		case D_UP:
				//pic[x + (y - 1) * iw] = PURPLE;
				pic[((int)x_arr[idx]) + ((int)y_arr[idx] - 1) * iw] = PURPLE;
				//y -= (false ? 1 : 2);
				break;
	  		case D_RIGHT:
				//pic[(x + 1) + y * iw] = PURPLE;
				pic[((int)x_arr[idx] + 1) + (int)y_arr[idx] * iw] = PURPLE;
				//x += (false ? 1 : 2);
				break;
	  		case D_DOWN:
				pic[((int)x_arr[idx]) + ((int)y_arr[idx] + 1) * iw] = PURPLE;
				//pic[x + (y + 1) * iw] = PURPLE;
				//y += (false ? 1 : 2);
				break;
	  		default:;
		}
	}

	
	*/



    



    



    //createImage(maxTurns, turnBuf, pic, x, y, iw, ih);



    //get image filename
    char fname[64];
    sprintf(fname, "dragon.png");
    lodepng_encode32_file(fname, (unsigned char*) pic, iw, ih);
	free(pic);
	cudaFree(pic_d);



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

// make all
// ./dragon
