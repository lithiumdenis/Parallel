#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "stdbool.h"
#include <cuda.h>
#include "lodepng.h"
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

//Копирование массива int типа в GPU функцией
int *CopyArrayToGPU(int *HostArray, int NumElements)
{
	int bytes = sizeof(int) * NumElements;
	int *DeviceArray;

	//Выделение памяти на GPU
	if (cudaMalloc(&DeviceArray, bytes) != cudaSuccess)
	{
		printf("CopyArrayToGPU(): Couldn't allocate mem for array on GPU.");
		return NULL;
	}

	//Копирование с хоста на девайс
	if (cudaMemcpy(DeviceArray, HostArray, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("CopyArrayToGPU(): Couldn't copy host array to GPU.");
		cudaFree(DeviceArray);
		return NULL;
	}
	return DeviceArray;
}

//Кернел для построения изображения с помощью CUDA
static __global__ void FractalKernel(unsigned *pic, int maxTurns, int iw, int *d_dirs, int *d_x_arr, int *d_y_arr)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < maxTurns)
	{
		switch (d_dirs[idx])
		{
		case D_LEFT:
			pic[(d_x_arr[idx] - 1) + d_y_arr[idx] * iw] = PURPLE;
			break;
		case D_UP:
			pic[(d_x_arr[idx]) + (d_y_arr[idx] - 1) * iw] = PURPLE;
			break;
		case D_RIGHT:
			pic[(d_x_arr[idx] + 1) + d_y_arr[idx] * iw] = PURPLE;
			break;
		case D_DOWN:
			pic[(d_x_arr[idx]) + (d_y_arr[idx] + 1) * iw] = PURPLE;
			break;
		default:;
		}
	}
}

int main(int argc, char *argv[])
{
	//Получение путей
	int numTurns = 0;
	int maxTurns = (1 << iters) - 1;
	unsigned char *turnBuf = (unsigned char *)calloc(maxTurns, sizeof(unsigned char));

	for (int i = 0; i < iters; i++)
	{
		//append RIGHT to the existing sequence
		turnBuf[numTurns] = RIGHT;
		//append inverse transpose of first numTurns turns
		int src = numTurns - 1;
		int dst = numTurns + 1;
		while (src >= 0)
		{
			turnBuf[dst] = (turnBuf[src] == LEFT) ? RIGHT : LEFT;
			src--;
			dst++;
		}
		//Обновляем numTurns
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

	//Пройдемся по всем направлениям
	for (int i = 0; i <= maxTurns; i++)
	{
		switch (dir)
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
		//Обновим информацию о минимумах и максимумах
		minx = x < minx ? x : minx;
		miny = y < miny ? y : miny;
		maxx = x > maxx ? x : maxx;
		maxy = y > maxy ? y : maxy;
		//Подготовим данные для следующего шага
		if (i != maxTurns)
		{
			if (turnBuf[i] == LEFT)
			{
				dir = (dir + 1) % 4;
			}
			else
			{
				dir = (dir + 3) % 4;
			}
		}
	}

	//Задаём начальную точку так, чтобы остаться в [0, iw) x [0, ih) при построении
	x = (-minx) * (false ? 1 : 2) + 1;
	y = (-miny) * (false ? 1 : 2) + 1;

	//Получаем т.н. bounding box, внутри которого будет находиться 
	//вся наша фигура на весь размер
	int iw = (maxx - minx) * (false ? 1 : 2) + 3;
	int ih = (maxy - miny) * (false ? 1 : 2) + 3;
	dir = D_UP;

	//Создаём массивы для хранения направлений, х и у
	int *h_dirs = new int[maxTurns];
	int *x_arr = new int[maxTurns];
	int *y_arr = new int[maxTurns];

	for (int i = 0; i < maxTurns; i++)
	{
		//Сохраняем промежуточные данные в массивы
		y_arr[i] = y;
		x_arr[i] = x;
		h_dirs[i] = dir;

		switch (dir)
		{
		case D_LEFT:
			x -= (false ? 1 : 2);
			break;
		case D_UP:
			y -= (false ? 1 : 2);
			break;
		case D_RIGHT:
			x += (false ? 1 : 2);
			break;
		case D_DOWN:
			y += (false ? 1 : 2);
			break;
		default:;
		}
		//Получаем данные для следующей итерации
		if (i != maxTurns)
		{
			if (turnBuf[i] == LEFT)
			{
				dir = (dir + 1) % 4;
			}
			else
			{
				dir = (dir + 3) % 4;
			}
		}
	}

	//Копируем необходимые данные на девайс
	int *d_dirs = CopyArrayToGPU(h_dirs, maxTurns);
	int *d_x_arr = CopyArrayToGPU(x_arr, maxTurns);
	int *d_y_arr = CopyArrayToGPU(y_arr, maxTurns);

	//Создаем картинку на хосте
	unsigned *pic = (unsigned *)malloc(iw * ih * sizeof(unsigned));
	//Заполняем по умолчанию фон черным цветом
	for (int i = 0; i < iw * ih; i++)
	{
		pic[i] = BLACK;
	}

	//Выделение памяти для картинки pic на девайсе
	unsigned *pic_d;
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

	// Вычисление кернела
	FractalKernel<<<(iw * ih + (ThreadsPerBlock - 1)) / ThreadsPerBlock, ThreadsPerBlock>>>(pic_d, maxTurns, iw, d_dirs, d_x_arr, d_y_arr);

	// Возврат с девайса
	if (cudaSuccess != cudaMemcpy(pic, pic_d, iw * ih * sizeof(unsigned), cudaMemcpyDeviceToHost))
	{
		fprintf(stderr, "copying from device failed\n");
		exit(-1);
	}

	// Сохранение результата в файл
	char fname[64];
	sprintf(fname, "dragon.png");
	lodepng_encode32_file(fname, (unsigned char *)pic, iw, ih);
	free(pic);
	cudaFree(pic_d);
	free(turnBuf);
	return 0;
}
// make all
// ./dragon
