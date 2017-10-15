#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

float factorial(float f) {
	if(f == 0)
		return 1;
	return (f * factorial(f-1));
}

float BesselJ0(float x)
{
	float k = 0;
	float prevSum = 0;
	float currSum = 0;
	do
	{
		prevSum = currSum;
		currSum = currSum + (pow(-1.0, k) / (pow(4.0, k) * pow(factorial(k), 2.0))) * pow(x, 2.0 * k);
		k++;
	}while(fabs(currSum - prevSum) > 0);
	return currSum;
}

__global__ void
besselKernel(const float *X, float *Y, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        Y[i] = BesselJ0(X[i]);
    }
}

int
main(void)
{
    float A = 0; //начало интервала
    float B = 1; //конец интервала
    float step = 0.1; //шаг по х 0.1 0.00001;

    //Количество точек всего (+0.5 решает проблему округления до int)
    int numElements = ((B-A) / step) + 1 + 0.5;

    printf("numElements = %d\n", numElements);

    size_t size = numElements * sizeof(float);
    
    // Выделение памяти на хосте для значений X, для кот. вычисляется функция 
    float *h_X = (float *)malloc(size);
    // Выделение памяти на хосте для значений Y, т.е. y(x), кот. будут возвращаться
    float *h_Y = (float *)malloc(size);

    // Проверка успешности выделения памяти на хосте
    if (h_X == NULL || h_Y == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    //Инициализация h_X значениями
    float cv = A;
    for (int i = 0; i < numElements; i++)
    {
        h_X[i] = cv;
        cv += step;
    }
    h_X[numElements - 1] = B;

    //Выделение памяти на девайсе (GPU)
    // Выделение памяти на девайсе для значений X, для кот. вычисляется функция
    float *d_X = NULL;
    err = cudaMalloc((void **)&d_X, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_X (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Выделение памяти на хосте для значений Y, т.е. y(x), кот. будут возвращаться
    float *d_Y = NULL;
    err = cudaMalloc((void **)&d_Y, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_Y (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Копирование x из хоста на девайс, т.е. из h_X в d_X
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector h_X from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Запускаем Kernel CUDA
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    besselKernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, numElements);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch besselKernel kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Копируем результат вычислений обратно на хост с девайса
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_Y, d_Y, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_Y from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Выводим на хосте результат
    for (int i = 0; i < numElements; i++)
    {
        printf("x = %f, y(x) = %f\n", h_X[i], h_Y[i]);
    }

    //Очистили память
    //На девайсе
    err = cudaFree(d_X);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector X (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_Y);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector Y (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //На хосте
    free(h_X);
    free(h_Y);

    //Освободим девайс
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Execution success\n");
    return 0;
}
//nvcc cutey.cu -o -lm cutey.out