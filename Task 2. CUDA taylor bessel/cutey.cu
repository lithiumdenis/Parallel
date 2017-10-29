#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void besselKernel(const double *X, double *Y, int numElements)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < numElements)
    {
        double k = 0;
        double prevSum = 0;
        double currSum = 0;
        double factCurr = 0;

        do
        {
            prevSum = currSum;

            //iterative factorial
            if(k == 0)
            {
                factCurr = 1;
            }
            else
            {
                factCurr = 1;
                for (int j = 1; j <= k; j++)
                {
                    factCurr *= j;
                }  
            }
            currSum = currSum + (powf(-1.0, k) / (powf(4.0, k) * powf(factCurr, 2.0))) * powf(X[index], 2.0 * k);
            k++;
        }while(fabs(currSum - prevSum) > 0);
        Y[index] = currSum;
    }
}

int main(void)
{
    double A = 0; //начало интервала
    double B = 1; //конец интервала
    double step = 0.00001; //шаг по х 0.1 0.00001;

    //Количество точек всего (+0.5 решает проблему округления до int)
    int numElements = ((B-A) / step) + 1 + 0.5;

    printf("numElements = %d\n", numElements);

    size_t size = numElements * sizeof(double);
    
    // Выделение памяти на хосте для значений X, для кот. вычисляется функция 
    double *h_X = (double *)malloc(size);
    // Выделение памяти на хосте для значений Y, т.е. y(x), кот. будут возвращаться
    double *h_Y = (double *)malloc(size);

    // Проверка успешности выделения памяти на хосте
    if (h_X == NULL || h_Y == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    //Инициализация h_X значениями
    double cv = A;
    for (int i = 0; i < numElements; i++)
    {
        h_X[i] = cv;
        cv += step;
    }
    h_X[numElements - 1] = B;

    //Выделение памяти на девайсе (GPU)
    // Выделение памяти на девайсе для значений X, для кот. вычисляется функция
    double *d_X = NULL;
    cudaError_t err = cudaMalloc((void **)&d_X, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_X (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Выделение памяти на хосте для значений Y, т.е. y(x), кот. будут возвращаться
    double *d_Y = NULL;
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
//compile: nvcc cutey.cu -o cutey.out
//execution: ./cutey.out
