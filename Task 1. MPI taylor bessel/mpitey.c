#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <cstdlib>

#define MPI_ROOT_PROCESS 0	//id главного процесса
#define A 0 				//начало интервала
#define B 1					//конец интервала
#define step 0.1			//шаг по х  0.1 0.00001; 

double factorial(double f) {
	if(f == 0)
		return 1;
	return (f * factorial(f-1));
}

double BesselJ0(double x)
{
	double k = 0;
	double prevSum = 0;
	double currSum = 0;
	do
	{
		prevSum = currSum;
		currSum = currSum + (pow(-1.0, k) / (pow(4.0, k) * pow(factorial(k), 2.0))) * pow(x, 2.0 * k);
		k++;
	}while(fabs(currSum - prevSum) > 0);
	return currSum;
}

void ResultToConsole(double *pointVector, double *approxValues, int N)
{
	//Выводим значения (точность - 10 знаков, выводить числа без отбрасывания нулей, показывать знак для положительных)
	std::cout.precision(10);
	std::cout.setf(std::ios::fixed | std::ios::showpos);
	std::cout<<"x         y(x)\n";
	for (int i = 0; i<N;i++)
	{
		std::cout<<pointVector[i]<< ' '<<approxValues[i]<<'\n';
	}
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc,&argv);
	//Получаем число процессов
	int procCount;
	MPI_Comm_size(MPI_COMM_WORLD,&procCount);
	//Получаем ранг процесса
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	//Если процесс главный
	if (rank == MPI_ROOT_PROCESS)
	{
		//Количество точек всего (+0.5 решает проблему округления до int)
		int N = ((B-A) / step) + 1 + 0.5; 
	
		//Массив со всеми точками от A до B
		double *pointVector = new double[N];
	
		double cv = A;
		for (int i=1;i<N;i++)
		{
			pointVector[i-1] = cv;
			cv += step;
		}
		pointVector[N-1] = B;

		//Отправляем количество точек всего
		MPI_Bcast(&N,1,MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);

		//Массив со всеми результатами вычислений
		double *approxValues = new double[N]; 

		//Посчитаем число точек для одного процесса
		int sendCount = (N / procCount) + 0.5; //(+0.5 решает проблему округления до int)

		//std::cout << "N: " << N << std::endl;

		//Вычислим, сколько элементов массива аргументов будет рассчитываться в каждом процессе
		int *sendArray = new int[procCount];
		for (int i = 0; i < procCount; i++)
		{
			sendArray[i] = sendCount;
			//Если нацело не делится, то прибавляем по единице к нескольким в начале
			if(i < N % procCount)
			{
				sendArray[i]++;
			}
			//std::cout << "sendArray: " << sendArray[i] << std::endl;
		}
			
		//Вычислим отступ от начала массива аргументов для каждого процесса
		int *offsetArray = new int[procCount];
		offsetArray[0] = 0; //для начального процесса отсчет sendArray[0] точек начинается с 0
		//для остальных процессов отсчет sendArray[i] точек начинается со следующей после той, 
		//на которой закончил отсчет предыдущий процесс
		for (int i = 1; i < procCount; i++)
		{
			offsetArray[i] = sendArray[i-1] + offsetArray[i-1];
			//std::cout << "offsetArray: " << offsetArray[i] << std::endl;
		}

		//Отправляем количество элементов каждому процессу
		MPI_Scatter(sendArray,1,MPI_INT,&sendArray[0],1,MPI_INT,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
		//Отправляем отступ от начала каждому процессу
		MPI_Scatter(offsetArray,1,MPI_INT,&offsetArray[0],1,MPI_INT,MPI_ROOT_PROCESS,MPI_COMM_WORLD);

		/////////////////Тем временем ROOT PROCESS тоже считает свою часть/////////////////////////////

		int pointCount = sendArray[0];
		int offsetVal = offsetArray[0];

		//Массив со всеми точками от offsetVal до offsetVal + pointCount
		double *currPointVector = new double[pointCount];
		
		double cv2 = A + offsetVal * step;
		//std::cout << rank << " rank, CV: " << cv << std::endl;
		for (int i = 1; i < pointCount + 1; i++)
		{	
			currPointVector[i - 1] = cv2;
			//std::cout << rank << " rank, pv: " << currPointVector[i - 1] << std::endl;
			cv2 += step;
		}

		//Считаем функцию
		for (int i = 0; i < pointCount; i++)
		{
			//std::cout << currPointVector[i] << std::endl;
			approxValues[i + offsetVal] = BesselJ0(currPointVector[i + offsetVal]);
		}
		//////////////////////////////////////////////////////

		//Теперь принимаем данные
		MPI_Gather(approxValues,sendArray[0],MPI_DOUBLE,approxValues,sendArray[0],MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
		//Выводим всё в консоль	
		//ResultToConsole(pointVector, approxValues, N);
	}
	//rank != ROOT_PROCESS
	else
	{
		//Получаем общее количество точек
		double N;
		MPI_Bcast(&N,1,MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
		//Получаем количество точек
		int pointCount;
		int offsetVal;
		
			//Получаем число точек
			MPI_Scatter(0,0,MPI_DOUBLE,&pointCount,1,MPI_INT,MPI_ROOT_PROCESS,MPI_COMM_WORLD);

			//std::cout << rank << "pcount " << pointCount << std::endl;
			//Получаем значение отступа
			MPI_Scatter(0,0,MPI_DOUBLE,&offsetVal,1,MPI_INT,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
			//std::cout << rank << "offsetVal: " << offsetVal << std::endl;

			//Массив со всеми точками от offsetVal до offsetVal + pointCount
			double *currPointVector = new double[pointCount];
		
			double cv = A + offsetVal * step;
			//std::cout << rank << " rank, CV: " << cv << std::endl;
			for (int i = 1; i < pointCount + 1; i++)
			{
				
				currPointVector[i - 1] = cv;
				//std::cout << rank << " rank, pv: " << currPointVector[i - 1] << std::endl;
				cv += step;
				
			}

			//Считаем функцию
			double *approxValues = new double[pointCount];
			for (int i = 0; i<pointCount; i++)
			{
				//std::cout << rank << " rank "<< currPointVector[i] << std::endl;
				approxValues[i] = BesselJ0(currPointVector[i]);
				std::cout << rank << " rank, approxValue:  "<< approxValues[i] << std::endl;
			}
			//Отправляем
			MPI_Gather(approxValues,pointCount,MPI_DOUBLE,0,0,MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}
//mpiCC mpitey.c -lm && mpiexec -n 11 ./a.out
