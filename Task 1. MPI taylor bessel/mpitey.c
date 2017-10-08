#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <cstdlib>

#define MPI_ROOT_PROCESS 0

double factorial(double f) {
	if(f == 0)
		return 1;
	return (f * factorial(f-1));
}

double GetApproxValue(double x, double Ac)
{
	double sum = 0;
	double k;

    	for (k = 0; k < Ac; k++)
	{
		sum = sum + (pow(-1.0, k) / (pow(4.0, k) * pow(factorial(k), 2.0))) * pow(x, 2.0 * k);
	}
	return sum;
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
		double A = 0; //начало интервала
		double B = 1; //конец интервала
		double Ac = 20; //число членов ряда Тейлора
		double step = 0.00001; //шаг по х  0.1


		int N = ((B-A) / step) + 1 + 0.5; //Количество точек (+0.5 решает проблему округления до int)
	
		//Создаём массив точек, для которых будем считать функцию
		double *pointVector = new double[N];
		//double step = (B-A)/(N-1);
		double cv = A;
		for (int i=1;i<N;i++)
		{
			pointVector[i-1] = cv;
			cv += step;
		}
		//Последняя точка может не совпадать с B, поэтому добавляем её отдельно
		pointVector[N-1] = B;

		//Теперь начинаем рассылку сообщений
		//Отправляем эпсилон
		MPI_Bcast(&Ac,1,MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);

		double *approxValues = new double[N]; // приближённые значения

		int sendCount = N / procCount;
		//Если количество процессов кратно числу точек, то всё хорошо
		if (N % procCount == 0)
		{ 
			//Отправляем процессам сообщение с количеством точек, для которых необходимо рассчитать значение функции
			MPI_Bcast(&sendCount,1,MPI_INT,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
			//Отправляем сами точки
			MPI_Scatter(pointVector,sendCount,MPI_DOUBLE,pointVector,sendCount,MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
			//Считаем функцию в своих точках
			for (int i = 0; i<sendCount; i++)
			{
				approxValues[i] = GetApproxValue(pointVector[i],Ac);
			}

			//Теперь принимаем данные
			MPI_Gather(approxValues,sendCount,MPI_DOUBLE,approxValues,sendCount,MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
		}
		//Иначе надо определить сколько кому отправлять
		else
		{
			//Отправляем сообщение другим процессам готовиться к векторному приёму/передаче
			{
				int tmp = -1;
				MPI_Bcast(&tmp,1,MPI_INT,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
			}
			//Формируем массив смещений и массив с числом элементов (смещения считаются в элементах от начала массива)
			int *sendArray = new int[procCount];
			int *offsetArray = new int[procCount];
			for (int i = 0; i<procCount; i++)
			{
				sendArray[i] = sendCount;
			}
			for (int i = 0; i<N%procCount; i++)
				sendArray[i]++;
			offsetArray[0]=0;
			for (int i=1; i<procCount; i++)
			{
				offsetArray[i]= sendArray[i-1]+offsetArray[i-1];
			}
			//Отправляем количество элементов для получения
			MPI_Scatter(sendArray,1,MPI_INT,&sendArray[0],1,MPI_INT,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
			//Отправляем данные
			MPI_Scatterv(pointVector,sendArray,offsetArray,MPI_DOUBLE,pointVector,sendArray[0],MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
			for (int i = 0;i<sendArray[0];i++)
			{
				approxValues[i] = GetApproxValue(pointVector[i],Ac);
			}
			//Теперь принимаем данные
			MPI_Gatherv(approxValues,sendArray[0],MPI_DOUBLE,approxValues,sendArray,offsetArray,MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
		}
		//Выводим значения (точность - 10 знаков, выводить числа без отбрасывания нулей, показывать знак для положительных)
		std::cout.precision(10);
		std::cout.setf(std::ios::fixed | std::ios::showpos);
		std::cout<<"x         y(x)\n";
		for (int i = 0; i<N;i++)
			std::cout<<pointVector[i]<< ' '<<approxValues[i]<<'\n';
	}
	else
	{
		//Получаем эпсилон
		double Ac;
		MPI_Bcast(&Ac,1,MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
		//Получаем количество точек
		int pointCount;
		MPI_Bcast(&pointCount,1,MPI_INT,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
		//Пришло положительное, значит число точек для всех процессов одинаковое и мы его получили
		if (pointCount>0)
		{
			double *points = new double[pointCount];
			//Получили точки
			MPI_Scatter(0,0,MPI_DOUBLE,points,pointCount,MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
			//Считаем функцию
			double *approxValues = new double[pointCount];
			for (int i = 0; i<pointCount; i++)
			{
				approxValues[i] = GetApproxValue(points[i],Ac);
			}
			//Отправляем
			MPI_Gather(approxValues,pointCount,MPI_DOUBLE,0,0,MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
		}
		else
		//Если пришло отрицательное значение, значит число точек будет получено через scatter
		{
			//Получаем число точек
			MPI_Scatter(0,0,MPI_DOUBLE,&pointCount,1,MPI_INT,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
			//Получаем сами точки
			double *points = new double[pointCount];
			MPI_Scatterv(0,0,0,MPI_DOUBLE,points,pointCount,MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
			//Считаем функцию
			double *approxValues = new double[pointCount];
			for (int i = 0; i<pointCount; i++)
			{
				approxValues[i] = GetApproxValue(points[i],Ac);
			}
			//Отправляем
			MPI_Gatherv(approxValues,pointCount,MPI_DOUBLE,0,0,0,MPI_DOUBLE,MPI_ROOT_PROCESS,MPI_COMM_WORLD);
		}
	}
	MPI_Finalize();
	return 0;
}

//mpiCC mpitey.c -lm && mpiexec -n 11 ./a.out

