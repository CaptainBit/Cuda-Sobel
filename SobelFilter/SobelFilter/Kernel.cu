//Cuda file 
#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/types_c.h>  
#include <opencv2/imgproc/imgproc.hpp>  



using namespace cv; 

__global__ void Kernel_EraseBackground(unsigned char *MatA, unsigned char *MatR, int rows, int cols, uchar b1, uchar b2, uchar b3, bool fChange, uchar f1, uchar f2, uchar f3, int min, int max);
__global__ void Kernel_SobelFilter(unsigned char *MatA, unsigned char *MatR, int rows, int cols, double mult);

__device__  double Min(double red, double blue, double green); 
__device__  double Max(double red, double blue, double green);
__device__ int sobel(int a, int b, int c, int d, int e, int f);

int iDivUp(int a, int b);
int iAlignUp(int a, int b);

extern "C" void ApplySobelFilter(Mat *ptMatA, Mat *ptMatR, double mult)
{
	cudaError_t error;
	//pointeurs des matrices 
	uchar *MatA, *MatR;
	//Dimension de la grid et des blocs 
	dim3 nbreThreadsParBlock(32, 32);
	dim3 nbreBloc(iDivUp(ptMatA->cols, 32), iDivUp(ptMatA->rows, 32));

	//Allouer espace pour le gpu 
	int memSize = ptMatA->rows * ptMatA->step1();

	cudaMalloc((void **)&MatA, memSize);
	cudaMalloc((void **)&MatR, memSize);

	//Envoyer matrice dans la mémoire du gpu 
	cudaMemcpy(MatA, ptMatA->data, memSize, cudaMemcpyHostToDevice);
	Kernel_SobelFilter << <nbreBloc, nbreThreadsParBlock >> >(MatA, MatR, ptMatA->step1(), ptMatA->rows, mult);
	//Wait the Kernel to be done
	cudaDeviceSynchronize();
	//Retourner la matrice résultante 
	cudaMemcpy(ptMatR->data, MatR, memSize, cudaMemcpyDeviceToHost);
	error = cudaFree(MatA);
	error = cudaFree(MatR);
}
extern "C" void EraseBackground(Mat *ptMatA, Mat *ptMatR, uchar backgroundColor[], uchar foregroundColor[], int threshMin, int threshMax)
{ 
	cudaError_t error;
	//pointeurs des matrices 
	uchar *MatA, *MatR;
	//Dimension de la grid et des blocs 
	dim3 nbreThreadsParBlock(32, 32);
	dim3 nbreBloc(iDivUp(ptMatA->cols, 32), iDivUp(ptMatA->rows, 32));

	//Allouer espace pour le gpu 
	int memSize = ptMatA->rows * ptMatA->step1();

	cudaMalloc((void **)&MatA, memSize);
	cudaMalloc((void **)&MatR, memSize);

	//Envoyer matrice dans la mémoire du gpu 
	cudaMemcpy(MatA, ptMatA->data, memSize, cudaMemcpyHostToDevice);
	if (foregroundColor != NULL) 
	{
		Kernel_EraseBackground << <nbreBloc, nbreThreadsParBlock >> >(MatA, MatR, ptMatA->step1(), ptMatA->rows, backgroundColor[0], backgroundColor[1], backgroundColor[2],
			true, foregroundColor[0], foregroundColor[1], foregroundColor[2], threshMin, threshMax);
	}
	else 
	{
		Kernel_EraseBackground << <nbreBloc, nbreThreadsParBlock >> >(MatA, MatR, ptMatA->step1(), ptMatA->rows, backgroundColor[0], backgroundColor[1], backgroundColor[2],
			false, 0, 0, 0, threshMin, threshMax);

	}
	
	//Wait the Kernel to be done
	cudaDeviceSynchronize();
	//Retourner la matrice résultante 
	cudaMemcpy(ptMatR->data, MatR, memSize, cudaMemcpyDeviceToHost);
	error = cudaFree(MatA);
	error = cudaFree(MatR);
	
} 
 
__global__ void Kernel_SobelFilter(unsigned char *MatA, unsigned char *MatR, int rows, int cols, double mult)
{
	//GradiantX
	int kernelx[3][3] = { { -1, 0, 1 },
							{ -2, 0, 2 },
							{ -1, 0, 1 } };

	//GradiantY
	int kernely[3][3] = { { -1, -2, -1 },
							{ 0,  0,  0 },
							{ 1,  2,  1 } };
	//X et Y dans la matrice 
	int ImgNumColonne = (blockIdx.x  * blockDim.x) + threadIdx.x;
	int ImgNumLigne = (blockIdx.y * blockDim.y) + threadIdx.y;
	int Index = (ImgNumLigne * rows) + (ImgNumColonne * 3);

	//Ne depasse pas l'accès de la matrice
	if ((ImgNumColonne < (rows) - 1) && (ImgNumLigne < (cols) -2)) 
	{
		
		//Emplacement dans la mémoire
		int x1 = Index;
		int x2 = Index + 1;
		int x3 = Index + 2;
		int x4 = ((ImgNumLigne + 1) * rows) + ((ImgNumColonne) * 3);
		int x5 = ((ImgNumLigne + 1) * rows) + ((ImgNumColonne ) * 3) + 1;
		int x6 = ((ImgNumLigne + 1) * rows) + ((ImgNumColonne)* 3) + 2;
		int x7 = ((ImgNumLigne + 2) * rows) + ((ImgNumColonne) * 3);
		int x8 = ((ImgNumLigne + 2) * rows) + ((ImgNumColonne)* 3) + 1;
		int x9 = ((ImgNumLigne + 2) * rows) + ((ImgNumColonne) * 3) + 2;

		int magX = (kernelx[0][0] * MatA[x1]) + (kernelx[0][1] * MatA[x2]) + (kernelx[0][2] * MatA[x3]) +
			(kernelx[1][0] * MatA[x4]) + (kernelx[1][1] * MatA[x5]) + (kernelx[1][2] * MatA[x6]) +
			(kernelx[2][0] * MatA[x7]) + (kernelx[2][1] * MatA[x8]) + (kernelx[2][2] * MatA[x9]);

		int magY = (kernely[0][0] * MatA[x1]) + (kernely[0][1] * MatA[x2]) + (kernely[0][2] * MatA[x3]) +
			(kernely[1][0] * MatA[x4]) + (kernely[1][1] * MatA[x5]) + (kernely[1][2] * MatA[x6]) +
			(kernely[2][0] * MatA[x7]) + (kernely[2][1] * MatA[x8]) + (kernely[2][2] * (MatA[x9]));

		

		int magT = hypotf(magX, magY)*mult;
		MatR[x5] = (uchar)magT;
	}

}
__global__ void Kernel_EraseBackground(unsigned char *MatA, unsigned char *MatR, int rows, int cols, uchar b1, uchar b2, uchar b3,bool fChange, uchar f1, uchar f2, uchar f3, int min, int max)
{ 
	//X et Y dans la matrice 
	int ImgNumColonne = (blockIdx.x  * blockDim.x) + threadIdx.x; 
	int ImgNumLigne = (blockIdx.y * blockDim.y) + threadIdx.y; 
	int Index = (ImgNumLigne * rows)  + (ImgNumColonne * 3);

	if ((ImgNumColonne < rows / 3) && (ImgNumLigne < cols))
	{
		double blue = (double)MatA[Index] / 255;
		double green = (double)MatA[Index + 1] / 255;
		double red = (double)MatA[Index + 2] / 255;

		double cMax = Max(red, blue, green);

		double cMin = Min(red, blue, green);

		double delta = cMax - cMin;

		//	HUE
		double h = 0;
		if (blue == cMax) {
			h = 60 * ((red - green) / delta + 4);
		}
		else if (green == cMax) {
			h = 60 * ((blue - red) / delta + 2);
		}
		else if (red == cMax) {
			h = 60 * ((green - blue) / delta);
			if (h < 0)
				h += 360;
		}

		//	SATURATION
		double saturation = 0;
		if (cMax != 0) {
			saturation = delta / cMax;
		}

		//	VALUE
		double value = cMax;

		if (h / 2 > min && h / 2 < max)
		{
			MatR[Index] = b1;
			MatR[Index + 1] = b2;
			MatR[Index + 2] = b3;
		}
		else 
		{
			if (fChange == true)
			{
				MatR[Index] = f1;
				MatR[Index + 1] = f2;
				MatR[Index + 2] = f3;
			}
			else 
			{
				MatR[Index] = h/2;
				MatR[Index + 1] = saturation*255;
				MatR[Index + 2] = value*255;
			}
		}
	}

	
	
	return; 
 
} 

__device__  double Min(double red, double blue, double green)
{ 
	if ((blue <= green) && (blue <= red))
		return blue;
	else if ((green <= blue) && (green <= red))
		return green;
	else
		return red;
} 
__device__  double Max(double red, double blue, double green)
{ 
	if ((blue >= green) && (blue >= red))
		return blue;
	else if ((green >= blue) && (green >= red))
		return green;
	else
		return red;
} 

__device__ int sobel(int a, int b, int c, int d, int e, int f) {
	return ((a + 2 * b + c) - (d + 2 * e + f));
}

int iDivUp(int a, int b) // Round a / b to nearest higher integer value

{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
int iAlignUp(int a, int b) // Align a to nearest higher multiple of b

{
	return (a % b != 0) ? (a - a % b + b) : a;
}

 
 