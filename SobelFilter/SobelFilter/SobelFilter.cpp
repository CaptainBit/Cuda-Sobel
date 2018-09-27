#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 
#include "stdafx.h" 
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/types_c.h>  
#include <opencv2/imgproc/imgproc.hpp>
#include "AxisCommunication.h"

using namespace cv;
using namespace std;

extern "C" void EraseBackground(Mat *ptMatA, Mat *ptMatR, uchar backgroundColor[], uchar foregroundColor[], int threshMin, int threshMax);
extern "C" void ApplySobelFilter(Mat *ptMatA, Mat *ptMatR, double mult);
void changeSobel(int i, void*);

Mat gray_scale, gray_sobel;
int main()
{
	int mult;
	//uchar backgroundColor[3] = {255,255,255};
	//Mat Image, imgResult;
	//Axis axis("10.128.3.4", "etudiant", "gty970");
	//while (true) 
	//{
		//axis.GetImage(Image);
		//imgResult = Image;

		//imshow("original", Image);

		//EraseBackground(&Image, &imgResult, backgroundColor, NULL, 75 / 2, 150 / 2);

		//Show result of the treshold
		//imshow("noBackground", imgResult);

		//Mat for gray_scale


		//Conversion sur 1 octet
		//cvtColor(imgResult, gray_scale, COLOR_BGR2GRAY);
		//cvtColor(imgResult, gray_sobel, COLOR_BGR2GRAY);

		//Appliquer filtre sobel
		gray_scale = imread("Ny.jpg", 0);
		gray_sobel = imread("Ny.jpg", 0);
		ApplySobelFilter(&gray_scale, &gray_sobel, 0.0625);
		imwrite("Nysobel.jpg", gray_sobel);
		//Show resutl
		imshow("Sobel", gray_sobel);

		//if we want to use the trackbar(doesnt work in a loop)
		createTrackbar("Mult:", "Sobel", &mult, 100, changeSobel);
		waitKey(0);
	//}
	//axis.ReleaseCam();
	
	return 0;
}
void changeSobel(int i, void*)
{

	ApplySobelFilter(&gray_scale, &gray_sobel, (double)i/100);
	imshow("Sobel", gray_sobel);
}