/////////////////////////////////////////////////////////////////////////////
//
// SOBEL header functions
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include<string>

using namespace std;
using namespace cv;



/** Function Headers */
void normaliseMatrix( Mat matrix );
void direction(cv::Mat &input, Mat_<int> kernel, cv::Mat &convo);
void magDir(cv::Mat &inputx, cv::Mat &inputy, cv::Mat &mag, cv::Mat &dir);

void sobel(Mat &frame_grey, Mat &xConvo, Mat &yConvo, Mat &mag, Mat &dir){
	// Create the SOBEL kernel in 1D, Y is transpose
	Mat_<int> kernel(3,3);
	Mat_<int>kernelT(3,3);
	kernel << -1, 0, 1, -2, 0, 2, -1, 0, 1;
	kernelT << 1, 2, 1, 0, 0, 0, -1, -2, -1;

	// Sobel filter
	direction(frame_grey,kernel, xConvo);
	direction(frame_grey, kernelT, yConvo);
	magDir( xConvo, yConvo, mag, dir);
	//imwrite( "x.jpg", xConvo );
	//imwrite( "y.jpg", yConvo );
	//imwrite( "Mag.jpg", mag );
	//imwrite( "Dir.jpg", dir );
}

void direction(cv::Mat &input, Mat_<int> kernel, cv::Mat &convo){

convo.create(input.size(), CV_32FC1);
// we need to create a padded version of the input
// or there will be border effects
int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

cv::Mat paddedInput;
cv::copyMakeBorder( input, paddedInput,
	kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
	cv::BORDER_REPLICATE );
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			float sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ){

					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					int kernalval = kernel.at<int>( kernelx, kernely );

					// do the multiplication
					sum = sum + ( imageval * kernalval );
				}
			}
			convo.at<float>(i, j) = sum;
		}
	}
}

void magDir(cv::Mat &inputx, cv::Mat &inputy, cv::Mat &mag, cv::Mat &dir)
{
	mag.create(inputx.size(), CV_32FC1);
	dir.create(inputx.size(), CV_32FC1);

	for ( int i = 0; i < inputx.rows; i++ )
	{
		for( int j = 0; j < inputx.cols; j++ )
		{
			float magx = inputx.at<float>(i, j);
			float magy = inputy.at<float>(i, j);
			float magTemp = sqrt((magx*magx) + (magy*magy));
			float dirTemp = atan2(magy,  magx);
			mag.at<float>(i, j) = magTemp;
			dir.at<float>(i, j) = dirTemp;
		}
	}
	normaliseMatrix( mag );
}
