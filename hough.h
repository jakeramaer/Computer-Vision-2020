/////////////////////////////////////////////////////////////////////////////
//
// HOUGH header functions
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
void hough_circle(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame,
	                Mat &accu_m, Mat &points);
vector<int> hough_line(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame);
void hough_clustered_lines(Mat &frame, vector<int> points_lines, Mat &accu_m);
void hough_combine(Mat &accu_circle, Mat &accu_clustered_lines, Mat& points);
void hough_detect(Mat& points, bool *detected, int center[2]);
/** */

void hough(Mat &frame, Mat &mag, Mat &dir, Mat &points){
  // Hough Circles
	int x = mag.rows;
	int y = mag.cols;
	int radius = min(x,y)/2;
	int sizes_circles[] = { x, y, radius };
	Mat accu_circles(3, sizes_circles, CV_32FC1, cv::Scalar(0));
	Mat points_circle;
	hough_circle( frame, mag, dir, accu_circles, points_circle);

	// Hough Lines
	Mat accu_clustered_lines;
	vector<int> points_lines = hough_line( frame , mag, dir);
	hough_clustered_lines(frame, points_lines, accu_clustered_lines);

	// Combine Circle and Line
	hough_combine(points_circle, accu_clustered_lines, points);
}


void hough_circle(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame,
	Mat &accu_m, Mat &points){
	int minRadius = 15;
	const int x = mag_frame.rows;
	const int y = mag_frame.cols;
	const int radius = min(x,y) / 4;  // can be max half the size of dimensions
  points = Mat(x/2, y/2, CV_32S, cvScalar(0));
	// Look for strong edges
	for( int i = 0; i < x; i++){
		for( int j = 0; j < y; j++){
			// Pixel threshold
			if(mag_frame.at<float>(i, j) > 50) {
				// Vote for circles corresponding to point's +ve and -ve direction
				for( int r = minRadius; r < radius; r++){
					float d = dir_frame.at<float>(i, j);
					int a = i + r*cos(d);
					int b = j + r*sin(d);
					// Check that the radius doesn't exceed the boarder
					if((a > 0) && (b > 0) && (a < x) && (b < y)){
						accu_m.at<int>(a, b, r) += 1;
					}
					a = i - r*cos(d);
					b = j - r*sin(d);
					// Check that the radius doesn't exceed the boarder
					if((a > 0) && (b > 0) && (a < x) && (b < y)){
						accu_m.at<int>(a, b, r) += 1;
					}
				}
			}
		}
	}

	// Condensing points to (x,y) by summing radii
	for( int i = 0; i < mag_frame.rows; i++){
		for( int j = 0; j < mag_frame.cols; j++){
			for( int r = minRadius; r < radius; r++){
				if(i/2 < x/2 && j/2 < y/2){
					points.at<int>(i/2, j/2) += accu_m.at<int>(i,j,r);
				}
			}
		}
	}
	normaliseMatrix( points );
	//imwrite("circle_hough.jpg", points);
}

void hough_ellipse(Mat &frame){
	//Calculate hough space for ellipse
	//Threshold
	//If above threshold, return true
}

vector<int> hough_line(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame){

	vector<int> points_lines;
	const int d = sqrt((mag_frame.rows*mag_frame.rows) +
	(mag_frame.cols*mag_frame.cols)); //Diameter of image
	Mat accu_m = Mat(d, 180, CV_32S, cvScalar(0));

	// Look for strong edges
	for( int i = 0; i < mag_frame.rows; i++){
		for( int j = 0; j < mag_frame.cols; j++){
			if(mag_frame.at<float>(i, j) > 45) { // Threshhold
				// If edge is strong, vote for lines corresponding to its point
				for( int r = 0; r < 180; r++){
					int p = (i*cos(r*CV_PI/180) + (j*sin(r*CV_PI/180)));
					if(p > 0) {
						accu_m.at<int>(p, r)++;
					}
				}
			}
		}
	}

	for( int p = 0; p < d; p++) {
		for( int r = 0; r < 180; r++) {
			if(accu_m.at<int>(p, r) > 100) { // Threshold
				float a = cos(r*CV_PI/180);
				float b = sin(r*CV_PI/180);
				int x0 = a*p;
				int y0 = b*p;
				int x1 = int(x0 + 1000*(-b));
				int y1 = int(y0 + 1000*(a));
				int x2 = int(x0 - 1000*(-b));
				int y2 = int(y0 - 1000*(a));
				points_lines.push_back(x1);
				points_lines.push_back(y1);
				points_lines.push_back(x2);
				points_lines.push_back(y2);
			}
		}
	}
  //imwrite("lines_hough.jpg", accu_m);
	return points_lines;
}

void hough_clustered_lines(Mat &frame, vector<int> points_lines, Mat &accu_m){
	vector<int> points_clustered_lines;
	// Reducing the resolution of the image to reduce spread of points
	accu_m = Mat(frame.rows/2, frame.cols/2, CV_32S, cvScalar(0));

	for( int p = 0; p < points_lines.size(); p = p + 4){
		float x1 = (float)points_lines[p]   /2;
		float y1 = (float)points_lines[p+1] /2;
		float x2 = (float)points_lines[p+2] /2;
		float y2 = (float)points_lines[p+3] /2;
		float m = (y1 - y2) / (x1 - x2);
		float c = y1 - (m*x1);

		for(int x = 0; x < frame.rows/2; x ++){
			int y = (int)(x*m) + c;
			if( (y > 0) && (y < frame.cols/2)){
				accu_m.at<int>(x, y)++;
			}
		}
	}
	normaliseMatrix(accu_m);
	//imwrite("clustered_hough.jpg", accu_m);
}

void hough_combine(Mat &accu_circle, Mat &accu_clustered_lines, Mat& points){
	const int x = accu_circle.rows;
	const int y = accu_circle.cols;
	points = Mat(x, y, CV_32S, cvScalar(0));
	for(int i = 0; i < x; i++) {
		for(int j = 0; j < y; j++) {
			points.at<float>(i, j) += accu_circle.at<float>(i, j);
			points.at<float>(i, j) += accu_clustered_lines.at<float>(i, j);
		}
	}
	normaliseMatrix(points);
	//imwrite("both1.jpg", points);
}

// Send in specific boxes, if any value is 255 return true
void hough_detect(Mat& points, bool *detected, int center[2]){
	const int x = points.rows;
	const int y = points.cols;
	for(int i=0; i < x; i++){
		for(int j = 0; j < y; j++){
			if(points.at<int>(i,j) > 235){
				*detected = true;
				// *2 because points is half-sized
				center[0] = i * 2;
				center[1] = j * 2;
			}
		}
	}
}
