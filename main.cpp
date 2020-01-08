/////////////////////////////////////////////////////////////////////////////
//
// Detect dartboards
//
// compile with "g++ main.cpp /usr/lib64/libopencv_core.so.2.4  /usr/lib64/libopencv_highgui.so.2.4  /usr/lib64/libopencv_imgproc.so.2.4  /usr/lib64/libopencv_objdetect.so.2.4 -std=c++11"
// run with a.out "image number"
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

#include "sobel.h"
#include "hough.h"

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, int imgnum );
void printFaces(std::vector<Rect> faces);
Rect vect_to_rect(vector<int> vect);
void calculate_all();
void sobel(cv::Mat &input, Mat_<int> kernel, cv::Mat &convo);
void magDir(cv::Mat &inputx, cv::Mat &inputy, cv::Mat &mag, cv::Mat &dir);
void hough_circle(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame, Mat &accu_m);
void hough_ellipse(Mat &mag_frame);
void normaliseMatrix( Mat matrix );
vector<int> hough_line(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame);
void hough_clustered_lines(Mat &frame, vector<int> points_lines, Mat &accu_m);
void hough_combine(Mat &accu_circle, Mat &accu_clustered_lines, Mat &points);
void hough_detect(Mat& points, bool *detected, int center[2]);
vector<Rect> ground_truth(int filenum);
float iou(Rect A, Rect B);
float f1_score(float FalsePos, float TruePos, float real_pos);

/** Global variables */
String cascade_name = "dart.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv ){
	string imgnum = argv[1];
	if (imgnum == "all"){
		calculate_all();
	} else {
		string filename =  "dartPictures/dart"+imgnum+".jpg";
		Mat frame = imread(filename, CV_LOAD_IMAGE_COLOR);
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
		detectAndDisplay( frame, stoi(imgnum) );
		imwrite( "detected.jpg", frame );
	}
	return 0;
}

void calculate_all(){
	for (int imgnum = 0; imgnum<16; imgnum++){
		cout << "Image: " << imgnum << endl;
		string filename =  "dartPictures/dart"+to_string(imgnum)+".jpg";
		Mat frame = imread(filename, CV_LOAD_IMAGE_COLOR);
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); };
		detectAndDisplay( frame, imgnum);
		imwrite( "detectedPictures/detected"+to_string(imgnum)+".jpg", frame );
	}
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, int imgnum){
	std::vector<Rect> detected_dartboards;
	vector<Rect> new_detected_dartboards;
	Mat frame_grey;
	cvtColor( frame, frame_grey, CV_BGR2GRAY );
  // GaussianBlur(frame_grey, frame_grey, Size(3,3), 2, 2); // Does this help?
	vector<Rect> truth_dartboards = ground_truth(imgnum);
	float iou_threshold = 0.5;
	float true_positives = 0;
	float false_positives = 0;
	equalizeHist( frame_grey, frame_grey );
	cascade.detectMultiScale( frame_grey,
		detected_dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE,
		Size(50, 50), Size(500,500) );
	// cout << "Original detected dartboards" << endl;
	// printFaces(detected_dartboards);

	// Create the SOBEL kernel in 1D, Y is transpose
	Mat_<int> kernel(3,3);
	Mat_<int>kernelT(3,3);
	kernel << -1, 0, 1, -2, 0, 2, -1, 0, 1;
	kernelT << 1, 2, 1, 0, 0, 0, -1, -2, -1;

	// Sobel filter
	Mat xConvo, yConvo, mag, dir;
	sobel(frame_grey,kernel, xConvo);
	sobel(frame_grey, kernelT, yConvo);
	magDir( xConvo, yConvo, mag, dir);
	//imwrite( "x.jpg", xConvo );
	//imwrite( "y.jpg", yConvo );
	//imwrite( "Mag.jpg", mag );
	//imwrite( "Dir.jpg", dir );

	// Hough Circles
	int x = mag.rows;
	int y = mag.cols;
	int radius = min(x,y)/2;
	int sizes_circles[] = { x, y, radius };
	Mat accu_circles(3, sizes_circles, CV_32FC1, cv::Scalar(0));
	Mat points_circle;
	hough_circle( frame, mag, dir, accu_circles, points_circle );

	// Hough Lines
	Mat accu_clustered_lines;
	vector<int> points_lines = hough_line( frame , mag, dir);
	hough_clustered_lines(frame, points_lines, accu_clustered_lines);
	Mat points;
	hough_combine(points_circle, accu_clustered_lines, points);
	// Draw detected dartboards from cascade for comparison
	for (int i = 0; i < detected_dartboards.size(); i++){
		// Create a cropped image of each voilaJones detection
		// '/2' because points is half the size of the frame
		Mat detected_frame = points(Rect((detected_dartboards[i].x / 2)-1,(detected_dartboards[i].y / 2)-1,(detected_dartboards[i].width / 2)-1,(detected_dartboards[i].height / 2)-1));
 		bool detected = false;
 		int center[2] = {0,0};
 		hough_detect(detected_frame, &detected, center);
		if(detected) {
			int oldX = (detected_dartboards[i].x);
			int oldY = (detected_dartboards[i].y);
			new_detected_dartboards.push_back(detected_dartboards[i]);
			rectangle(frame,
								Point(oldX, oldY),
								Point(oldX + detected_dartboards[i].width,
											oldY + detected_dartboards[i].height),
								Scalar( 0, 255, 0 ), 2);

		}
	}
	// cout << "New Detected Dartboards" << endl;
	// printFaces(new_detected_dartboards);
	//Calculate detected dartboard iou
	for (int j = 0; j < truth_dartboards.size(); j++){
		for( int i = 0; i < new_detected_dartboards.size(); i++ ){
			float iou_result = iou(new_detected_dartboards[i], truth_dartboards[j]);
			// std::cout << "IOU for ground truth face " << truth_dartboards[j] << " with detected face " << detected_faces[i] << " = " << iou_result << std::endl;
			if (iou_result >= iou_threshold){
				true_positives++;
				//Prevents TP being higher than 1 for each face. Assumes that there are no faces inside faces. In which cases we would need to double break
				break;
			}
		}
	}
	false_positives += new_detected_dartboards.size()-true_positives;
	int real_pos = truth_dartboards.size();
	float f1 = f1_score(false_positives, true_positives, real_pos);
	float tpr;
	//Handling for div 0
	if (true_positives == 0) {
		tpr = 0;
	} else {
		tpr = true_positives/real_pos;
	}
	std::cout << "TPR " << tpr << std::endl;
	std::cout << "F1: " << f1 << "\n" << std::endl;
	for (int i = 0; i < truth_dartboards.size(); i++){
		rectangle(frame, Point(truth_dartboards[i].x, truth_dartboards[i].y), Point(truth_dartboards[i].x + truth_dartboards[i].width, truth_dartboards[i].y + truth_dartboards[i].height), Scalar( 0, 0, 255 ), 2);
	}
}

void printFaces(std::vector<Rect> faces){
	for (int i = 0; i < faces.size(); i++){
		std::cout << faces[i] << std::endl;
	}
}

int ***malloc3dArray(int dim1, int dim2, int dim3){
	int i, j, k;
	int ***array = (int ***) malloc(dim1 * sizeof(int **));
	for (i = 0; i < dim1; i++) {
		array[i] = (int **) malloc(dim2 * sizeof(int *));
		for (j = 0; j < dim2; j++) {
			array[i][j] = (int *) malloc(dim3 * sizeof(int));
		}
	}
	return array;
}



void normaliseMatrix( Mat matrix ) {
	double min, max;
	cv::minMaxLoc(matrix, &min, &max);
	for ( int i = 0; i < matrix.rows; i++ )
	{
		for( int j = 0; j < matrix.cols; j++ )
		{
			//Normalize calculation
			matrix.at<float>(i, j) = (((matrix.at<float>(i, j) - min) * 255.0) / (max - min));
		}
	}
}

float f1_score(float FalsePos, float TruePos, float RealPos){
	float precision;
	if (TruePos == 0) {
		precision = 0;
	} else {
		precision = TruePos/(TruePos+FalsePos);
	}
	float recall;
	if (TruePos == 0) {
		recall = 0;
	} else {
		recall = TruePos/(RealPos);
	}
	if (precision == 0 && recall == 0){
		return 0;
	} else {
		return 2*(precision*recall)/(precision+recall);
	}
}

//Using https://www.geeksforgeeks.org/csv-file-management-using-c/
//Can optimise by storing into an array
vector<Rect> ground_truth(int filenum){
	// std::cout << filenum << std::endl;
	// File pointer
    ifstream fin("dart.csv");
	int file2, count = 0;
	vector<string> row;
    string line, word, temp;
	vector<int> face_coords;
	vector<Rect> face_coords_set;
	while (getline(fin, line)) {
        row.clear();
        istringstream iss(line);
        while (getline(iss, word, ',')) {
            row.push_back(word);
        }
        file2 = stoi(row[0]);
        if (file2 == filenum) {
            count = 1;
			row.erase(row.begin());
			// std::cout << "row size = " << row.size() << std::endl;
			for (int i=0; i<row.size(); i=i+4){
				face_coords_set.push_back(Rect(stoi(row[i]),stoi(row[i+1]),stoi(row[i+2])-stoi(row[i]),stoi(row[i+3])-stoi(row[i+1])));
			}
			return face_coords_set;
            break;
        }
    }
    if (count == 0){
        // cout << "Record not found\n";

	}
	return face_coords_set;
}


Rect vect_to_rect(vector<int> vect){
	return Rect(vect[0], vect[1], vect[2]-vect[0], vect[3]-vect[1]);
}

float iou(Rect A, Rect B){
	// std::cout << "A " << A << std::endl;
	// std::cout << "B " << B << std::endl;
    Rect intersection = A&B;
	// std::cout << "inter " << intersection.area() << std::endl;
    float iou = (float)intersection.area() / ((float)A.area()+(float)B.area()-(float)intersection.area());
	// std::cout << "iou " << std::fixed << std::setprecision(5) << iou << std::endl;
    return iou;
}
