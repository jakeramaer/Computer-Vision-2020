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
float iou_score(Rect A, Rect B);
void printRectangles(Mat &frame, vector<Rect> new_detected_dartboards);
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
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n");
		return -1; };
		detectAndDisplay( frame, stoi(imgnum) );
		imwrite( "detected.jpg", frame );
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, int imgnum){
	std::vector<Rect> detected_dartboards;
	Mat frame_grey;
	float iou_threshold = 0.5;
	float true_positives = 0;
	float false_positives = 0;
	cvtColor( frame, frame_grey, CV_BGR2GRAY );
	vector<Rect> truth_dartboards = ground_truth(imgnum);
	//GaussianBlur(frame_grey, frame_grey, Size(3,3), 2); // Does this help?
	equalizeHist( frame_grey, frame_grey );
	cascade.detectMultiScale( frame_grey,detected_dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE,Size(50, 50), Size(500,500) );
	// cout << "Original detected dartboards" << endl;
	// printFaces(detected_dartboards);
	// Sobel filter
	Mat xConvo, yConvo, mag, dir;
	sobel(frame_grey, xConvo, yConvo, mag, dir);
	// Hough
	Mat points;
	hough(frame, mag, dir, points);
	vector<Rect> new_detected_dartboards;
	int inc = 0;
	for (int i = 0; i < detected_dartboards.size(); i++){
		// Create a cropped image of each voilaJones detection
		// '/2' because points is half the size of the frame
		Rect rect = Rect((detected_dartboards[i].x / 2)-1, (detected_dartboards[i].y / 2)-1,(detected_dartboards[i].width / 2)-1,(detected_dartboards[i].height / 2)-1);
		Mat violaJones = points(rect);
		bool detected = false;
		int center[2] = {0,0};
		hough_detect(violaJones, &detected, center);
		if(detected) {
			int width = detected_dartboards[i].width;
			int height = detected_dartboards[i].height;
			int oldX = detected_dartboards[i].x;
			int oldY = detected_dartboards[i].y;
			int newX = oldX - width/2 + center[1];
			int newY = oldY - height/2 + center[0];
			Rect newRect = Rect(newX, newY, width, height);
			if(inc == 0){
				new_detected_dartboards.push_back(newRect);
			}
			bool add = false;
			for(int k = 0; k < new_detected_dartboards.size(); k++) {
				float iou = iou_score(new_detected_dartboards[k], newRect);
				if(iou > 0){
					if(new_detected_dartboards[k].width > newRect.width){
						add = true;
						new_detected_dartboards.erase(new_detected_dartboards.begin() + k);
						inc++;
						break;
					}else{
						add = false;
						inc++;
						break;
					}
				}else{ add = true; }
				inc++;
			}
			if(add){
				new_detected_dartboards.push_back(newRect);
			}
		}
	}
	// cout << "New detected dartboards" << endl;
	// printFaces(new_detected_dartboards);
	printRectangles(frame, new_detected_dartboards);
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

float iou_score(Rect A, Rect B){
	Rect intersection = A&B;
	float iou = (float)intersection.area() /
	((float)A.area()+(float)B.area()-(float)intersection.area());
	return iou;
}

void printFaces(std::vector<Rect> faces){
	for (int i = 0; i < faces.size(); i++){
		std::cout << faces[i] << std::endl;
	}
}

void printRectangles(Mat &frame, vector<Rect> new_detected_dartboards){
	for (int i = 0; i < new_detected_dartboards.size(); i++){
		rectangle(frame,Point(new_detected_dartboards[i].x, new_detected_dartboards[i].y),Point(new_detected_dartboards[i].x + new_detected_dartboards[i].width,new_detected_dartboards[i].y + new_detected_dartboards[i].height),Scalar( 0, 255, 0 ), 2);
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

void calculate_all(){
	for (int imgnum = 0; imgnum<16; imgnum++){
		cout<<"Image: "<<imgnum<<endl;
		string filename =  "dartPictures/dart"+to_string(imgnum)+".jpg";
		Mat frame = imread(filename, CV_LOAD_IMAGE_COLOR);
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); };
		detectAndDisplay( frame, imgnum);
		//imwrite( "detected"+to_string(imgnum)+".jpg", frame );
	}
}

void normaliseMatrix( Mat matrix ) {
	double min, max;
	cv::minMaxLoc(matrix, &min, &max);
	for ( int i = 0; i < matrix.rows; i++ )
	{
		for( int j = 0; j < matrix.cols; j++ )
		{
			//Normalize calculation
			matrix.at<float>(i, j) = (((matrix.at<float>(i, j) - min) * 255.0) /
			(max - min));
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
