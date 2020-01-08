/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
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
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame , int imageNumber);

/** Global variables */
String cascade_name = "dart.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
        // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	int imageNumber = atoi(argv[2]);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	
	// 3. Detect Faces and Display Result
        detectAndDisplay( frame , imageNumber);

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame , int imageNumber)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	// Loading true position coordinates into matrix
        vector<double> matrix;
        fstream file;
        file.open("dart.csv");
        string line;
        while (getline( file, line,'\n'))
	{
             istringstream templine(line); 
             string data;
	     while (getline( templine, data,',')) 
	        {
	           matrix.push_back(atof(data.c_str())); 
                }
	}
        vector<double> coordinates;
        for(int i = 0; i < matrix.size(); i = i+5)
             {
             if(matrix[i] == imageNumber)
		{
                   for(int j = 1; j < 5; j++)
		     {
		        coordinates.push_back(matrix[i+j]);
                     }
                }
             }
        file.close();
        
        //IOU
        for(int j = 0; j < coordinates.size(); j = j+4){
             Rect r1 = Rect(Point(coordinates[j], coordinates[j+1]), Point(coordinates[j+2], coordinates[j+3]));
             float maxiou = 0;
             for( int i = 0; i < faces.size(); i++ ) {
                 Rect r2 = Rect(Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height));
                 float is = (r1 & r2).area();
                 float u = (r1 | r2).area();
                 float iou = (double)is/u;
                 if(iou > maxiou){
                     maxiou = iou;
                 }
             }
             // Draw RED box around true position 
             rectangle(frame, Point(coordinates[j], coordinates[j+1]), Point(coordinates[j+2], coordinates[j+3]), Scalar( 0, 0, 255 ), 2);
             // Print IOU
             std::cout << "IOU for dartboard " << (j/4) + 1 << " = "<< maxiou << std::endl;
        }

        // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

        // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
        

}
