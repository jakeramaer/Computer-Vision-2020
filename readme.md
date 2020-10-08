# Computer Vision - Detecting Dartboards Coursework (2020)
(Graded at 70%)

## General
This project used standard computer vision techniques to detect the presence of dartboards within a variety of sample images.

A sample output is given below.



## Key Features
* Sobel operator for object edge detection
* Hough transforms for common dartboard characteristics (Overlapping lines, concentric circles)
	
## Technologies
Project is created with:
* GLM (OpenGL) version: 0.9.7.2

## Setup
compile with "g++ main.cpp /usr/lib64/libopencv_core.so.2.4  /usr/lib64/libopencv_highgui.so.2.4  /usr/lib64/libopencv_imgproc.so.2.4  /usr/lib64/libopencv_objdetect.so.2.4 -std=c++11"

run with a.out "image number"
