# Computer Vision - Detecting Dartboards Coursework (2020)
(Graded at 70%)

This project used standard computer vision techniques to detect the presence of dartboards within a variety of sample images.

A a sample hough transform output for concentric circles and clustered lines is given below.

![Dartboard Hough Transform](https://github.com/jakeramaer/Computer-Vision-2020/blob/master/0hough.jpg)

This detector uses the hough transform to estimate the center of the dartboard and creates a bounding box around it based on the size of the dartboard.

![Dartboard Hough Transform](https://github.com/jakeramaer/Computer-Vision-2020/blob/master/detected0.jpg)

## Key Features
* Sobel operator for object edge detection
* Hough transforms for common dartboard characteristics (Overlapping lines, concentric circles)

## Setup
compile with "g++ main.cpp /usr/lib64/libopencv_core.so.2.4  /usr/lib64/libopencv_highgui.so.2.4  /usr/lib64/libopencv_imgproc.so.2.4  /usr/lib64/libopencv_objdetect.so.2.4 -std=c++11"

run with a.out "image number"
