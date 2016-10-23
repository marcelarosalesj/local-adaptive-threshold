/*
From this tutorial
http://www.learnopencv.com/applycolormap-for-pseudocoloring-in-opencv-c-python/

*/
//#include <opencv2/contrib/contrib.hpp> // had to remove for opencv3
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp" // I added this to make it work with opencv3
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace cv;
using namespace std; // for the cout...

void getHistogram(Mat input){
	std::vector<int> hist(256,0);
	int biggest = 0;
	// Store pixels' values
	cout << "cols: "<<input.cols << "rows:" <<input.rows<< endl;
	for(int i = 0; i < input.rows; i++){
		for(int j = 0; j < input.cols; j++){
			 hist[ int(input.at<uchar>(i,j)) ]  += 1;
			 if(hist[ int(input.at<uchar>(i,j)) ] > biggest)
			 	biggest = hist[ int(input.at<uchar>(i,j)) ];

		}
	}
	// Generate histogram
	int localheight=256*1;
	Mat display(localheight , 256, CV_8UC1, Scalar(0) );
	for(int i = 0; i < 256 ; i++){
		// cout << int(hist[i]) << " (" << hist[i] * (localheight-1)/(biggest+0) <<  ") , " << i << endl;
		display.at<uchar>(  (localheight-1) - hist[i] * (localheight-1)/(biggest+0) , i) = 255;

	}
	// Display histogram
	imshow("Histogram", display);

}

int main()
{
	// Read image
	Mat img = imread("chronology.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	resize(img, img, Size(img.cols/2, img.rows/2));
	imshow("Image", img);

	// Get histogram from input image
	getHistogram(img);

	// Normal Threshold
	//Mat imgth( img.rows, img.cols, CV_8UC1, Scalar(0) );
	//threshold( img, imgth, threshold_value, max_BINARY_value, BINARY );


	//imshow("Image Threshold", imgth);

	waitKey(0);
    cout<<"Goodbye!"<<endl;

    return 0;
}