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


struct HistData {
	int high;
	int highpos;
	int half;
	int halfindex;

};


int getHistogram(Mat input){
	std::vector<int> hist(256,0);
	HistData data;
	data.high = 0;
	data.highpos = 0;
	data.half = 0;
	// Store pixels' values
	for(int i = 0; i < input.rows; i++){
		for(int j = 0; j < input.cols; j++){
			 hist[ int(input.at<uchar>(i,j)) ]  += 1;
			 if(hist[ int(input.at<uchar>(i,j)) ] > data.high){
			 	data.high = hist[ int(input.at<uchar>(i,j)) ];
			 	data.highpos = j;
			 }			 	
		}
	}
	// Generate histogram
	int localheight=256*1;
	Mat display(localheight , 256, CV_8UC1, Scalar(0) );
	for(int i = 0; i < 256 ; i++){
		// cout << int(hist[i]) << " (" << hist[i] * (localheight-1)/(data.high+0) <<  ") , " << i << endl;
		display.at<uchar>(  (localheight-1) - hist[i] * (localheight-1)/(data.high+0) , i) = 255;

	}


	// Get the half
	double areahalf = (input.rows*input.cols/2);
	bool flag=true;
	int idx = 0;
	while(flag){
		data.half += hist[idx];
		idx++;
		if(data.half >= areahalf)
			flag=false;
	}
	cout << "THE INDEX IS    "<<idx<<endl;
	cout << "   area: "<< data.half <<endl;
	data.halfindex=idx;
	// Draw half
	for(int i=0; i < display.rows; i++){
		display.at<uchar>( i, idx ) = 180;
	}


	// Display histogram
	imshow("Histogram", display);

	return data.halfindex;
}

Mat localAdaptiveThresholding(Mat input, int granularity){
	Mat output(input.rows, input.cols, CV_8UC1, Scalar(0));
	cout << "INPUT R:"<<input.rows<<" C:"<<input.cols<<endl;
		
	int dx, dx2, dy, dy2;
	dx = input.cols/granularity;
	dx2 = input.cols/granularity + input.cols%granularity;
	dy = input.rows/granularity;
	dy2 = input.rows/granularity + input.rows%granularity;

	cout << " dx: " << dx << " dx2: " << dx2 << endl;
	cout << " dy: " << dy << " dy2: " << dy2 << endl;

	// Rect(int x, int y, int width, int height)
	for(int i=0; i<granularity; i++){
		for(int j=0; j<granularity; j++){
			if( i < (granularity-1) ){
				if( j < (granularity-1) ){
					cout << " CONTAINER 1    (" << i<< "," <<j <<")" <<endl;
					Mat roi = input( Rect(j*dx, i*dy, dx, dy ) );
					imshow("ROI", roi);

					Mat roith;
					int tv = getHistogram(roi);
					cout << "EN LATH::"<<tv<<endl;
					int const max_BINARY_value = 255;
					int threshold_type = 1;
					threshold( roi, roith, tv, max_BINARY_value, threshold_type );
					
					imshow("ROI TH", roith);
					roith.copyTo(output(cv::Rect(j*dx,i*dy,roith.cols, roith.rows)));

					//waitKey(0);



				} else {
					cout << " CONTAINER 2    (" << i<< "," <<j <<")" <<endl;
					Mat roi = input( Rect(j*dx, i*dy, dx2, dy ) );
					imshow("ROI", roi);

					Mat roith;
					int tv = getHistogram(roi);
					cout << "EN LATH::"<<tv<<endl;
					int const max_BINARY_value = 255;
					int threshold_type = 1;
					threshold( roi, roith, tv, max_BINARY_value, threshold_type );
					
					imshow("ROI TH", roith);
					roith.copyTo(output(cv::Rect(j*dx,i*dy,roith.cols, roith.rows)));
					//waitKey(0);


				}
			} else {
				if( j < (granularity-1) ){
					cout << " CONTAINER 3    (" << i<< "," <<j <<")" <<endl;
					Mat roi = input( Rect(j*dx, i*dy, dx, dy2 ) );
					imshow("ROI", roi);	

					Mat roith;
					int tv = getHistogram(roi);
					cout << "EN LATH::"<<tv<<endl;
					int const max_BINARY_value = 255;
					int threshold_type = 1;
					threshold( roi, roith, tv, max_BINARY_value, threshold_type );
					
					imshow("ROI TH", roith);
					roith.copyTo(output(cv::Rect(j*dx,i*dy,roith.cols, roith.rows)));
					//waitKey(0);



				} else {
					cout << " CONTAINER 4    (" << i<< "," <<j <<")" <<endl;
					Mat roi = input( Rect(j*dx, i*dy, dx2, dy2 ) );
					imshow("ROI", roi);

					Mat roith;
					int tv = getHistogram(roi);
					cout << "EN LATH::"<<tv<<endl;
					int const max_BINARY_value = 255;
					int threshold_type = 1;
					threshold( roi, roith, tv, max_BINARY_value, threshold_type );
					
					imshow("ROI TH", roith);
					roith.copyTo(output(cv::Rect(j*dx,i*dy,roith.cols, roith.rows)));
					//waitKey(0);




				}

			}

			

		}
	}


	return output;
}	

int main()
{
	// Read image
	Mat img = imread("chronology.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	resize(img, img, Size(img.cols/2, img.rows/2));
	imshow("Image", img);

	// OpenCV histogram function
	// http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html
	Mat g_hist;
	int histSize = 256;
	float range[] = { 0, 256 } ; //the upper boundary is exclusive
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	calcHist( &img, 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );

	int hist_w = 256; int hist_h = 256;
	int bin_w = cvRound( (double) hist_w/histSize );
	Mat histImage( hist_h, hist_w, CV_8UC1, Scalar(0) );
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ ) {
	    line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
	                     Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
	                     Scalar( 255), 2, 8, 0  );
	}
	imshow("calcHist Demo", histImage );


	// Get histogram from input image
	/*int threshold_value;
	threshold_value = getHistogram(img);

	// Normal Threshold
	Mat imgth( img.rows, img.cols, CV_8UC1, Scalar(0) );
	//int threshold_value = 200;
	int const max_BINARY_value = 255;
	int threshold_type = 1;
		 	//0: Binary
		    //1: Binary Inverted
		    //2: Threshold Truncated
		    //3: Threshold to Zero
		    //4: Threshold to Zero Inverted
		  
	cout << " threshold_value = "<< threshold_value << " type:" << threshold_type<<endl;
	threshold( img, imgth, threshold_value, max_BINARY_value, threshold_type );

	imshow("Image Threshold", imgth);
	*/
	// Local Adaptive Threshold
	Mat imgth_2( img.rows, img.cols, CV_8UC1, Scalar(0) );
	imgth_2 = localAdaptiveThresholding(img, 6);
	imshow("Image Threshold 2 (Local Adaptive)", imgth_2);	

	waitKey(0);
    cout<<"Goodbye!"<<endl;

    return 0;
}