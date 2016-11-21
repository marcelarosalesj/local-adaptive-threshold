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
#include <stack>
#include <opencv2/text.hpp>

using namespace cv;
using namespace std;

struct HistData {
	int high;		// Amount of pixels in the highest valley
	int highpos;	// Position of the highest
	int high2;		// Amount of pixels in the second highest valley
	int high2pos;	// Position of the second highest
	int meanHighest;
	int half;		// 1/2 of total amount of pixels in the image
	int halfindex;	// Position of half
	int first; 		// First position [0-255]
	int last; 		// Last position [0-255]
	int middle;		// Middle position [0-255]
};




struct HistData getHistogram(Mat input, bool showDisplay){
	std::vector<int> hist(256,0);
	HistData data; 	// Here we're going to store the data for this histogram
	data.high = 0;	
	data.highpos = 0;
	data.high2 = 0;
	data.high2pos = 0;
	data.half = 0;

	// Store pixels' values
	for(int i = 0; i < input.rows; i++) {
		for(int j = 0; j < input.cols; j++) {
			// Store the values in the corresponding position in the histogram vector
			hist[ int(input.at<uchar>(i,j)) ]  += 1; 
			// Check if that value is the highest
			if( hist[int(input.at<uchar>(i,j))] > data.high ){
				//data.high2 = data.high; // Not the way
				//data.high2pos = data.high2pos;
				data.high = hist[ int(input.at<uchar>(i,j)) ]; 	// Amount of pixels
			 	data.highpos = j;								// Position of the highest

			}			 	
		}
	}

	// Generate the Mat Display that will provide a visualization for the histogram
	int localheight=256*1;
	Mat display(localheight , 256, CV_8UC1, Scalar(0) );
	for(int i = 0; i < 256 ; i++){
		// cout << int(hist[i]) << " (" << hist[i] * (localheight-1)/(data.high+0) <<  ") , " << i << endl;
		display.at<uchar>(  (localheight-1) - hist[i] * (localheight-1)/(data.high+0) , i) = 255;
	}


	// Get two highest peaks
	stack<int> peaks;
	stack<int> valleys;
	int last=0, next=0;
	bool growing=true;
	for(int i=0; i<hist.size(); i++){
		if(growing){
			if( (hist[i]-last)>=0 ){
				last = hist[i];
			} else {
				peaks.push(i);
				growing=false;
			}
		} else if(!growing){
			if( (hist[i]-last )<=0 ){
				last = hist[i];
			} else {
				valleys.push(i);
				growing=true;
			}
		}
	}

	cout << "PEAKS"<<endl;
	while(!peaks.empty()){
		int p = peaks.top();
		cout << p << " : "<<hist[p] <<endl;
		peaks.pop();

		for(int i=0; i < display.rows; i++){
			display.at<uchar>( i, p) = 255;
		}

	}
	/*cout << "VALEYYS"<<endl;
	while(!valleys.empty()){
		int v = valleys.top();
		cout << v <<endl;
		valleys.pop();
		for(int i=0; i < display.rows; i++){
			display.at<uchar>( i, v) = 255;
		}
	}*/

	

	// Get the half 
	double areahalf = (input.rows*input.cols/2); // Amount of pixels in the half
	bool flag=true;
	data.halfindex = 0;
	while(flag){
		data.half += hist[data.halfindex];
		data.halfindex++;
		if(data.half >= areahalf)
			flag=false;
	}	

	// Get first
	for(int i=0; i<hist.size(); i++){
		if(hist[i] != 0){
			data.first=i;
			i=hist.size()+1;
		}
	}
	// Get last
	for(int i=hist.size()-1; i>0; i--){
		if(hist[i] != 0){
			data.last = i;
			i=0;
		}
	}
	// Get middle
	data.middle = (data.first+data.last)/2;

	// Draw half, last and first
	/*for(int i=0; i < display.rows; i++){
		//display.at<uchar>( i, data.halfindex ) = 180;
		//display.at<uchar>( i, data.first ) = 180;
		//display.at<uchar>( i, data.last ) = 180;
		//display.at<uchar>( i, data.middle ) = 180;
		display.at<uchar>( i, data.highpos ) = 255;
		display.at<uchar>( i, data.high2pos ) = 255;
	}*/

	// Display histogram
	if(showDisplay){
		imshow("Histogram", display);
	}

	return data;
}

Mat localAdaptiveThresholding(Mat input, int granularity){
	Mat output(input.rows, input.cols, CV_8UC1, Scalar(0));
	//cout << "INPUT R:"<<input.rows<<" C:"<<input.cols<<endl;
		
	int dx, dx2, dy, dy2;
	dx = input.cols/granularity;
	dx2 = input.cols/granularity + input.cols%granularity;
	dy = input.rows/granularity;
	dy2 = input.rows/granularity + input.rows%granularity;

	//cout << " dx: " << dx << " dx2: " << dx2 << endl;
	//cout << " dy: " << dy << " dy2: " << dy2 << endl;

	// Rect(int x, int y, int width, int height)
	for(int i=0; i<granularity; i++){
		for(int j=0; j<granularity; j++){

			if( i < (granularity-1) ){
				if( j < (granularity-1) ){
					// First type of container

					// Cut dx*dy from input Mat
					Mat roi = input( Rect(j*dx, i*dy, dx, dy ) );
					//imshow("ROI", roi);

					Mat roith;
					struct HistData tv = getHistogram(roi, false);
					int const max_BINARY_value = 255;
					int threshold_type = 0;
					threshold( roi, roith, tv.middle , max_BINARY_value, threshold_type );
					
					//imshow("ROI TH", roith);
					// Paste thresholded small-Mat into the bigger one (which is the result)
					roith.copyTo(output(cv::Rect(j*dx,i*dy,roith.cols, roith.rows)));

					//waitKey(0);



				} else {
					// Second type of container

					// Cut input Mat 
					Mat roi = input( Rect(j*dx, i*dy, dx2, dy ) );
					//imshow("ROI", roi);

					Mat roith;
					struct HistData tv = getHistogram(roi, false);
					int const max_BINARY_value = 255;
					int threshold_type = 0;
					threshold( roi, roith, tv.middle , max_BINARY_value, threshold_type );
					
					//imshow("ROI TH", roith);
					roith.copyTo(output(cv::Rect(j*dx,i*dy,roith.cols, roith.rows)));
					//waitKey(0);
				}
			} else {
				if( j < (granularity-1) ){
					//Third type of container

					// Cut input Mat
					Mat roi = input( Rect(j*dx, i*dy, dx, dy2 ) );
					//imshow("ROI", roi);	

					Mat roith;
					struct HistData tv = getHistogram(roi, false);
					int const max_BINARY_value = 255;
					int threshold_type = 0;
					threshold( roi, roith, tv.middle , max_BINARY_value, threshold_type );
					
					//imshow("ROI TH", roith);
					roith.copyTo(output(cv::Rect(j*dx,i*dy,roith.cols, roith.rows)));
					//waitKey(0);
				} else {
					// Fourth type of container

					// Cut input Mat
					Mat roi = input( Rect(j*dx, i*dy, dx2, dy2 ) );
					//imshow("ROI", roi);

					Mat roith;
					struct HistData tv = getHistogram(roi, false);
					int const max_BINARY_value = 255;
					int threshold_type = 0;
					threshold( roi, roith, tv.middle , max_BINARY_value, threshold_type );
					
					//imshow("ROI TH", roith);
					roith.copyTo(output(cv::Rect(j*dx,i*dy,roith.cols, roith.rows)));
					//waitKey(0);
				}

			} // If for selecting contaniers

			

		} // For - go through columns
	} // For - go through rows


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
	imshow("OpenCV Histogram", histImage );


	// Get histogram from input image
	struct HistData threshold_value;
	threshold_value = getHistogram(img, true);

	// Normal Threshold
	Mat imgth( img.rows, img.cols, CV_8UC1, Scalar(0) );
	int const max_BINARY_value = 255;
	int threshold_type = 1;
		 	//0: Binary
		    //1: Binary Inverted
		    //2: Threshold Truncated
		    //3: Threshold to Zero
		    //4: Threshold to Zero Inverted
		  
	cout << " threshold_value = "<< threshold_value.middle << " type:" << threshold_type<<endl;
	threshold( img, imgth, threshold_value.middle, max_BINARY_value, threshold_type );

	imshow("Image Threshold 1 (No Local Adaptive)", imgth);
	
	// Local Adaptive Threshold
	Mat imgth_2( img.rows, img.cols, CV_8UC1, Scalar(0) );
	cout << "n?"<<endl;
	cout << " (The input matrix will be divided in n*n subregions for the locally adaptive threshold)" << endl;
	int n;
	cin >> n;
	imgth_2 = localAdaptiveThresholding(img, n);
	imshow("Image Threshold 2 (Local Adaptive)", imgth_2);	

	waitKey(0);
    cout<<"Goodbye!"<<endl;

    return 0;
}