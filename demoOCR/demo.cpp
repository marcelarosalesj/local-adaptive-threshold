/*
From this tutorial
http://www.learnopencv.com/applycolormap-for-pseudocoloring-in-opencv-c-python/


Considering this
https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/character_recognition.cpp
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
#include <opencv2/text/ocr.hpp>

using namespace cv;
using namespace std;
using namespace cv::text;

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

	//cout << "PEAKS"<<endl;
	while(!peaks.empty()){
		int p = peaks.top();
		//cout << p << " : "<<hist[p] <<endl;
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

int main(int argc, char* argv[]) {



    if (argc < 3) { // We expect 3 arguments: the program name, the source path and the destination path
        std::cerr << "Usage: " << argv[0] << " FILENAME NUM_DIV" << std::endl;
        return 1;
    }

	// Read image
	string fn = argv[1];
	Mat img = imread("img/"+fn+".jpg",CV_LOAD_IMAGE_GRAYSCALE);
	resize(img, img, Size(img.cols/2, img.rows/2));
	imshow("Image", img);

	// Get histogram from input image
	struct HistData threshold_value;
	threshold_value = getHistogram(img, true);

	Mat imgth = localAdaptiveThresholding(img, atoi( argv[2]) );
	imshow("Image Threshold 2 (Local Adaptive)", imgth);	

	string vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; // must have the same order as the clasifier output classes


	// Method one
 	// Worked
 	Ptr<OCRHMMDecoder::ClassifierCallback> ocr = loadOCRHMMClassifierCNN("OCRBeamSearch_CNN_model_data.xml"); 	

	double t_r = (double)getTickCount();
    vector<int> out_classes;
    vector<double> out_confidences;

    ocr->eval(img, out_classes, out_confidences);
    
    cout << "OCR output = \"" << vocabulary[out_classes[0]] 
    	 << "\" with confidence " << out_confidences[0] << ". Evaluated in "
		 << ((double)getTickCount() - t_r)*1000/getTickFrequency() << " ms." << endl << endl;


/*
	// Method two
	// Didn't find anything
 	Ptr< OCRBeamSearchDecoder::ClassifierCallback> ocr = loadOCRBeamSearchClassifierCNN("OCRBeamSearch_CNN_model_data.xml");
 	vector<int> out_classes;
    vector< vector<double> > probabilities;
	ocr->eval(imgth, probabilities, out_classes);
	cout << " 1) "<< out_classes.empty() << endl;
	cout << " 2) "<< probabilities.empty() << endl;

*/

/*
	// Method three
	// Didn't find anything :(
	
	Mat transition_probabilities;
    string filename = "OCRHMM_transitions_table.xml";
    FileStorage fs(filename, FileStorage::READ);
    fs["transition_probabilities"] >> transition_probabilities;
	fs.release();
    
    Mat emission_probabilities = Mat::eye((int)vocabulary.size(), (int)vocabulary.size(), CV_64FC1);


    Ptr< OCRBeamSearchDecoder > ocr = OCRBeamSearchDecoder::create(
    			loadOCRBeamSearchClassifierCNN("OCRBeamSearch_CNN_model_data.xml"),
    			vocabulary,
    			transition_probabilities,
    			emission_probabilities,
    			OCR_DECODER_VITERBI, 
    			500 );

	string output;
    cout << "HOLA"<<endl;
	vector<Rect> * component_rects = NULL;
	vector<string> * component_texts = NULL;
	vector<float> * component_confidences = NULL;
	int component_level = 2 ;


    ocr->run(imgth, output, component_rects, component_texts, component_confidences, OCR_LEVEL_WORD );
    cout << "OUTPUT:"<<output<<":"<<endl;

*/


	waitKey(0);
    cout<<"Goodbye!"<<endl;

    return 0;
}