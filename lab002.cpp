#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<math.h>
#include<vector>
using namespace cv;
using namespace std;
//*************This demo shows the results of images processed by equalized-histogram
//*******method, as well as shows that the v channel of hsv image in openCV
//*******is actually the r-channel of rgb image(not the grey in hsi).
//*******Perhaps the reason is that the base hue is red
//
//***get the look-up table from the equalized histogram
void GetGreyHistEqualizationLUT(Mat& lutGrey,Mat& hist, int size) {
	uchar* p = lutGrey.ptr();
	double val=0;
	for (int i = 0; i < 256; i++) {
		val += hist.at<float>(i);
		p[i] = saturate_cast<uchar>(255*val / size);
	}
}
//***show the colored histogram ,didn't finish yet
void ColoredHistEqualization() {
}
//***show the normalized histogram(line-type & rectangle-type)
void ShowNormalizedHist(Mat& img,Mat& hist) {
	int bins = 256; //the hist size
	int histSize[] = { bins };
	float range[] = { 0,256 };//the caculating limit
	const float* ranges[] = { range };
	int channels[] = { 0 };// the index of img.channel 
	calcHist(&img, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
	//imshow("histOrigin", hist);
	//double maxVal = 0;
	//minMaxLoc(hist, 0, &maxVal, 0, 0);
	//std::cout << maxVal << std::endl;
	Mat histTmp = hist.clone();
	int hist_h = 256, hist_w = 512;//set the info of the display image
	int bin_w = cvRound((double)hist_w / histSize[0]);//the width of each bin
	Mat histImg(hist_h, hist_w, CV_8UC3,Scalar(0,0,0));
	//normalize the hist.
	normalize(histTmp, histTmp, 0, histImg.rows, NORM_MINMAX, -1, Mat());
	//****!!!!!!NOTE THAT:the origin of image in openCV is 
	//in the <<Left Top Corner>>
	for (int i = 1; i < histSize[0]; i++) {
		line(histImg, Point(bin_w*(i - 1), hist_h - cvRound(histTmp.at<float>(i - 1))),
			Point(bin_w*i, hist_h - cvRound(histTmp.at<float>(i))), 
			Scalar(255, 0, 0), 2, 8, 0);
	}
	imshow("histLine", histImg);
	for (int i = 1; i < histSize[0]; i++) {
		rectangle(histImg,Point(bin_w*(i-1),hist_h),
			Point(bin_w*i, hist_h - cvRound(histTmp.at<float>(i))),
			Scalar(255, 0, 0), 1, 8, 0);
	}
	imshow("input", img);
	imshow("hist", histImg);	
	waitKey();
}
int main(int argc, char** argv) {
	Mat img = imread("C:/test3.jpg", IMREAD_COLOR);
	Mat grey = imread("C:/test3.jpg", IMREAD_GRAYSCALE);
	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);
	Mat histGrey;
	Mat lutGrey(1, 256, CV_8U);
	//********definition for color images
	vector<Mat> bgrPlanes;
	vector<Mat> hsvPlanes;
	split(img, bgrPlanes);
	split(hsv, hsvPlanes);
	waitKey();
	Mat cv_hist;
	equalizeHist(grey, cv_hist);
	ShowNormalizedHist(grey, histGrey);
	GetGreyHistEqualizationLUT(lutGrey, histGrey, grey.cols*grey.rows);
	//************using equalized histogram process grey image
	Mat newGrey;
	Mat newhist;
	LUT(grey, lutGrey, newGrey);
	ShowNormalizedHist(newGrey, newhist);
	imshow("newGrey", newGrey);
	imshow("input", grey);
	waitKey(0);
	//***********process the color image(RGB)
	Mat lutB(1, 256, CV_8U);
	Mat bHist;
	ShowNormalizedHist(bgrPlanes[0], bHist);
	GetGreyHistEqualizationLUT(lutB, bHist, img.cols*img.rows);
	Mat lutG(1, 256, CV_8U);
	Mat gHist;
	ShowNormalizedHist(bgrPlanes[1], gHist);
	GetGreyHistEqualizationLUT(lutG, gHist, img.cols*img.rows);
	Mat lutR(1, 256, CV_8U);
	Mat rHist;
	Mat b;
	Mat g;
	Mat r;
	ShowNormalizedHist(bgrPlanes[2], rHist);
	GetGreyHistEqualizationLUT(lutR, rHist, img.cols*img.rows);
	LUT(bgrPlanes[0], lutB, b);
	LUT(bgrPlanes[1], lutG, g);
	LUT(bgrPlanes[2], lutR, r);
	imshow("B", b);
	imshow("G", g);
	imshow("R", r);
	Mat color;
	vector<Mat> bgr = { b,g,r };
	merge(bgr, color);
	destroyWindow("input");
	destroyWindow("hist");
	imshow("res", color);
	//imshow("src", hsv);
	//**********process the color image(HSV)
	Mat newColor;
    Mat vhist;
	Mat lutv(1, 256, CV_8U);
	Mat v;
	ShowNormalizedHist(hsvPlanes[2],vhist);
	GetGreyHistEqualizationLUT(lutv,vhist,img.cols*img.rows);
	LUT(hsvPlanes[2],lutv,v);
	hsvPlanes[2]=v;
	merge(hsvPlanes,newColor);
	cvtColor(newColor, newColor, COLOR_HSV2BGR);
	//destroyWindow("input");
	//destroyWindow("hist");
	imshow("hsv", newColor);
	imshow("src", img);
	waitKey();
	return 0;
}
