#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
using namespace cv;
using namespace std;
//********This demo shows how canny detector works.**********
Mat src, srcGrey;
Mat dst, edges;
const char* windowName = "edge_Canny";
int lowerThreshold = 0;
const int max_lowerThreshold = 100;
const int kernerSize = 5;
static int ratioThreshold = 3;
const Size ksize = Size(kernerSize, kernerSize);
static void CannyThreshold(int, void*) {
	double sigma = 0.3*((kernerSize - 1)*0.5 - 1) + 0.8;
	//********denoise the image
	GaussianBlur(srcGrey, edges, ksize,sigma);
	Canny(edges, edges, lowerThreshold, lowerThreshold*ratioThreshold, kernerSize);
	dst = Scalar::all(0);
	src.copyTo(dst, edges);
	imshow(windowName, dst);
}

int main(int argc, char* arr[]) {
	src = imread("c:/test3.jpg", IMREAD_COLOR);
	cvtColor(src, srcGrey, COLOR_BGR2GRAY);
	dst.create(src.size(), src.type());
	namedWindow(windowName);
	createTrackbar("MinThreshold", windowName, &lowerThreshold, max_lowerThreshold, CannyThreshold, 0);
	createTrackbar("MinThreshold", windowName, &ratioThreshold, 3, CannyThreshold, 0);
	CannyThreshold(0, 0);
	waitKey();
	return 0;



}
