#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgcodecs/imgcodecs.hpp>
#include<iostream>
#include<vector>
using namespace cv;
using namespace std;
//******This demo shows the effect of different filter kernels in the space domain.
//******The summary:(not done yet)
enum kernelType
{
	BLUR = 1,
	GAUSSIAN=2,
	SOBEL=3,
	LAPLACE=4,
	ROBERT=5,
	HIGHRAISE=6

};
static float arrGauss1[9] = { 1.f/16, 2.f/16, 1.f/16,
					          2.f/16, 4.f/16, 2.f/16,
					          1.f/16, 2.f/16, 1.f/16 };
static float arrRobertX[4] = { -1.f,0,
						         0,1.f };
static float arrRobertY[4] = { 0,-1.f,
							 1.f,0 };
static float arrSobelX[9] = { -1.f,-2.f,-1.f,
							     0,   0,   0,
							   1.f, 2.f, 1.f };
static float arrSobelY[9] = { -1.f,0,1.f,
							  -2.f,0,2.f,
							  -1.f,0,1.f };
void ModifiedFilter(Mat& img,int kernelSize,int kernelType,bool colored=false) {
	Mat src = img.clone();
	Mat dst;
	double sigma = 0.3*((kernelSize - 1)*0.5 - 1) + 0.8;//magic from openCV tutorial
	if (kernelType == GAUSSIAN) {
		Mat kernel=(Mat_<float>(3, 3, arrGauss1));
		Mat kernelGauss = getGaussianKernel(kernelSize, sigma);
		cout << kernelGauss << endl;
		Point anchor = Point(-1, -1);// the default anchor point ,can be changed
		switch (kernelSize)
		{
			//kernelsize=3,using the kernel constructed by my self
		case 3:filter2D(src, dst, -1, kernel, anchor); 
			break;

		default:
			filter2D(src, dst, -1, kernelGauss, anchor);
			break;
		}
	 }
	else if(kernelType==ROBERT)
	{
		Mat kernel = (Mat_<float>(2, 2, arrRobertX));
		Point anchor = Point(-1, -1);
		Mat dstx, dsty;
		//!!!!!!!Notice that one Robert kernel processes only one direction
		filter2D(src, dstx, -1, kernel,anchor);
		kernel = (Mat_<float>(2, 2, arrRobertY));
		filter2D(src, dsty, -1, kernel, anchor);
		addWeighted(dstx, 1, dsty, 1, 0, dst);
	}
	else if (kernelType == HIGHRAISE) {
		//*****the idea is that getting the mask image(keep the details)
		//by subscribing the smoothed part from the origin image
		//after that adding the k*mask to the origin image
		Mat smooth;
		Mat mask;
		int k = 3;
		Size ksize = Size(kernelSize, kernelSize);
		GaussianBlur(src, smooth, ksize, sigma);
		addWeighted(src, 1, smooth, -1, 0, mask);
		addWeighted(src, 1, mask, k, 0, dst);
	}
	if (colored == false) {
		imshow("myfilter", dst);
		waitKey(2000);
	}
	else
	{
		img = dst;
	}
}
void SmoothFilter(Mat& img, int kernelSize,int kernelType,bool colored=false) {
	Mat src = img.clone();
	Mat dst;
	double sigma = 0.3*((kernelSize - 1)*0.5 - 1) + 0.8;
	switch (kernelType)
	{ 
	case BLUR :
	    blur(src, dst, Size(kernelSize, kernelSize), Point(-1, -1), BORDER_DEFAULT);
	break;
	case GAUSSIAN:
		GaussianBlur(src, dst, Size(kernelSize, kernelSize), sigma); 
		break;
	default:
		break;
	}
	if (colored == false) {
		imshow("blur", dst);
		waitKey(2000);
	}
	else
	{
		img = dst;
	}
}
void SharpenFilter(Mat& img, int kernelType,bool colored=false) {
	Mat src = img.clone();
	Mat dst;
	Mat dstx, dsty;
	switch (kernelType)
	{
	case SOBEL:
		Sobel(src, dstx, -1, 1, 0, 3);
		Sobel(src, dsty, -1, 0, 1, 3);
		addWeighted(dstx, 1, dsty, 1, 0, dst);
		break;
	case LAPLACE:
		Laplacian(src, dst, -1);
		break;
	default:
		break;
	}
	if (colored == false) {
		imshow("sharpen", dst);
		waitKey(2000);
	}
	else
	{
		img =dst;
	}
}
int main(int argc, char** argv) {
	Mat imgColor = imread("C:/test3.jpg", IMREAD_COLOR);
	Mat imgGrey = imread("C:/test3.jpg", IMREAD_GRAYSCALE);
	Mat imgHsv;
	Mat dstHsv;
	Mat dstColor;
	vector<Mat> bgr;
	vector<Mat> hsv;
	cvtColor(imgColor, imgHsv, COLOR_BGR2HSV);
	imshow("src", imgGrey);
	//***smooth the grey image
	SmoothFilter(imgGrey, 3,BLUR);
	SmoothFilter(imgGrey, 5,BLUR);
	SmoothFilter(imgGrey, 9,BLUR);
	//ModifiedFilter(imgGrey, 3, GAUSSIAN);
	//***smooth the grey image by Gaussian kernel
	SmoothFilter(imgGrey, 3, GAUSSIAN);
	SmoothFilter(imgGrey, 5, GAUSSIAN);
	SmoothFilter(imgGrey, 9, GAUSSIAN);
	//***sharpen the grey image
	ModifiedFilter(imgGrey, 2, ROBERT);
	SharpenFilter(imgGrey, SOBEL);
	SharpenFilter(imgGrey, LAPLACE);
	//***highraise filter
	ModifiedFilter(imgGrey, 3, HIGHRAISE);
	//***smooth color image by normalizedBoxFilter
	split(imgHsv, hsv);
	SmoothFilter(hsv[2], 3, BLUR,true);
	merge(hsv, dstHsv);
	cvtColor(dstHsv, dstColor, COLOR_HSV2BGR);
	imshow("smooth3", dstColor);
	split(imgHsv, hsv);
	SmoothFilter(hsv[2], 5, BLUR,true);
	merge(hsv, dstHsv);
	cvtColor(dstHsv, dstColor, COLOR_HSV2BGR);
	imshow("smooth5", dstColor);
	split(imgHsv, hsv);
	SmoothFilter(hsv[2], 9, BLUR,true);
	merge(hsv, dstHsv);
	cvtColor(dstHsv, dstColor, COLOR_HSV2BGR);
	imshow("smooth9", dstColor);
	imshow("src", imgColor);
	//waitKey(0);

	//***smooth color image by GaussianFilter
	split(imgHsv, hsv);
	SmoothFilter(hsv[2], 3, GAUSSIAN,true);
	merge(hsv, dstHsv);
	cvtColor(dstHsv, dstColor, COLOR_HSV2BGR);
	imshow("smooth3", dstColor);
	split(imgHsv, hsv);
	SmoothFilter(hsv[2], 5, GAUSSIAN, true);
	merge(hsv, dstHsv);
	cvtColor(dstHsv, dstColor, COLOR_HSV2BGR);
	imshow("smooth5", dstColor);
	split(imgHsv, hsv);
	SmoothFilter(hsv[2], 9, GAUSSIAN, true);
	merge(hsv, dstHsv);
	cvtColor(dstHsv, dstColor, COLOR_HSV2BGR);
	imshow("smooth9", dstColor);
	imshow("src", imgColor);
	//waitKey(0);
	//***sharpen color 
	//*********HSV image-------sobel
	split(imgHsv,hsv);
	SharpenFilter(hsv[2], SOBEL, true);
	merge(hsv, dstHsv);
	cvtColor(dstHsv, dstColor, COLOR_HSV2BGR);
	imshow("sobel", dstColor);
	//********RGB image------sobel
	split(imgColor, bgr);
	for (int i = 0; i < 3; i++)
		SharpenFilter(bgr[i], SOBEL, true);
	merge(bgr, dstColor);
	imshow("test-sobel", dstColor);
	//********RGB image-----laplace
	split(imgColor, bgr);
	for (int i = 0; i < 3; i++)
		SharpenFilter(bgr[i], LAPLACE, true);
	merge(bgr, dstColor);
	imshow("laplace", dstColor);
	//**********HSV image-----laplace
	split(imgHsv, hsv);
	SharpenFilter(hsv[2], LAPLACE, true);
	merge(hsv, dstHsv);
	cvtColor(dstHsv, dstColor, COLOR_HSV2BGR);
	imshow("test-laplace", dstColor);
	//**********RGB image-------robert
	split(imgColor, bgr);
	for (int i = 0; i < 3; i++)
		ModifiedFilter(bgr[i], 2,ROBERT, true);
	merge(bgr, dstColor);
	imshow("robert", dstColor);
	//**********HSV image-------robert
	split(imgHsv, hsv);
	ModifiedFilter(hsv[2], 2,ROBERT, true);
	merge(hsv, dstHsv);
	cvtColor(dstHsv, dstColor, COLOR_HSV2BGR);
	imshow("test-robert", dstColor);
	waitKey(0);
	return 0;
}