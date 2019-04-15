#include<opencv2/opencv.hpp>	
#include<iostream>
using namespace std;
using namespace cv;
//This demo shows my local thresholding func to process grey level image.
//The main idea is processing the image according to the coherent histogram
Mat src;
Mat grey,greyHist;
Mat edges;
int lower_thresh=0;
int upper_thresh=99;
int kernelSize=3;
int mpType = 3;
int threshType;

void ShowNormalizedHist(Mat& img,Mat& hist) {
	int bins = 256; //the hist size
	int histSize[] = { bins };
	float range[] = { 0,256 };//the caculating limit
	const float* ranges[] = { range };
	int channels[] = { 0 };// the index of img.channel 
	calcHist(&img, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
	//imshow("histOrigin", hist);
	//double maxVal = 0;
	//minMaxLoc(hist, 0, &maxVal, 0, 0)a
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
	//imshow("input", img);
	imshow("hist", histImg);	
	//waitKey();
}
void myThreshold(int, void*) {
	uchar* p;
	Mat dst = grey.clone();
	CV_Assert(dst.type() == CV_8U);
	int rows = grey.rows;
	int cols = grey.cols;
	for (int i = 0; i < rows; i++) {
		p = dst.ptr<uchar>(i);
		for (int j = 0; j < cols; j++) {
			int tmp = p[j];
			if (tmp > upper_thresh || tmp < lower_thresh) {
				p[j] = 0;
			}
			else {
				p[j] = 255;
			}
		}
	}
	//imshow("dst", dst);
	//imwrite("localThresh1.jpg", dst);
	//dilate(dst, dst, kernel);
	//erode(dst, dst, kernel);Mat dst = grey.clone();
	Size ksize = Size(kernelSize, kernelSize);
	Mat kernel = getStructuringElement(2, ksize);
	morphologyEx(dst, dst, MORPH_CLOSE, kernel);
	imshow("dst", dst);
	Canny(dst, dst, 80, 240);
	imshow("canny", dst);
}
void myMophology(int, void*) {
	Mat dst = grey.clone();
	Size ksize = Size(kernelSize, kernelSize);
	Mat kernel = getStructuringElement(2, ksize);
	morphologyEx(dst, dst, MORPH_CLOSE, kernel);
	Canny(dst, dst, 80, 240);
	imshow("canny", dst);
}
void otsuThreshold() {
	Mat dst = grey.clone();
	threshold(dst, dst, 0, 255, THRESH_OTSU);
	imshow("otsu", dst);
	Canny(dst, dst, 80, 240);
	imshow("otsu_canny", dst);

}
int main() {
	src = imread("C:/edges11.jpg", IMREAD_COLOR);
	namedWindow("src");
	cvtColor(src, grey, COLOR_BGR2GRAY);
	//cvtColor(src, blurGrey, COLOR_BGR2GRAY);
	//GaussianBlur(blurGrey,grey , Size(kernelSize, kernelSize), 3);
	ShowNormalizedHist(grey, greyHist);
	imshow("src", src);
	namedWindow("dst");
	namedWindow("canny");
	otsuThreshold();
	createTrackbar("low_thresh", "dst", &lower_thresh, 255, myThreshold);
	createTrackbar("upp_thresh", "dst", &upper_thresh, 255, myThreshold);
	createTrackbar("kernel", "dst", &kernelSize, 25, myThreshold);
	myThreshold(0, 0);
	waitKey();
	return 0;
}