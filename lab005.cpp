#include<opencv.hpp>
#include<math.h>
using namespace std;
using namespace cv;
static Mat img;
static Mat imgDFT;
static Mat imgIDFT;
static Mat imgMag;
static int d0;
static int n;

//
Mat GetDFTImage(Mat& src) {
    Mat expand;
	Size dftSize;
	int width = src.rows;
	int height = src.cols;
	dftSize.width = getOptimalDFTSize(width);
	cout << dftSize.width << "," << src.rows << endl;
	dftSize.height = getOptimalDFTSize(height);
	//expand the origin image with zero adding to the border
	//copyMakeBorder copies the src image to the center
	copyMakeBorder(src, expand,
		0, dftSize.width - width,
		0, dftSize.height - height, 
		BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(expand),Mat::zeros(expand.size(),CV_32F) };
	Mat imgDFT;
	merge(planes, 2, imgDFT);
	dft(imgDFT, imgDFT);
	// -2 means 11111110 
	imgDFT = imgDFT(Rect(0, 0, imgDFT.cols & -2, imgDFT.rows & -2));
	int cx = imgDFT.cols / 2;
	int cy = imgDFT.rows / 2;
	Mat q0(imgDFT, Rect(0, 0, cx, cy));
    Mat q1(imgDFT, Rect(cx, 0, cx, cy));
	Mat q2(imgDFT, Rect(0, cy, cx, cy));
	Mat q3(imgDFT, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
    
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	return imgDFT;

}
Mat GetMagnitudeImg(Mat& src) {
	CV_Assert(src.channels() == 2);
	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	Mat img = src.clone();
	Mat planes[2];
	split(img, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magImg = planes[0];
	magImg += Scalar::all(1);
	log(magImg, magImg);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);
	return magImg;
}
Mat GetIDFTImage(Mat& src) {
	CV_Assert(src.channels() == 2);
	src = src(Rect(0, 0, src.cols & -2, src.rows & -2));
	int cx = src.cols / 2;
	int cy = src.rows / 2;
	Mat img = src;
	Mat q0(img, Rect(0, 0, cx, cy));
    Mat q1(img, Rect(cx, 0, cx, cy));
	Mat q2(img, Rect(0, cy, cx, cy));
	Mat q3(img, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
    
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	idft(img, img, DFT_SCALE);//DFT_REAL_OUTPUT could be ok for conjugate img.

	img.convertTo(img, CV_8U);	
	Mat planes[2];
	split(img, planes);
	Mat Re = planes[0];
	return Re;
}
Mat IdealHPfilter(Mat& src,double d0=20) {
	CV_Assert(src.channels() == 2);
	Mat planes[2];
	Mat imgDFT;
	split(src, planes);
	Mat Re = planes[0];
	Mat Im = planes[1];
	double u0 = src.cols / 2;
	double v0 = src.rows / 2;
	for (int i = 0; i < src.rows; i++) {
		float* p1 = Re.ptr<float>(i);
		float* p2 = Im.ptr<float>(i);
		double v = i - v0;
		for (int j = 0; j < src.cols; j++) {
			double u = j - u0;
			double d = sqrt(pow(u, 2) + pow(v, 2));
			if (d < d0) {
				p1[j] = 0;
				p2[j] = 0;
			}
		}
	}
	merge(planes, 2, imgDFT);
	return imgDFT;

}
Mat IdealLPfilter(Mat& src,double d0=20) {
    CV_Assert(src.channels() == 2);
	Mat planes[2];
	Mat imgDFT;
	split(src, planes);
	Mat Re = planes[0];
	Mat Im = planes[1];
	double u0 = src.cols / 2;
	double v0 = src.rows / 2;
	for (int i = 0; i < src.rows; i++) {
		float* p1 = Re.ptr<float>(i);
		float* p2 = Im.ptr<float>(i);
		double v = i - v0;
		for (int j = 0; j < src.cols; j++) {
			double u = j - u0;
			double d = sqrt(pow(u, 2) + pow(v, 2));
			if (d >= d0) {
				p1[j] = 0;
				p2[j] = 0;
			}
		}
	}
	merge(planes, 2, imgDFT);
	return imgDFT;
}
Mat BWLPfilter(Mat& src, double d0 = 10, int n = 1) {
	CV_Assert(src.channels() == 2);
	Mat planes[2];
	Mat imgDFT;
	split(src, planes);
	Mat Re = planes[0];
	Mat Im = planes[1];
	double u0 = src.cols / 2;
	double v0 = src.rows / 2;
	for (int i = 0; i < src.rows; i++) {
		float* p1 = Re.ptr<float>(i);
		float* p2 = Im.ptr<float>(i);
		double v = i - v0;
		for (int j = 0; j < src.cols; j++) {
			double u = j - u0;
			double d = sqrt(pow(u, 2) + pow(v, 2));
			p1[j] /= (1 + pow(d / d0, 2 * n));
			p2[j] /= (1 + pow(d / d0, 2 * n));
		}
	}
	merge(planes, 2, imgDFT);
	return imgDFT;
}
Mat BWHPfilter(Mat& src, double d0 = 10, int n = 1) {
	CV_Assert(src.channels() == 2);
	Mat planes[2];
	Mat imgDFT;
	split(src, planes);
	Mat Re = planes[0];
	Mat Im = planes[1];
	double u0 = src.cols / 2;
	double v0 = src.rows / 2;
	for (int i = 0; i < src.rows; i++) {
		float* p1 = Re.ptr<float>(i);
		float* p2 = Im.ptr<float>(i);
		double v = i - v0;
		for (int j = 0; j < src.cols; j++) {
			double u = j - u0;
			double d = sqrt(pow(u, 2) + pow(v, 2));
			p1[j] /= (1 + pow(d0 / d, 2 * n));
			p2[j] /= (1 + pow(d0 / d, 2 * n));
		}
	}
	merge(planes, 2, imgDFT);
	return imgDFT;
}
//Callback func of TrackBar
static void IdealHP(int, void*) {
     imgDFT = GetDFTImage(img);
	 Mat imgHP = IdealHPfilter(imgDFT,d0);
     imgMag = GetMagnitudeImg(imgHP);
	 imshow("spectrum", imgMag);
	 imgIDFT = GetIDFTImage(imgHP);
	 imshow("idealHP", imgIDFT);
}
static void IdealLP(int, void*) {
     imgDFT = GetDFTImage(img);
	 Mat imgLP = IdealLPfilter(imgDFT,d0);
     imgMag = GetMagnitudeImg(imgLP);
	 imshow("spectrum", imgMag);
	 imgIDFT = GetIDFTImage(imgLP);
	 imshow("idealLP", imgIDFT);
}
static void ButterworthLP(int, void*) {
     imgDFT = GetDFTImage(img);
	 Mat imgLP = BWLPfilter(imgDFT, d0,n);
     imgMag = GetMagnitudeImg(imgLP);
	 imshow("spectrum", imgMag);
	 imgIDFT = GetIDFTImage(imgLP);
	 imshow("ButterworthLP", imgIDFT);
}
static void ButterworthHP(int, void*) {
     imgDFT = GetDFTImage(img);
	 Mat imgHP = BWHPfilter(imgDFT,d0,n);
     imgMag = GetMagnitudeImg(imgHP);
	 imshow("spectrum", imgMag);
	 imgIDFT = GetIDFTImage(imgHP);
	 imshow("ButterworthHP", imgIDFT);
}
//This project builds filters in frequency domain, using DFT and IDFT functions in opencv
int main(int argc, char* argv[]) {
    img = imread("c:/test3.jpg", IMREAD_GRAYSCALE);
	namedWindow("idealHP");
	namedWindow("idealLP");
	namedWindow("ButterworthHP");
	namedWindow("ButterworthLP");
	d0 = 10;
	n = 1;
	createTrackbar("D0", "idealHP", &d0, 500,IdealHP,0);
	createTrackbar("D0", "idealLP", &d0, 100,IdealLP,0);
	createTrackbar("D0", "ButterworthHP", &d0, 500, ButterworthHP, 0);
	createTrackbar("n", "ButterworthHP", &n, 100, ButterworthHP, 0);
	createTrackbar("D0", "ButterworthLP", &d0, 100, ButterworthLP, 0);
	createTrackbar("n", "ButterworthLP", &n, 100, ButterworthLP, 0);
	imshow("src", img);
	waitKey(0);
	return 0;
}
