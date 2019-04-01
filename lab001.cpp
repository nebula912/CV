#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<math.h>
//using namespace std;
#define BINARY_PARAMETER 128
#define LOG_PARAMETER 1
#define GAMMA_PARAMETER 0.5


using namespace cv;
int mylog(int r) {
	double c = 1.0;
	double v=r/255.0;
	double s = c * log(1 + v);
	int d = (int)(s * 255);
	d = d > 255 ? 255 : d;
	d = d < 0 ? 0 : d;
	return d;
}
int myexp(int r) {
	double c = 1.0;
	double v = r / 255.0;
	double s = c * pow(v, GAMMA_PARAMETER );
	int d = (int)(s * 255);
	d = d > 255 ? 255 : d;
	d = d < 0 ? 0 : d;
	return d;

}
int main(int argc, char** argv) {
	Mat img=imread("c:/test1.jpg",1);
	namedWindow("src");
	imshow("src", img);
	Mat* grayPtr= new Mat();
	Mat& Gray=*grayPtr;
	Mat binary;
	Mat gamma;
	Mat log;
	Mat invert;
	cvtColor(img, Gray, COLOR_BGR2GRAY);
	namedWindow("gray");
	imshow("gray", Gray);
	waitKey(2000);
	destroyWindow("gray");
	int width = Gray.rows;
	int height = Gray.cols;
	int type = CV_8U;
	gamma = Gray.clone();
	Mat lookUpTable(1, 256,type);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i) {
		p[i] = saturate_cast<uchar>(pow(i / 255.0, 0.2)*255.0);
	}
	LUT(Gray, lookUpTable, gamma);
	imshow("gamma", gamma);
	waitKey(-1);
	invert.create(width, height, img.type());
	binary.create(width, height, Gray.type());
	log.create(width, height, type);
	gamma.create(width, height, type);
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			int d = Gray.at<uchar>(i,j);
			binary.at<uchar>(i, j) = d>BINARY_PARAMETER?255:0;
			log.at<uchar>(i, j) = mylog(d);
			//gamma.at<uchar>(i, j) = myexp(d);
			uchar b = img.at<Vec3b>(i, j)[0];
			uchar g = img.at<Vec3b>(i, j)[1];
			uchar r = img.at<Vec3b>(i, j)[2];
			invert.at<Vec3b>(i, j)[0] =255- b;
			invert.at<Vec3b>(i, j)[1] =255- g;
			invert.at<Vec3b>(i, j)[2] =255- r;

		}
	}
	imshow("gray", binary);
	waitKey(2000);
	destroyWindow("gray");
	imshow("gray", log);
	waitKey(2000);
	destroyWindow("gray");
	imshow("gray", gamma);
	waitKey(2000);
	destroyWindow("gray");
	imshow("src", invert);
	
	waitKey(0);
	destroyAllWindows();
	return 0;

}