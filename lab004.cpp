#include<random>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<vector>
using namespace cv;
using namespace std;
static Mat imgColor;
static Mat imgGrey;
enum filterType
{
	ARITHMETIC,
	GEOMETRIC,
	HARMONIC,
	INVERSE_HARMONIC,
	MEDIAN,
	ADAPTIVE_MEAN_VALUE,
	ADAPTIVE_MEDIAN,
};
enum noiseType {
	GAUSS,
	PEPPER,
	SALT,
	PEPPER_SALT
};
void innerSort(float* &arr, int n) {
	for (int i = n - 1; i > 0; i--) {
		for (int j = 0; j < i; j++) {
			if (arr[j] > arr[j + 1]) {
				int temp;
				temp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = temp;
			}
		}
	}
}
double filterx(Mat src,int smax=7) {
	double res = 0;
	double zmin = 0, zmax =0;
	double zmed,zxy;
	int subsize = 3;
	int x = (src.rows - 1) / 2;
	int y = (src.cols - 1) / 2;
	Point xy = Point(x, y);
	uchar* p;
	zxy = src.at<uchar>(xy);
		while (subsize <= smax) {
			int n = subsize*subsize;
			float* arr = new float[n];
			int win = (subsize - 1) / 2;
			int index = 0;
			for (int i = y-win; i <= y+win; i++) {
				p = src.ptr<uchar>(i);
				for (int j = x-win; j <= x+win; j++) {
					arr[index++] = p[j];
				}
			}
			innerSort(arr, n);
			zmin = arr[0];
			zmax = arr[n - 1];
			zmed = arr[(n - 1) / 2];
			delete[] arr;
			if (zmed - zmin > 0 && zmed - zmax < 0) {
				if (zxy - zmin > 0 && zxy - zmax < 0)res = zxy;
				else res = zmed;
				break;
			}
			else
			{
				subsize+=2;
				if (subsize > smax) {
					res = zmed;
				}
			}
		}
		return res;
}
//***ratio for pepper/salt noise,should be (0,1) mu & sigma for Gaussian noise
//************Mersenne twister method to generate random number
void addNoise(Mat& img, int noiseType, double ratio=0.1, double mu = 0, double sigma = 32) {
	//accept only char type matrices
	CV_Assert(img.depth() == CV_8U);
	imshow("origin", img);
	//check the parameter
	ratio = ratio < 0 ? 0 : ratio;
	ratio = ratio > 1 ? 1 : ratio;
	int channels = img.channels();
	int nRows = img.rows;
	int nCols = img.cols*channels;
	int num = img.rows*img.cols*ratio;
	uchar* p;
	//random generator
	random_device rd;
	mt19937 gen(rd());
	normal_distribution<> GaussRand(mu, sigma);
	switch (noiseType)
	{
	case GAUSS:
		for ( int i = 0; i < nRows; ++i) {
			p = img.ptr<uchar>(i);
			for (int j = 0; j < nCols; ++j) {
				auto tmp= saturate_cast<uchar>(p[j] + GaussRand(gen));
				p[j] = tmp;
			}
		}
		imshow("Gauss_Noise", img);
		break;
	case PEPPER:
		nCols /= channels;
		for (int i = 0; i < num; i++) {
			int row = (int)(gen() % nRows);
			int col = channels*(int)(gen() % nCols);
			p = img.ptr<uchar>(row);
			for (int j = 0; j < channels; j++) {
				*(p + col + j) = 0;
			}
	    }
		imshow("Pepper_Noise", img);
		break;
	case SALT:	
		nCols /= channels;
		for (int i = 0; i < num; i++) {
			int row = (int)(gen() % nRows);
			int col = channels*(int)(gen() % nCols);
			p = img.ptr<uchar>(row);
			for (int j = 0; j < channels; j++) {
				*(p + col + j) = 255;
			}
	    }
		imshow("Salt_Noise", img);
		break;
	case PEPPER_SALT:	
		nCols /= channels;
		for (int i = 0; i < num/2; i++) {
			int row = (int)(gen() % nRows);
			int col = channels*(int)(gen() % nCols);
			p = img.ptr<uchar>(row);
			for (int j = 0; j < channels; j++) {
				*(p + col + j) = 255;
			}
	    }
		for (int i = 0; i < num/2; i++) {
			int row = (int)(gen() % nRows);
			int col =channels* (int)(gen() % nCols);
			p = img.ptr<uchar>(row);
			for (int j = 0; j < channels; j++) {
				*(p + col + j) = 0;
			}
	    }
		imshow("Pepper-Salt_Noise", img);
		break;
	default:
		break;
	}
	
}
//***filter for denoising,return the value at
double filter(Mat src, int filterType, double q=-1.5,int smax=7) {
	CV_Assert(src.depth() == CV_8U);
	double res=0;
	int rows = src.rows;
	int cols = src.cols;
	double sum1 = 0;
	double sum2 = 0;
	double mn = rows * cols;

	//for adaptive mean-value filter
	double sigmaL = 0;
	double ml = 0;
	uchar* p;
	switch (filterType)
	{
	case ARITHMETIC:
		res = 0;	
		for (int i = 0; i < rows; i++)
		{
			p = src.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
			     res += p[j];
			}
		}
		res /= mn;
		break;
	case GEOMETRIC:
		res = 1;
		for (int i = 0; i < rows; i++)
		{
			p = src.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				if (p[j] != 0)res *= pow(p[j],1.0/mn);
			}
		}
		//double power = 1.0/(rows*cols);
		break;
	case HARMONIC:
		res = 0;
		for (int i = 0; i < rows; i++) {
			p = src.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				if (p[j] != 0) res += 1.0/ p[j];
			}
		}
		res = mn / res;
		break;
	case INVERSE_HARMONIC:
		for (int i = 0; i < rows; i++) {
			p = src.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				if (p[j] != 0) {
					sum1 += pow(p[j], q + 1);
					sum2 += pow(p[j], q);
				}
			}
		}
		res = sum1 / sum2;
		break;

	case ADAPTIVE_MEAN_VALUE:
		//get local mean value
		for (int i = 0; i < rows; i++)
		{
			p = src.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				ml+= p[j];
			}
		}
		ml /= mn;
		//get the local square variance
		for (int i = 0; i < rows; i++)
		{
			p = src.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				sigmaL+= pow((p[j]-ml),2);
			}
		}
		sigmaL /= mn;
		res = sigmaL;
		break;
	default:
		break;
	}

	return res;
}
//***q for inverse harmonic wave filter
void denoise(Mat& src, int filterType, int kernelSize = 5, double q = -1.5, double sigmaN = 32, bool colored = false) {
	Mat img_buf ;
	if (colored == true) img_buf = src;
	else { img_buf = src.clone(); }
	int rows = src.rows;
	int cols = src.cols;
	Mat dst(rows,cols,CV_8U);
	int h = (kernelSize - 1) / 2;
	int w = (kernelSize - 1) / 2;
	//handle the border
	Mat img(rows + 2 * h, cols + 2 * w, CV_8U);
	copyMakeBorder(img_buf, img, h, h, w, w, BORDER_REFLECT);
	//for adaptive mean value
	double ml, sigmaL;
	uchar* p;
	Mat kernel;
	Size ksize(kernelSize, kernelSize);
	switch (filterType)
	{
	case ARITHMETIC:
		if (colored) {
			blur(img_buf, src, ksize);
			break;
		}
		blur(img, dst, ksize);
		imshow("arithmetic", dst);
		break;
	case GEOMETRIC:
		for (int i = 0; i < rows - 0; i++) {
			p = img.ptr<uchar>(i);
			for (int j = 0; j < cols - 0; j++) {
				kernel = img(Rect(j - 0, i - 0, kernelSize, kernelSize));
				p[j] = saturate_cast<uchar>(filter(kernel,GEOMETRIC));
			}
		}
		copyMakeBorder(img(Rect(h,w,rows-h,cols-w)), dst, 0, 0, 0, 0, BORDER_DEFAULT);
		if (colored) { src = dst; break; }
		imshow("geometric", dst);
		break;
	case MEDIAN:
		medianBlur(img, dst, kernelSize);
		imshow("median", dst);
		break;
	case HARMONIC:
		for (int i = 0; i < rows - 0; i++) {
			p = img.ptr<uchar>(i);
			for (int j = 0; j < cols - 0; j++) {
				kernel = img(Rect(j , i , kernelSize, kernelSize));
				p[j] = saturate_cast<uchar>(filter(kernel,HARMONIC));
			}
		}
		copyMakeBorder(img_buf(Rect(h,w,rows-h,cols-w)), dst, 0, 0, 0, 0, BORDER_DEFAULT);
		imshow("harmonic", dst);
		break;
	case INVERSE_HARMONIC:
			for (int i = 0; i < rows - 0; i++) {
			p = img.ptr<uchar>(i);
			for (int j = 0; j < cols - 0; j++) {
				kernel = img(Rect(j - 0, i - 0, kernelSize, kernelSize));
				p[j] = saturate_cast<uchar>(filter(kernel,INVERSE_HARMONIC,q));
			}
		}
		copyMakeBorder(img(Rect(h,w,rows-h,cols-w)), dst, 0, 0, 0, 0, BORDER_DEFAULT);
		imshow("inverse", dst);
		break;
	case ADAPTIVE_MEAN_VALUE:
		for (int i = 0; i < rows - 0; i++) {
			p = img.ptr<uchar>(i);
			for (int j = 0; j < cols - 0; j++) {
				kernel = img(Rect(j - 0, i - 0, kernelSize, kernelSize));
				ml = filter(kernel, ARITHMETIC);
				sigmaL = filter(kernel, ADAPTIVE_MEAN_VALUE);
				//cout << ml << "," << sigmaL << endl;
				auto ratio = pow(sigmaN, 2) / sigmaL;
				ratio = ratio > 1 ? 1: ratio;
				double temp = p[j] - ratio*((double)p[j] - ml);
				p[j] = saturate_cast<uchar>(temp);
			}
		}
		copyMakeBorder(img(Rect(h,w,rows-h,cols-w)), dst, 0, 0, 0, 0, BORDER_DEFAULT);
		imshow("adaptive mean value", dst);
		break;
	case ADAPTIVE_MEDIAN:	
		for (int i =0 ; i < rows - 0; i++) {
			p = img.ptr<uchar>(i);
			for (int j = 0; j < cols - 0; j++) {
				kernel = img(Rect(j - 0, i - 0, kernelSize, kernelSize));
				p[j] = saturate_cast<uchar>(filterx(kernel,7));
			}
		}
		copyMakeBorder(img(Rect(h,w,rows-h,cols-w)), dst, 0, 0, 0, 0, BORDER_DEFAULT);
		imshow("adaptive-median", dst);
		break;
	default:
		break;
	}
}
int main(int argc, char* argv[]) {
	imgColor = imread("c:/test3.jpg", IMREAD_COLOR);
	imgGrey = imread("C:/test3.jpg", IMREAD_GRAYSCALE);
	//********mean-value processing with grey image
	for (int i = 0; i < 4; i++) {
		Mat noise = imgGrey.clone();
	    addNoise(noise, i);
		for (int j = 0; j < 4; j++) {
			denoise(noise, j);
		}
		waitKey();
	}
	//*********median processing with grey image
	for (int i = 0; i < 4; i++) {
		Mat noise = imgGrey.clone();
		addNoise(noise, i);
		denoise(noise, MEDIAN, 5);
		denoise(noise, MEDIAN, 9);
		waitKey();
	}
	//********adaptive processing with grey image
	for (int i = 0; i < 4; i++) {
		Mat noise = imgGrey.clone();
		addNoise(noise, i);
		denoise(noise, ADAPTIVE_MEAN_VALUE, 7);
		denoise(noise, ADAPTIVE_MEDIAN, 7);
		waitKey();
	}
	//********mean-value processing with color image
	//***********NOTE THAT:the noise of the polluted RGB image£¨even in one channel£©
	//*********************will diffuse in all the channels of the transformed HSV image
	for (int i = 0; i < 4; i++) {
		vector<Mat> bgr;
		Mat noiseColor = imgColor.clone();
		Mat color;
		addNoise(noiseColor, i);
		split(noiseColor, bgr);
		for (int j = 0; j < 3; j++) {
			denoise(bgr[j], GEOMETRIC, 5, 1, 32, true);
		}
		merge(bgr, color);
		imshow("denoise_color", color);
		waitKey();
	}	for (int i = 0; i < 4; i++) {
		vector<Mat> bgr;
		Mat noiseColor = imgColor.clone();
		Mat color;
		addNoise(noiseColor, i);
		split(noiseColor, bgr);
		for (int j = 0; j < 3; j++) {
			denoise(bgr[j], ARITHMETIC, 5, 1, 32, true);
		}
		merge(bgr, color);
		imshow("denoise_color", color);
		waitKey();
	}
	
	waitKey(0);
	return 0;
}