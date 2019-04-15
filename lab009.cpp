#include<opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
//This demo shows extracting shaped words in a picture.
enum denoiseType{
    GAUSSIAN,
    ADAPTIVE_MEDIAN
};
enum threshType{
     THRESH_BINARY,
     THRESH_BINARY_INV,
     THRESH_TRIANGLE=16,
};
Mat src,dst;
Mat edges,edges_inv;
Mat srcGrey;
Mat hist,histImg;
char file[]="c:/edges22.jpg";
char srcwin[]="srcImg";
char dstwin[]="finalImg";
int sigma=3;
int thresh=50;
int lowerThresh=10;
int upperThresh=100;
int kernelSize=3;
int erodeShape=2;
int index;
int threshType;
Size ksize=Size(kernelSize,kernelSize);

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
static void denoise(Mat& src,int denoiseType,int kernelsize,double sigmaN=32){
       int rows=src.rows;
       int cols=src.cols;
       int h = (kernelSize-1)/2;
       int w = (kernelSize-1)/2;
       Mat img_buf(rows+2*h,cols+2*w,CV_8U);
       copyMakeBorder(src,img_buf,h,h,w,w,BORDER_REFLECT);
       uchar* p;
       Mat kernel;
       Size ksize_in=Size(kernelsize,kernelsize);
       switch (denoiseType)
       {
           case GAUSSIAN:
                GaussianBlur(img_buf,src,ksize_in,sigma);
               break;
           case ADAPTIVE_MEDIAN:
              for (int i =0 ; i < rows - 0; i++) {
			         p = img_buf.ptr<uchar>(i);
			       for (int j = 0; j < cols - 0; j++) {
				        kernel = img_buf(Rect(j - 0, i - 0, kernelSize, kernelSize));
				        p[j] = saturate_cast<uchar>(filterx(kernel,7));
			}
		}
		copyMakeBorder(img_buf(Rect(h,w,rows-h,cols-w)), src, 0, 0, 0, 0, BORDER_DEFAULT);
	   	imshow("adaptive-median", dst);
               break;
           default:
               break;
       }
}
static void myThreshold(Mat src,Mat dst,double thresh,double maxVal,int threshType){

}
static void FindCharacters(int ,void*){
     GaussianBlur(srcGrey,edges,ksize,sigma);
     threshold(edges,edges,thresh,255,threshType);
     
     Canny(edges,edges,lowerThresh,upperThresh);
     dst = Scalar::all(0);
     src.copyTo(dst,edges);
     imshow("edges",edges);
     //threshold(edges,edges_inv,10,255,THRESH_BINARY_INV);
     imshow("edges_inv",edges_inv);
     erode(edges_inv,edges_inv,getStructuringElement(erodeShape,ksize));
     imshow("edges_inv",edges_inv);
     imshow(dstwin,dst);
}
void showNormalizedHist(Mat& img,Mat& hist){
     int bins=256;
     int histSize[]={bins};
     float range[]={0,256};
     const float* ranges[]={range};
     int channels[]={0};
     calcHist(&img,1,channels,Mat(),hist,1,histSize,ranges);
     Mat histTmp=hist.clone();
     int hist_h=256,hist_w=512;
     int bin_w=cvRound((double)hist_w/histSize[0]);
     Mat histImg(hist_h,hist_w,CV_8UC3,Scalar(0,0,0));
     normalize(histTmp,histTmp,0,histImg.rows,NORM_MINMAX,-1,Mat());
     int GreyPeak=0;
     for(int i=1;i<histSize[0];i++){
         int y=cvRound(histTmp.at<float>(i));
         if(GreyPeak<y){
             index=i;
             GreyPeak=y;
         }
         rectangle(histImg,Point(bin_w*(i-1),hist_h),Point(bin_w*i,hist_h-y),Scalar(255,0,0),1,8,0);
     }
     imshow("hist",histImg);
}
int main(int argc,char* argv[]){
    src = imread(file,IMREAD_COLOR);
    imshow(srcwin,src);
    namedWindow(dstwin);
    cvtColor(src,srcGrey,COLOR_BGR2GRAY);
    dst.create(src.size(),src.type());
    //createTrackbar("lowerThresh",dstwin,&lowerThresh,150,FindCharacters);
    //createTrackbar("upperThresh",dstwin,&upperThresh,500,FindCharacters);
    createTrackbar("BlurKsize",dstwin,&kernelSize,21,FindCharacters);
    FindCharacters(0,0);
    
    waitKey();
    return 0;
}