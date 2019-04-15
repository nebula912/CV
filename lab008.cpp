#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
//This demo shows the mophologic operations in image-processing
Mat src,erode_dst,dilate_dst;
Mat open_dst,close_dst;
int kernerSize=3;
int kernelType=1;//one for cross type
const int maxType=2;
const int maxKernelSize=21;
char* erodeWin="erode opreation";
char* dilateWin="dilate opreation";
char* filename="C:/canny2.jpg";
void erodeOperation(int ,void*){
     Size ksize=Size(kernerSize,kernerSize);
     Mat kernel=getStructuringElement(kernelType,ksize);
     erode(src,erode_dst,kernel);
     imshow(erodeWin,erode_dst);

}
void dilateOperation(int ,void*){
     Size ksize=Size(kernerSize,kernerSize);
     Mat kernel=getStructuringElement(kernelType,ksize);
     dilate(src,dilate_dst,kernel);
     imshow(dilateWin,dilate_dst);
}
void openOperation(int ,void*){

}
int main(int argc,char* argv[]){
    src=imread(filename,IMREAD_GRAYSCALE);
    namedWindow(erodeWin);
    namedWindow(dilateWin);
    createTrackbar("kernelType",erodeWin,&kernelType,maxType,erodeOperation);
    createTrackbar("kernelSize",erodeWin,&kernerSize,maxKernelSize,erodeOperation);
    createTrackbar("kernelType",dilateWin,&kernelType,maxType,dilateOperation);
    createTrackbar("kernelSize",dilateWin,&kernerSize,maxKernelSize,dilateOperation);
    erodeOperation(0,0);
    dilateOperation(0,0);
    waitKey();
    return 0;
    

}
