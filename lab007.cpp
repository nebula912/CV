#include<iostream>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
using namespace cv;
using namespace std;
static const char* testImage="C:/Hough-line-test.jpg";
static Mat src,dst;
static Mat edges;
static Mat colorEdges,colorEdgesP;
int thresholdHough;
double threshold1,threshold2;
//************This demo shows the effect of Hough-Line Transform.
int main(int argc, char* argv[]){
     src=imread(testImage,IMREAD_GRAYSCALE);
     //detect edge by Canny 
     Canny(src,edges,threshold1,threshold2,3);
     cvtColor(edges,colorEdges,COLOR_GRAY2BGR);
     colorEdgesP=colorEdges.clone();
     //Standard Hough Line Transform
     vector<Vec2f> lines;
     HoughLines(edges,lines,1,CV_PI/180,thresholdHough);
     //Draw the lines
     for(size_t i=0;i<lines.size();i++){
         
     }

}
