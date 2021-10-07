#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat img, imgGr, imgBl,  imgCa, imgDi;
void detectCont(Mat imgDi,Mat img);

int main(/*int argc, char** argv*/)
{
    /*if (argc != 2)
    {
        cout << "Give an image to Load and display" << endl;
        return -1;
    }*/
    img = imread(/*argv[1]*/"Resources/vectalphabet.jpg", IMREAD_COLOR);
    cvtColor(img, imgGr, COLOR_BGR2GRAY);
    GaussianBlur(imgGr,imgBl,Size(3,3),3,0);
    Canny(imgBl, imgCa, 25, 75);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCa, imgDi, kernel);
    detectCont(imgDi,img);
    imshow("Contours", img);
    waitKey(0);
}
void detectCont(Mat imgDi, Mat img)
{
    vector<vector<Point>> contour;
    vector<Vec4i> hierarchy;
    findContours(imgDi, contour, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
   // drawContours(img, contour, -1, Scalar(0,255,0),3);
    vector<vector<Point>> cornPts(contour.size());
    vector<Rect> Rect(contour.size());
    
    for (int i = 0; i < contour.size(); i++)
    {
        double per = arcLength(contour[i], true);
        approxPolyDP(contour[i], cornPts[i], 0.07 * per, true);
        drawContours(img, contour, i, Scalar(0, 255, 0), 1);
        Rect[i]=boundingRect(cornPts[i]);
        rectangle(img, Rect[i].tl(), Rect[i].br(), Scalar(0, 0, 255), 1);
        
    }
}