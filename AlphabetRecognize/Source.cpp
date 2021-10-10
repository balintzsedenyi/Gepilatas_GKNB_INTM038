#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;
Mat detectCont(Mat,Mat*);
Mat preprocess(Mat);
void Cascade(CascadeClassifier,Mat);
int main(/*int argc, char** argv*/)
{
    /*if (argc != 2)
{
    cout << "Give an image to load and display" << endl;
    return -1;
}*/
    Mat imgCl;
    Mat img = imread(/*argv[1]*/"Resources/J.png", IMREAD_COLOR);
    CascadeClassifier J;
    if (!J.load("J.xml"))
    {
        cout << "--(!)Error loading cascade\n";
        return -1;
    }
    imgCl=preprocess(img);
    imshow("Crop", imgCl);
    imshow("noCrop", img);
    Cascade(J, imgCl);
    waitKey(0);
    return 0;
}

Mat preprocess(Mat img)
{
    Mat imgGr, imgBl, imgCa, imgDi, imgRe;
    cvtColor(img, imgGr, COLOR_BGR2GRAY);
    GaussianBlur(imgGr, imgBl, Size(3, 3), 3, 0);
    Canny(imgBl, imgCa, 25, 75);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCa, imgDi, kernel);
    imgRe=detectCont(img,&imgDi);
    return imgRe;
}

Mat detectCont(Mat img, Mat* imgDi)
{
    vector<vector<Point>> contour;
    vector<Vec4i> hierarchy;
    findContours(*imgDi, contour, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> cornPts(contour.size());
    vector<Rect> boundRect(contour.size());
    
    for (int i = 0; i < contour.size(); i++)
    {
        double per = arcLength(contour[i], true);
        approxPolyDP(contour[i], cornPts[i], 0.07 * per, true);
        //drawContours(img, contour, i, Scalar(0, 255, 0), 1);
        boundRect[i]=boundingRect(cornPts[i]);
        //rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 1);
    }
    Mat imgCrop;
    imshow("Contour", img);
    imgCrop = img(boundRect[0]);
    return imgCrop;
    
}

void Cascade(CascadeClassifier J, Mat img)
{
    Mat imgGr;
    cvtColor(img, imgGr, COLOR_BGR2GRAY);
    resize(imgGr, imgGr, Size(100, 100));
    resize(img, img, Size(100, 100));
    std::vector<Rect> Js;
    J.detectMultiScale(imgGr, Js, 1.1, 0, 1,Size(70,70));
    for (int i = 0; i < Js.size(); i++)
    {
        rectangle(img, Js[i].tl(), Js[i].br(), Scalar(0, 255, 0));
    }
    imshow("detect", img);
}