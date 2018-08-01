#include <vector>
#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

Mat create_hue_mask(Mat image, InputArray lower_color, InputArray upper_color, Size kernel_size)
{
    Mat mask;
    
    // filter
    inRange(image, lower_color, upper_color, mask);

    // close operation, to fill the inside holes
    Mat erosion_dst, dilation_dst;
    Mat kernel = getStructuringElement(MORPH_RECT, kernel_size); // each element
    dilate(mask, dilation_dst, kernel);
    erode(dilation_dst, erosion_dst, kernel);

    return erosion_dst;
}

int main(int argc, char** argv)
{
    Mat bgrImage = imread("../pic/333.jpg");

    Mat hsvImage;
    cvtColor(bgrImage, hsvImage, COLOR_BGR2HSV);

    // set color filter
    int low_H = 25;
    int low_S = 127;
    int low_V = 80;
    int high_H = 31;
    int high_S = 255;
    int high_V = 230;

    // set the dialate & erode element size
    Size kernel_size = Size(10, 10);

    // set the minimum width of the labels (pixel)
    int label_width = 100;

    // get the label mask
    Mat labelImg = create_hue_mask(hsvImage, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), kernel_size);

    //****** get contours ******/
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(labelImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point());
    cout << "contours num: " << contours.size() << endl;

    //****** get bounding rectangulars of each contour 
    //       and calculate the mean x (column) value
    //       of the desired rectangulars ******/
    Rect boundRect;
    int rec_x = 0;
    int num = 0;

    for(int i=0; i<contours.size(); i++)
    {
        boundRect = boundingRect(Mat(contours[i]));
        if(boundRect.width > label_width)
        {
            rec_x += 2 * boundRect.x + boundRect.width;
            num ++;
        }
    }

    float x_mean = float(rec_x) / (float)num / 2.0;
    cout << x_mean << endl;

    return 0;
}
