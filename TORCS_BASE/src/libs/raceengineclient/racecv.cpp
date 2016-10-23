#include <stdlib.h>
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include "racecv.h"
using namespace cv;

/* 
 SRC indicates a pixel image we want to modify.
 gray scale, resize and vertically flip the given image and write on DST.
 Additionally, write modified image on given destination, BUF.
*/
void ReImagePreprocess(unsigned char* src, unsigned char* dst, int vw, int vh, int i)
{
  Mat srcMat(vh, vw, CV_8UC3, src);
  Mat gray;
  Mat tmp;
  Mat resized;
  cvtColor(srcMat, gray, CV_RGB2GRAY);
  resize(gray, tmp, Size(280, 210), 0, 0, CV_INTER_LINEAR);
  flip(tmp, resized, 0);
  char buf[100];
  int bufsize = 100;
  snprintf(buf, bufsize, "/home/yeonhoo/captures/capture_%d.png", i);
  imwrite(buf, resized);
  memcpy(dst, resized.data, 280*210*3);
}