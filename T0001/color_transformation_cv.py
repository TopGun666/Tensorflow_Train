#coding=utf-8
import cv2.cv as cv

im = cv.LoadImage('../pic/QQ20171118-0.jpg', cv.CV_LOAD_IMAGE_COLOR)
res = cv.CreateImage(cv.GetSize(im), cv.CV_8UC2, 3)  # cv.CV_32F, cv.IPL_DEpTH_16S,...

cv.Convert(im, res)
cv.ShowImage("Converted", res)
res2 = cv.CreateImage(cv.GetSize(im), cv.CV_8UC2, 3)
cv.CvtColor(im, res2, cv.CV_RGB2BGR) # HLS, HSV,YCrCb, ...
cv.ShowImage("CvtColor", res2)
cv.WaitKey(0)
'''
cv.Convert():#将图片从一个颜色空间转到另一个颜色空间
cv.CvtColor(src, dst, code)：
cv2:
cv2.cvtColor(input_image, flag) # 函数实现图片颜色空间的转换，flag 参数决定变换类型。如 BGR->Gray flag 就可以设置为 cv2.COLOR_BGR2GRAY 。
'''

