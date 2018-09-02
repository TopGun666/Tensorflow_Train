import cv2
import cv2.cv as cv
import numpy as np
import tensorflow as tf

img = cv2.imread("../pic/QQ20171118-0.jpg")

# res = cv2.resize(img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)


height, width = img.shape[:2]
res = cv2.resize(img, (2 * width, 2 * height), interpolation = cv2.INTER_CUBIC)

print img.shape[0]  # Read the first dimension length of the matrix
# The first dimension is 1920
# The second dimension is 2160
# The third dimension is 3
print res.shape[0]

cv2.imshow("zoom.png", res)
cv2.waitKey(0)

