import cv2

img = cv2.imread('../pic/QQ20171118-0.jpg', 0)

rows, cols = img.shape

M = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow("Img_Rotation", dst)
cv2.waitKey(0)
