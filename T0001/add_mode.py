import cv2

grayImage = cv2.imread('../pic/QQ20171118-0.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
# Optional parameter CV_LOAD_IMAGE_COLOR(BGR),CV_LOAD_GRAYSCALE(grayscale),CV_LOAD_IMAGE_UNCHANGED(neither)
cv2.imwrite('MyPicGray.png',grayImage)

