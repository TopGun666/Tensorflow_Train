import cv2

img = cv2.imread('../pic/QQ20171118-0.jpg')
b, g, r, = cv2.split(img)
img = cv2.merge((b, g, r))

cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)
cv2.imshow("merge", img)

cv2.waitKey(0)


