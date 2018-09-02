import numpy as np
import cv2

img = cv2.imread('../pic/QQ20171118-0.jpg', 1)
img_res = cv2.resize
cv2.imshow('image', img)
k = cv2.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv2.destroyALLWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png', img)
    cv2.destroyALLWindows()
