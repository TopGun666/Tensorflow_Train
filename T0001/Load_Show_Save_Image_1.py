import cv2.cv as cv

# Load images
image = cv.LoadImage('../pic/QQ20171118-0.jpg', cv.CV_LOAD_IMAGE_COLOR) # Load the image
# Or just: image = cv.LoadImage('/Users/heyu/Pictures/picture/IMG_1438.JPG')

cv.NamedWindow('a_window', cv.CV_WINDOW_AUTOSIZE) # Facultative
cv.ResizeWindow('a_window', 640, 480)
cv.ShowImage('a_window', image) # show the image

# cv.SaveImage("thumb.png", thumb)
cv.WaitKey(0) # Wait for user input and quit
