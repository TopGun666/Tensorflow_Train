import cv2.cv as cv
import cv2

image = cv.LoadImage('../pic/QQ20171118-0.jpg', cv.CV_LOAD_IMAGE_COLOR) # load the image
frame = cv.LoadImage('../pic/QQ20171118-0.jpg', cv.CV_LOAD_IMAGE_COLOR) # load the image

font = cv.InitFont(cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8) # Creates a font

y = image.height / 2 # y position of the text
x = image.width /4 # x position of the text

cv.PutText(image, "Hello World !", (x, y), font, cv.RGB(255, 255, 255)) # Draw the text
cv2.putText(frame, 'Hello World', (300, 100), 0, 0.5, (0, 0, 255),2)

cv.ShowImage('Hello World', image) # Show the image

cv.WaitKey(0)
