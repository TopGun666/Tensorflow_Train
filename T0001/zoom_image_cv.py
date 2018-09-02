import cv2.cv as cv

im = cv.LoadImage("../pic/QQ20171118-0.jpg")  # get the image

thumb = cv.CreateImage((im.width / 2, im.height / 2), 8, 3)  # Create an image that is


cv.Resize(im, thumb)  # resize the original image into thumb
# cv.PyrDown(im, thumb)

cv.ShowImage('Hello World', im)  # Show the image
cv.ShowImage("thumb.png", thumb)  # Show the thumb image
cv.SaveImage("thumb.png", thumb)  # save the thumb image

cv.WaitKey(0)
