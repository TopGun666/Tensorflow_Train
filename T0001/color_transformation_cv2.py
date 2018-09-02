import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Read each frame of the video
    _, frame = cap.read()

    # Convert pictures from BGR space to HSV space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the blue range in the HSV space
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Get the blue part according to the blue threshold defined above
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask= mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k ==27:
        break

cv2.destroyAllWindows()

