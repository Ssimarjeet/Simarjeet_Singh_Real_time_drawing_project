import cv2
import numpy as np

# Mouse click event handler to get RGB values
def mouseRGB(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        colorB = frame[y, x, 0]
        colorG = frame[y, x, 1]
        colorR = frame[y, x, 2]
        print('BGR values at ({}, {}): {}, {}, {}'.format(x, y, colorB, colorG, colorR))

# Create named window and set mouse callback
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouseRGB)

# Create the video capture object for phone camera (assuming it's the default camera, index 0)
vid = cv2.VideoCapture(0)

if not vid.isOpened():
    print("Error opening camera stream")

frame_counter = 0

while True:
    # Read a frame from the camera
    ret, frame = vid.read()
    
    if not ret:
        break
    
    frame_counter += 1
    
    # Convert frame to HSV
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Display the frame in HSV color space
    cv2.imshow('Frame', image_hsv)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
vid.release()
cv2.destroyAllWindows()

print("Total frames processed:", frame_counter)


