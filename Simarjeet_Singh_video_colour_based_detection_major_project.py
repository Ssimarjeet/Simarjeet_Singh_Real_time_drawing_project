import cv2
import numpy as np
import imutils
import time

# Define color range for bright pink in HSV format
low_pink = np.array([160, 100, 100])
high_pink = np.array([179, 255, 255])

# Create the video capture object for phone camera (assuming it's the default camera, index 0)
vid = cv2.VideoCapture(0)

if not vid.isOpened():
    print("Error opening video stream or camera")

frame_counter = 0
prev_centroid = None
lines = []

while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break

    # Convert the frame to HSV format
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_counter += 1

    # Create a mask based on the defined color range
    mask_image = cv2.inRange(image_hsv, low_pink, high_pink)

    # Find contours in the mask
    contours = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if frame_counter == 1:
        frame_original = frame.copy()

    # Loop over the contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Draw the contour on the frame with softer lines
                cv2.drawContours(frame, [contour], -1, (255, 0, 255), 2, lineType=cv2.LINE_AA)

                # Draw a line from the previous centroid to the current centroid
                if prev_centroid is not None:
                    lines.append((prev_centroid, (cx, cy)))

                # Update the previous centroid
                prev_centroid = (cx, cy)

    # Draw all the lines
    for line in lines:
        cv2.line(frame, line[0], line[1], (255, 0, 0), 2, lineType=cv2.LINE_AA)

    # Draw the most recent centroid
    if prev_centroid is not None:
        cv2.circle(frame, prev_centroid, 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Slow down the frame processing
    time.sleep(0.05)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
vid.release()
cv2.destroyAllWindows()

print("Total frames processed:", frame_counter)
