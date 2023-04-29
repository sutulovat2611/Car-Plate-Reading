import cv2
import numpy as np

img = cv2.imread("images2/025.jpg", 0)

# Threshold and Invert the image to find the contours
ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)

# Find the contours
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

x, y, r, b = [img.shape[1]/2, img.shape[0]/2, 0, 0]
# Iterate over all the contours, and keep updating the bounding rect

for cnt in contours:
    rect = cv2.boundingRect(cnt)
    if rect[0] < x:
        x = rect[0]
    if rect[1] < y:
        y = rect[1]
    if rect[0] + rect[2] > r:
        r = rect[0] + rect[2]
    if rect[1] + rect[3] > b:
        b = rect[1] + rect[3]

bounding_rect = [x, y, r-x, b-y]
print(bounding_rect)

# Debugging Purpose.
# cv2.rectangle(img, (x, y), (r, b), np.array([0, 255, 0]), 3)