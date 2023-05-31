# Import the necessary packages
import functools
import cv2
import numpy as np


# Read the image and convert to grayscale
image = cv2.imread("./target/9.jpg")
h, w, *_ = image.shape

# Resizing images  
if ( w < 300):
     scale_percent = 320 # percent of original size
elif ( w < 800):
    scale_percent = 220 # percent of original size
else:
    scale_percent = 50 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# copying the image to use later
result = image.copy()

# Converting to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blurring and thresholding to reveal the characters on the license plate
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 1)

# Perform connected components analysis on the thresholded image and initialize the mask to hold only the components we are interested in
_, labels = cv2.connectedComponents(thresh)
mask = np.zeros(thresh.shape, dtype="uint8")

# Loop over the unique components
for (i, label) in enumerate(np.unique(labels)):
    # If this is the background label, ignore it
    if label == 0:
        continue
    # Otherwise, constrsuct the label mask to display only connected component for the current label
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
 
    # Add to our mask
    mask = cv2.add(mask, labelMask)

# Find contours and get bounding box for each contour
cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(c) for c in cnts]

# Sort the bounding boxes from left to right
def compare(rect1, rect2): 
    return rect1[0] - rect2[0]
boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )



char = 0
# Get contours
for bnd in boundingBoxes:
    x,y,w,h = bnd
    if (h>0.5*image.shape[0] and h<0.95*image.shape[0]):
        # Crops out the detected area from the original image
        crop = result[y-7:y+h+10, x-7:x+w+10]       
        crop_def = result[y:y+h, x:x+w]       

        # Crops out the detected area from the original image & saves the cropped image
        try:
            cv2.imwrite('results/'+str(char)+'.jpg', crop)
        except:
            cv2.imwrite('results/'+str(char)+'.jpg', crop_def)
        char+=1
        

