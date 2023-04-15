import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
# from pillow_heif import register_heif_opener

#Open heic format image
# register_heif_opener()

image = cv2.imread("images/IMG1.jpg")

#Turn image into grayscale image
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#Use median blur first to remove some noise when tresholding
blurred = cv2.medianBlur(image,5)
#Apply Otsu's Binarization
ret,image = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

## Aply adaptive tresholding
# image = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 30)
## Apply edge detection
# image = cv2.Canny(image, 30, 200) #Edge detection

## Apply sobel filter
# scale = 1
# delta = 0
# ddepth = cv2.CV_16S
# grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=1,
#                    delta=0, borderType=cv2.BORDER_DEFAULT)
# # Gradient-Y
# # grad_y = cv.Scharr(gray,ddepth,0,1)
# grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=1,
#                   delta=0, borderType=cv2.BORDER_DEFAULT)

# abs_grad_x = cv2.convertScaleAbs(grad_x)
# abs_grad_y = cv2.convertScaleAbs(grad_y)

# image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


cv2.imwrite("result.jpg", image)



# # Function to generate white pixel horizontal projection profile
# def getHorizontalProjectionProfile(image):
#     # Convert black spots to zeros
#     image[image == 0]   = 0
#     # Convert white spots to ones
#     image[image == 255] = 1
#     horizontal_projection = np.sum(image, axis =1)  
#     return horizontal_projection

# # Function to generate white pixel vertical projection profile
# def getVerticalProjectionProfile(image): 
#     # Convert black spots to zeros 
#     image[image == 0]   = 0
#     # Convert white spots to ones 
#     image[image == 255] = 1
#     vertical_projection = np.sum(image, axis = 0)
#     return vertical_projection 

# # image = np.array(image)
# # result = getHorizontalProjectionProfile(image)
# # result = getVerticalProjectionProfile(image)
# # plt.plot(result)
# # plt.show()

