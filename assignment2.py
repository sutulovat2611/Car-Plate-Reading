
import cv2
import numpy as np
from skimage.util import random_noise
from PIL import Image, ImageFilter
from scipy.signal import wiener
import matplotlib.pyplot as plt


def noise_reduction(img):

    # img = cv2.medianBlur(img,3)
    img = cv2.GaussianBlur(img,(3,3),1)
    img = cv2.bilateralFilter(img, 11 , 31, 27)

    # Denoising
    dst = cv2.fastNlMeansDenoising(img,None,10, 7,21)

    # Sharpening
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(dst, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
    return sharpened

def add_noise(img):
    # Convert into array
    img = np.array(img)

    # Add noise to the image
    img = random_noise(img, mode = 's&p', amount = 0.01) # salt-and-pepper 
    img= random_noise(img, mode = 'gaussian', seed=None, clip=True) #gaussian
    img = random_noise(img, mode='speckle', var=0.01, clip=True) #speckle
    img = random_noise(img, mode='poisson', seed=1) #poisson

    # Convert back to non-array
    img_to_show = np.array(255*img, dtype = 'uint8')
    cv2.imshow('blur',img_to_show)
    cv2.waitKey(0)
    return img_to_show

def resize(img):
    scale_percent = 25 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def normalize(img):
    norm_image = cv2.normalize(img, None, 0 , 255,cv2.NORM_MINMAX)
    norm_image = cv2.equalizeHist(norm_image)
    return img

if __name__ == "__main__":
    # Image pre-processing

    # Load image
    img = cv2.imread("images/IMG_1201.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting to gray scale

    # Step 1: Image resizing
    img = resize(img)

    # Step 2: Normalization
    img = normalize(img)

    # Step 3: Noise reduction
    # Adding noise to an image to test the noise reduction technique
    img = add_noise(img)
    img = noise_reduction(img)
    # # Step 3: Image segmentation

    # Step 4: Perspective correction


