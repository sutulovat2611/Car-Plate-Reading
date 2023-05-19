from os import listdir
import cv2
import numpy as np
from skimage.util import random_noise
from imutils import grab_contours
import copy

def noise_reduction(img):
    """
    noise_reduction function removes noise from the image by applying Gaussian Blur, Bilateral Filtering and Non-local means denoising.
    Then it sharpens the image to restore the level of detail.
    :img: the image that is to be processed
    :return: noise reduced image
    """
    # Gaussian Blur
    img = cv2.GaussianBlur(img,(3,3),1)
    # Bilateral Filtering
    img = cv2.bilateralFilter(img, 11 , 31, 27)

    # Non-local means denoising.
    img = cv2.fastNlMeansDenoising(img,None,10, 7,21)

    # Sharpening
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -10, kernel) # applying the sharpening kernel to the image
    return sharpened

def add_noise(img):
    """
    add_noise function is needed for testing noise reduction techniques. It adds different types of noise to an image.
    :img: the image that is to be processed
    :return: noisy image
    """
    # Convert into array
    img = np.array(img)

    # Add noise to the image
    img = random_noise(img, mode = 's&p', amount = 0.01) # salt-and-pepper 
    img= random_noise(img, mode = 'gaussian', seed=None, clip=True) #gaussian
    img = random_noise(img, mode='speckle', var=0.01, clip=True) #speckle
    img = random_noise(img, mode='poisson', seed=1) #poisson

    # Convert back to non-array
    img_to_show = np.array(255*img, dtype = 'uint8')
    return img_to_show

def resize(img): 
    """
    As the name suggests, resize function resizes an image based on its original size. If the original dimensions are less than 2000 pixels
    it is resized to 60% of it's original size, otherwise to 50%
    :img: the image that is to be processed
    :return: resized image
    """  
    # Determining the original width and height
    orig_height = img.shape[0]
    orig_width = img.shape[1]
    if (orig_width < 2000 or orig_height < 2000):
        scale_percent = 60 # percent of original size
    else:
        scale_percent = 50 # percent of original size
    width = int(orig_width * scale_percent / 100)
    height = int(orig_height * scale_percent / 100)
    # Building an image with new dimensions
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def normalize(img):
    """
    normalize function performs image normalization with min-max normalization and histogram equalization 
    :img: the image that is to be processed
    :return: normalized image
    """
    # Min-max normalization 
    norm_image = cv2.normalize(img, None, 0 , 255,cv2.NORM_MINMAX)
    # Histogram equalization
    norm_image = cv2.equalizeHist(norm_image)
    return img

def adaptive_threshold(img):
    """
    adaptive_threshold performs median blur and adaptive threshold in order to prepare image for car plate detection using algorithm 2
    :img: the image that is to be processed
    :return: normalized image
    """
    # Median blur
    img = cv2.medianBlur(img,5)
    # Apply adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 30)
    return img

def algorithm1(processed_img, orig_image, img_name):   
    """
    Detects the car plate area by looking for a rectangular shape object with four sides. 
    Uses Canny edge detection algorithm for detecting edges and findContours() function to find the contours of the detected edges.
    Assumptions: approximate contour should have 4 sides & 2.5 < width/height < 5.
    Then, crops out the located area and saves to the results folder.
    :processed_img: processed image to work with
    :orig_image: resized image to crop out the detected car plate area from it
    :img_name: String containing the image name to save in the results folder
    """
    # Thresholding
    _, thresh = cv2.threshold(processed_img, 120, 255, cv2.THRESH_TRUNC)

    # Detecting the edges with Canny algorithm
    img_edged = cv2.Canny(thresh, 0, 200, 255) 
        
    # Finding contours from the edged image
    cnts, _ = cv2.findContours(img_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sorting the identified contours
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30] # sorting contours based on the contour area
   
    # Finding contours with four sides
    i=0
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True) # approximates the curve of polygon with precision 0.018
        # choosing contours with four sides
        if len(approx) == 4: 
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            if aspect_ratio > 2.5 and aspect_ratio < 5:
                new_img=orig_image[y:y+h,x:x+w] # create new image of with detected car plate
                cv2.imwrite('results/'+img_name+'_alg1.jpg',new_img)
                i+=1
                break

def algorithm2(img, img_resized, img_name):
    """
    Detects the car plate area by looking for the white numbers & characters with the black background.
    Processes the image using Morphological Transformation, erosion and dilation.
    Uses Canny edge detection algorithm for detecting edges and findContours() function to find the contours of the detected edges.
    Assumptions: approximate contour should have 1 <= width/height <= 6.
    Then, crops out the located area and saves to the results folder.
    :processed_img: processed image to work with
    :orig_image: resized image to crop out the detected car plate area from it
    :img_name: String containing the image name to save in the results folder
    """
    ori_img = copy.deepcopy(img_resized)
    # Smoothening images and reducing noise, while preserving edges.
    bfilter = cv2.bilateralFilter(img, 11, 17, 17)

    # Detecting edges using Canny algorithm
    edged = cv2.Canny(bfilter, 240, 255)

    # Pass the shape and size of the kernel, and get the desired kernel.
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 1))

    # Difference between the erosion and dilation
    light = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, squareKern)

    #  Erodes away the boundaries of foreground object
    thresh = cv2.erode(light, None, iterations=2)

    # Increases the white region in the image or size of foreground object increases.
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Finding contours and collecting them, then sorting
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
    cv2.drawContours(img_resized, cnts, -1, (0, 255, 0), 2)
   
    # Determining the contour of the car plate
    location = None
    crop = None
    for c in cnts:
        # Approximating contour
        epsilon = 0.01 * cv2.arcLength(c, True) # accuracy 1%
        approx = cv2.approxPolyDP(c, epsilon, True)
        # Drawing approximate rectangle around bounding picture
        _, _, w, h = cv2.boundingRect(c)
        # Determining the shape 
        if w/h >= 1 and w/h <=6 :
            location = approx
            break

    # if the car plate is supposingly found
    if location is not None:
        # masking out everything but the car plate area
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        cv2.bitwise_and(img_resized, img_resized, mask=mask)

        # Crops out the detected area from the original image
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        crop = ori_img[x1-5:x2+5, y1-5:y2+5]       
    else:
        print("location not found")

    # Saves cropped image
    cv2.imwrite('results/'+img_name+'_alg2.jpg',crop)
    return crop
    
def processing(folder_name, image_name):
    """
    processing function performs each step of the car plate detection for both algorithm 1 and 2 by calling respective functions. The outcomes are stores in the resusults folder
    :folder_name: the name of the folder where the pictures are stored
    :image_name: the name of the picture that is to be processed
    """
    new_image_name = image_name[:-4] # will be used for result pictures
    
    # Load image
    orig_image = cv2.imread(folder_name + "/"+ image_name)

    # IMAGE PRE-PROCESSING
    # Step 1: Image resizing
    img_resized = resize(orig_image)

    # convert picture to gray scale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Step 2: Normalization
    img_norm = normalize(img_gray)

    # Step 3: Noise reduction
    # img = add_noise(img) # adding noise to an image to test the noise reduction techniques

    # Remove noise
    img_noiseReduce = noise_reduction(img_norm)
    
    # CAR PLATE DETECTION
    # Step 4: Algorithm 1 to detect
    try:
        algorithm1(img_noiseReduce, img_resized, new_image_name)
    except:
        print("Algorithm 1 did not detect the car plate for " + new_image_name +".jpg")

    # Apply adaptive tresholding
    img_adapt = adaptive_threshold(img_noiseReduce)

    # Step 5: Algorithm 2 to detect
    try:
        algorithm2(img_adapt, img_resized, new_image_name)
    except:
        print("Algorithm 2 did not detect the car plate for " + new_image_name+".jpg")


if __name__ == "__main__":
    folder1 = "./set1"
    folder2 = "./set2"

    # Test cases for folder 1
    # for image in listdir(folder1):
    #     processing(folder1, image)

    # # Test cases for folder 2
    # for image in listdir(folder2):
    #     processing(folder2, image)




 

