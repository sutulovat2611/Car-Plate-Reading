
import cv2
import numpy as np
from skimage.util import random_noise
from imutils import grab_contours
import os
from os import listdir


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
    sharpened = cv2.filter2D(dst, -10, kernel) # applying the sharpening kernel to the input image & displaying it.
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
    return img_to_show

def resize(img):   
    orig_height = img.shape[0]
    orig_width = img.shape[1]
    if (orig_width < 2000 or orig_height < 2000):
        scale_percent = 60 # percent of original size
    else:
        scale_percent = 50 # percent of original size
    width = int(orig_width * scale_percent / 100)
    height = int(orig_height * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def normalize(img):
    norm_image = cv2.normalize(img, None, 0 , 255,cv2.NORM_MINMAX)
    norm_image = cv2.equalizeHist(norm_image)
    return img

def adaptive_treshold(img):
    #Median blur
    img = cv2.medianBlur(img,5)
    # Aply adaptive tresholding
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 30)
    return img

def algorithm1(processed_img, orig_image, img_name):   
    _, thresh = cv2.threshold(processed_img, 120, 255, cv2.THRESH_TRUNC)

    # Detecting the edges with Canny algorithm
    img_edged = cv2.Canny(thresh, 0, 200, 255) 
    # cv2.imshow("Canny Image", img_edged)
    # cv2.waitKey(0)

    # Finding contours from the edged image
    cnts, _ = cv2.findContours(img_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # Sorting the identified contours
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30] # sorting contours based on the min are of 30 and ignoring the ones below that
    screenCnt = None
    image2 = orig_image.copy()
    cv2.drawContours(image2,cnts,-1,(0,255,0),3)
    # cv2.imshow("Top 30 contours",image2)
    # cv2.waitKey(0)

    # Finding contours with four sides
    i=0
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True) # approximates the curve of polygon with precision
        # choosing contours with four sides
        if len(approx) == 4: 
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            if aspect_ratio > 2.5 and aspect_ratio < 5:
                screenCnt = approx
                new_img=orig_image[y:y+h,x:x+w] # create new image of with detected car plate
                cv2.imwrite('results/'+img_name+'_alg1.jpg',new_img)
                i+=1
                break

    # Drawing the selected contour on the original image
    # cv2.drawContours(orig_image, [screenCnt], -1, (0, 255, 0), 3)
    # cv2.imshow("image with detected license plate", orig_image)
    # cv2.waitKey(0)

def algorithm2(img, img_resized, img_name):
    # Smoothening images and reducing noise, while preserving edges.
    bfilter = cv2.bilateralFilter(img, 11, 17, 17)

    # Detecting edges using Canny algorithm
    edged = cv2.Canny(bfilter, 240, 255)

    # Pass the shape and size of the kernel, and get the desired kernel.
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))

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
    # cv2.imshow("cont",img_resized)
    # cv2.waitKey(0)

    # Determining the contour of the car plate
    location = None
    crop = None
    for c in cnts:
        # Approximating contour
        epsilon = 0.01 * cv2.arcLength(c, True) # accuracy 1%
        approx = cv2.approxPolyDP(c, epsilon, True)
        # Drawing approximate rectangle around bounding picture
        _, _, w, h = cv2.boundingRect(c)
        # approx = cv2.approxPolyDP(c, 10, True)
        #  len(approx) >= 2 and len(approx) < 6 and 
        # Determining the shape 
        if w/h >= 1 and w/h <=6 :
            location = approx
            break

    # if the car plate is supposingly found
    if location is not None:
        # masking out everything but the car plate area
        mask = np.zeros(img.shape, np.uint8)
        new_img = cv2.drawContours(mask, [location], 0, 255, -1)
        new_img = cv2.bitwise_and(img_resized, img_resized, mask=mask)
        # cv2.imshow("bitwise",new_img)
        # cv2.waitKey(0)
        # TO BE ADDED
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        crop = img[x1-5:x2+5, y1-5:y2+5]       
    else:
        print("location not found")

    # Cropped Image
    cv2.imwrite('results/'+img_name+'_alg2.jpg',crop)
    # cv2.imshow("Cropped image",crop)
    # cv2.waitKey(0)
    return crop
    
    # blur = cv2.GaussianBlur(crop, (5,5), 0)
    # unsharp_image = cv2.addWeighted(crop, 2, blur, -1, 0)
    # # plt.imshow(cv2.cvtColor(unsharp_image, cv2.COLOR_BGR2RGB))
    # cnts, _ = cv2.findContours(unsharp_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # cv2.drawContours(unsharp_image, cnts, -1, (0, 255, 0), 2)
    # cv2.imshow("contour2",unsharp_image)
    # cv2.waitKey(0)

def processing(folder_name, image_name):
    new_image_name = image_name[:-4] # will be used for result pictures
    
    # Load image
    orig_image = cv2.imread(folder_name + "/"+ image_name)

    # IMAGE PRE-PROCESSING
    # Step 1: Image resizing
    img_resized = resize(orig_image)

    # convert picture to gray scale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Step 2: Normalization
    img = normalize(img_gray)

    # Step 3: Noise reduction
    # img = add_noise(img) # adding noise to an image to test the noise reduction techniques

    # Remove noise
    img = noise_reduction(img)
    
    # Aply adaptive tresholding
    img= adaptive_treshold(img)

    # CAR PLATE DETECTION
    # Step 4: Algorithm 1 to detect
    try:
        algorithm1(img, img_resized, new_image_name)
    except:
        print("Algorithm 1 did not detect the car plate for " + new_image_name +".jpg")

    # Step 5: Algorithm 2 to detect
    try:
        algorithm2(img, img_resized, new_image_name)
    except:
        print("Algorithm 2 did not detect the car plate for " + new_image_name+".jpg")


if __name__ == "__main__":
    folder1 = "./images"
    folder2 = "./images2"

    # # Test cases for folder 2
    # for image in listdir(folder2):
    #     processing(folder2, image)

    # Test cases for folder 1
    for image in listdir(folder1):
        processing(folder1, image)

    # Results:
    # Both algos together for set 2: 23/45 51% accuracy
    # Both algos together for set 1: 9/15 60% accuracy
    
    # Set 1 success: 002, 004, 005, 006, 009, 010, 012, 014 (Alex's algo) 53%
    # Set 2 success: 001,002, 003, 004, 005, 007, 009, 015, 021, 022, 023, 026, 027, 030, 033, 034, 038, 039, 042 (Alex's algo) 40%
    # Problem: does not detect properly when there is a big gap between numbers
    # Problem 2: gets distracted by things outside of the car 025

    # Set 2 success: 000, 002, 008, 015, 019, 029 my algo
    # Set 1 success: 011, 014 my algo


 

