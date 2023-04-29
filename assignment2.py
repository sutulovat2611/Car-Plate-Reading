
import cv2
import numpy as np
from skimage.util import random_noise
from imutils import contours, grab_contours
import pytesseract
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)


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

def segment(processed_img, orig_image):   
    # Detecting the edges with Canny algorithm
    img_edged = cv2.Canny(processed_img, 0, 200, 255) 
    cv2.imshow("Canny Image", img_edged)
    cv2.waitKey(0)

    # Finding contours from the edged image
    cnts, _ = cv2.findContours(img_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # Sorting the identified contours
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30] # sorting contours based on the min are of 30 and ignoring the ones below that
    screenCnt = None
    image2 = orig_image.copy()
    cv2.drawContours(image2,cnts,-1,(0,255,0),3)
    cv2.imshow("Top 30 contours",image2)
    cv2.waitKey(0)

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
                print(x,y,w,h)
                new_img=img_resized[y:y+h,x:x+w] # create new image of with detected car plate
                cv2.imwrite('./'+str(i)+'.jpg',new_img)
                i+=1
                break

    # Drawing the selected contour on the original image
    cv2.drawContours(img_resized, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow("image with detected license plate", img_resized)
    cv2.waitKey(0)

def tresholding_otsu(img):
    _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
    cv2.imshow("Mean Adaptive Thresholding", thresh)
    cv2.waitKey(0)
    return thresh


def algorithm2(img, img_resized):
    # Smoothening images and reducing noise, while preserving edges.
    bfilter = cv2.bilateralFilter(img, 11, 17, 17)

    # Detecting edges using Canny algorithm
    edged = cv2.Canny(bfilter, 240, 255)

    # Pass the shape and size of the kernel, and get the desired kernel.
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))

    # Difference between the erosion and dilation
    light = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, squareKern)

    #  Erodes away the boundaries of foreground object
    thresh = cv2.erode(light, None, iterations=1)

    # Increases the white region in the image or size of foreground object increases.
    thresh = cv2.dilate(thresh, None, iterations=1)
    
    # Finding contours and collecting them, then sorting
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
    cv2.drawContours(img_resized, cnts, -1, (0, 255, 0), 2)
    cv2.imshow("cont",img_resized)
    cv2.waitKey(0)

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
        cv2.imshow("bitwise",new_img)
        cv2.waitKey(0)
        # TO BE ADDED
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        crop = img[x1-5:x2+5, y1-5:y2+5]       
    else:
        print("location not found")

    # Cropped Image
    cv2.imshow("Cropped image",crop)
    cv2.waitKey(0)
    return crop
    
    # blur = cv2.GaussianBlur(crop, (5,5), 0)
    # unsharp_image = cv2.addWeighted(crop, 2, blur, -1, 0)
    # # plt.imshow(cv2.cvtColor(unsharp_image, cv2.COLOR_BGR2RGB))
    # cnts, _ = cv2.findContours(unsharp_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # cv2.drawContours(unsharp_image, cnts, -1, (0, 255, 0), 2)
    # cv2.imshow("contour2",unsharp_image)
    # cv2.waitKey(0)



     
if __name__ == "__main__":
    # IMAGE PRE-PROCESSING

    # Load image
    orig_image = cv2.imread("images/014.jpg")
    # cv2.imshow("Original Image",orig_image)
    # cv2.waitKey(0)

    # Step 1: Image resizing
    img_resized = resize(orig_image)
    # cv2.imshow("Resized image",img_resized)
    # cv2.waitKey(0)

    # convert picture to gray scale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray Scale Image", img_gray)
    # cv2.waitKey(0)

    # Step 2: Normalization
    img = normalize(img_gray)
    # cv2.imshow("Normalized Image", img)
    # cv2.waitKey(0)

    # Step 3: Noise reduction
    img = add_noise(img) # adding noise to an image tokj test the noise reduction technique
    cv2.imshow("Noise Image", img)
    cv2.waitKey(0)

    # Remove noise
    img = noise_reduction(img)
    cv2.imshow("Noise Reduced Image", img)
    cv2.waitKey(0)

    # img = tresholding_otsu(img)

    # # Step 4: Image segmentation
    # segment(img, img_resized)
    algorithm2(img, img_resized)

    # Set 1 success: 002, 004, 005, 006, 009, 010, 012, 014