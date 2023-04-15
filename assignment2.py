
import cv2
import numpy as np
from skimage.util import random_noise
from imutils import contours
import pytesseract


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
    return img_to_show

def resize(img):
    scale_percent = 30 # percent of original size
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
    img_edged = cv2.Canny(processed_img, 30, 200) 
    # cv2.imshow("Canny Image", edged)
    # cv2.waitKey(0)

    # Finding contours from the edged image
    cnts, _ = cv2.findContours(img_edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # image_copy = orig_image.copy() # making the copy of the resized image
    # cv2.drawContours(image_copy ,cnts,-1,(0,255,0),3) # draw contours on the image
    # # cv2.imshow("contours",image1)
    # # cv2.waitKey(0)

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
            screenCnt = approx
            x,y,w,h = cv2.boundingRect(c) # finding coordinates
            print(x,y,w,h)
            new_img=img_resized[y:y+h,x:x+w] # create new image of with detected car plate
            cv2.imwrite('./'+str(i)+'.jpg',new_img)
            i+=1
            # break

    # Drawing the selected contour on the original image
    cv2.drawContours(img_resized, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow("image with detected license plate", img_resized)
    cv2.waitKey(0)

def tresholding_otsu(img):
    #Use median blur first to remove some noise when tresholding
    blurred = cv2.medianBlur(img,5)
    #Apply Otsu's Binarization
    ret,otsu = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return otsu


     
if __name__ == "__main__":
    # IMAGE PRE-PROCESSING

    # Load image
    orig_image = cv2.imread("images/IMG8.jpg")
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
    # cv2.imshow("Noise Image", img)
    # cv2.waitKey(0)

    # Remove noise
    img = noise_reduction(img)
    cv2.imshow("Noise Reduced Image", img)
    cv2.waitKey(0)

    # Step 4: Image segmentation
    segment(img, img_resized)
    