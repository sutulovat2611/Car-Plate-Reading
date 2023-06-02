from os import listdir
import os
import cv2
import numpy as np
import random
import math
import functools


def Weight_Initialization():
    # Initializing of the Weights. Random float number between -0.5 to 0.5 for weights.
    np.random.seed(1)
    wji= np.random.uniform(-0.5, 0.5, size=(HIDDEN_NEURONS, INPUT_NEURONS))
    wkj = np.random.uniform(-0.5, 0.5, size=(OUTPUT_NEURONS, HIDDEN_NEURONS))
    bias_j = np.random.uniform(0, 1, size=(HIDDEN_NEURONS))
    bias_k = np.random.uniform(0, 1, size=(OUTPUT_NEURONS))
    return  wji,wkj,bias_j,bias_k


def Forward_Input_Hidden(inputs,wji, bias_j):
    # Forward Propagation from Input -> Hidden Layer.
    # Obtain the results at each neuron in the hidden layer.
    # Calculate 𝑁𝑒𝑡𝑗and 𝑂𝑢𝑡𝑗
    Netj = np.dot(inputs,wji.T) 
    Outj = 1/(1 + math.e**-(Netj + np.transpose(bias_j)))
    return Netj,Outj

def Forward_Hidden_Output(Netj, wkj, bias_k):
    # Forward Propagation from Input -> Hidden Layer.
    # Obtain the results at each neuron in the hidden layer.
    # Calculate 𝑁𝑒𝑡kand 𝑂𝑢𝑡k
    Netk = np.dot(Netj,wkj.T) 
    Outk = 1/(1 + math.e**-(Netk + np.transpose(bias_k)))
    return Netk, Outk

def Check_for_End(Outk, targets, user_set, i):
    # Check whether the total error is less than the error set by the user or the number of iterations is reached.
    # returns true or false
    def Error_Correction(outs, targets):
        total_error= np.sum(((outs - targets)**2))/OUTPUT_NEURONS
        return total_error
    
    if Error_Correction(Outk, targets)< user_set or i > ITTERATIONS:
        return True
    else: 
        return False
        
def Weight_Bias_Correction_Output(Outk, targets, Outj):
    # Correction of Weights and Bias between Hidden and Output Layer.
    # Calculate 𝑑𝑤𝑘𝑘𝑗 and 𝑑𝑏𝑘𝑘𝑗
    dwkkj =  np.empty((0, len(Outk)))
    for i in range(len(Outj)):
        temp =(Outk - targets) * Outk*(1 - Outk) * Outj[i]
        dwkkj = np.vstack([dwkkj,temp])
    dbkkj = (Outk - targets) * Outk*(1 - Outk) 
    dwkkj = dwkkj.T
    return dwkkj,dbkkj

def Weight_Bias_Correction_Hidden(outj,outk,inputs,target,wkj):
    # Correction of Weights and Bias between Input and Hidden Layer.
    # Calculate 𝑑𝑤𝑗𝑗𝑖 and 𝑑𝑏𝑗𝑗𝑖
    skl = (outk - target) * outk*(1-outk)
    dwjji= np.multiply.outer(outj *(1 - outj) * np.dot(skl,wkj),inputs)
    dbjii = outj *(1 - outj) * np.dot(skl,wkj)
    return dwjji, dbjii

def Weight_Bias_Update(wkj,dwkkj, bias_k, dbkkj, wji, dwjji,bias_j,dbjii ):
    # Saving_Weights_Bias() implemented inside
    # Update Weights and Bias.
    # Calculate 𝑤𝑘𝑘𝑗+ and 𝑏𝑘𝑘𝑗+
    
    wkkj = wkj - N*dwkkj
    bkkj = bias_k - N*dbkkj

    # Calculate 𝑤𝑗𝑗𝑖+ and 𝑏𝑗𝑗𝑖+
    wjji = wji - N *dwjji
    bjji = bias_j - N* dbjii
    return wkkj,bkkj,wjji,bjji

def Saving_Weights_Bias(wkkj,bkkj,wjji,bjji):
    # Save 𝑤𝑘𝑘𝑗 and 𝑏𝑘𝑘𝑗
    # Save 𝑤𝑗𝑗𝑖 and 𝑏𝑗𝑗𝑖
    wkj,bias_k,wji,bias_j = wkkj,bkkj,wjji,bjji
    return wkj,bias_k,wji,bias_j



def auto_segmentation():
    target_fd = "./target"
    file_list = os.listdir(target_fd)

    counter = 0
    for img in file_list:
        naming = img[:-4]

        # Read the image and convert to grayscale
        image = cv2.imread("./target/" + img)
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
        for (_, label) in enumerate(np.unique(labels)):
            # If this is the background label, ignore it
            if label == 0:
                continue
            # Otherwise, constrsuct the label mask to display only connected component for the current label
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255        
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
                    cv2.imwrite('results/'+str(counter)+"_"+str(char)+"_"+naming[char]+'.jpg', crop)
                except:
                    cv2.imwrite('results/'+str(counter)+"_"+str(char)+"_"+naming[char]+'.jpg', crop_def)
                char+=1
                
        counter+=1


def img_manipulation(target_fd,name):

    image =cv2.imread(os.path.join(target_fd,name)) # reading image
    resized = cv2.resize(image, (28,28)) # resizing image
    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # convert picture to gray scale
    _,img = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY) # applying thresholding
       
    # Representing the image in array form
    x_flattend = img.reshape(1, 28*28)
    x_flattend = np.squeeze(x_flattend)
    x_flattend = x_flattend/255

    # Z-score normalisation
    mean = np.mean(x_flattend)
    std = np.std(x_flattend)
    x_normalized = (x_flattend - mean) / std

    return x_normalized



if __name__ == "__main__":
    # Initializing values 
    OUTPUT_NEURONS = 20
    INPUT_NEURONS = 28* 28 # total number of pixels in an image
    HIDDEN_NEURONS = 100
    ITTERATIONS = 300 # max number of iterations to be performed
    ERROR = 0.001 # acceptable error rate
    N = 0.1
    j= 0
    alphabets_targets = {'B':10, 'F':11, 'L':12, 'M':13, 'P':14, 'Q':15, 'T':16, 'U':17, 'V':18, 'W':19}
    accuracy = 0
    total = 0
    wkkj,bkkj,wjji,bjji = 0,0,0,0
    ############################################################################################################
    # TRAINING
    ############################################################################################################
    # Accessing train cases folder
    target_fd = "./train_case"
    file_list = os.listdir(target_fd)

    # Randomizing the dataset order
    random.seed(1)
    random.shuffle(file_list)

    # Going through each case in the folder
    for name in file_list:
        # Determining the target label
        label = name[0] 
        try:
            label_value = int(label)
        except ValueError:
            label_value = alphabets_targets.get(label)
            assert label_value is not None, "label_value is None"

        # Showing the target in the array form
        targets = [0]* 20
        targets[label_value] = 1

        # Image manipulation
        inputs  = img_manipulation(target_fd,name)

        # Starting training with Weights and Bias initialization
        if(j == 0):
            wji,wkj,bias_j,bias_k = Weight_Initialization()
            j+=1
        i=0
        # Performing training
        while i < ITTERATIONS:
            # Forward Propagation from Input -> Hidden Layer.
            netj,outj = Forward_Input_Hidden(inputs, wji, bias_j)
            # Forward Propagation from Hidden -> Output Layer.
            netk,outk = Forward_Hidden_Output(outj, wkj, bias_k)
            # Checking for error value
            if(Check_for_End(outk, targets, ERROR, i)):
                break
            else:
                dwkkj,dbkkj = Weight_Bias_Correction_Output(outk,targets, outj)
                dwjji, dbjii = Weight_Bias_Correction_Hidden(outj,outk,inputs,targets,wkj)
                wkj,bias_k,wji,bias_j = Weight_Bias_Update(wkj,dwkkj, bias_k, dbkkj, wji, dwjji,bias_j,dbjii)
        
        # Finding the training data that does not met the criteria
        if (np.max(outk)<0.9):
            print(name)
            print(np.max(outk))

            i+=1
    # Saving weights & bias
    Saving_Weights_Bias(wkj,bias_k,wji,bias_j)

    ############################################################################################################
    # TESTING WITH MANUALLY CROPPED IMAGES
    ############################################################################################################
    # Accessing test cases folder
    target_fd = "./test_case"
    for name in listdir(target_fd):
        total+=1 # Counting the total number of images

        # Determining the target label
        label = name[0]
        try:
            label_value = int(label)
        except ValueError:
            label_value = alphabets_targets.get(label)
            assert label_value is not None, "label_value is None"

        # Image manipulation
        inputs  = img_manipulation(target_fd,name)

        # Performing testing
        netj,outj = Forward_Input_Hidden(inputs, wji, bias_j)
        netk,outk = Forward_Hidden_Output(outj, wkj, bias_k)

        # Determinig how many values are classified right
        if np.argmax(outk) == label_value :
            accuracy +=1

    print("Accuracy for testing with manually cropped images: " + str(accuracy/total))

    ############################################################################################################
    # TESTING WITH AUTO-SEGMENTED IMAGES OF MALAYSIAN CAR PLATE
    ############################################################################################################

    total = 0
    accuracy = 0
    # Performing auto segmentation
    auto_segmentation()

    # Accessing the folder with segmentation results
    target_fd = "./results"
    
    # Determining the target label
    for name in listdir(target_fd):
        total+=1 # Counting the total number of images
        label = name[4]
        try:
            label_value = int(label)
        except ValueError:
            label_value = alphabets_targets.get(label)
            assert label_value is not None, "label_value is None"

        # Image manipulation
        inputs  = img_manipulation(target_fd,name)

        # Performing testing
        netj,outj = Forward_Input_Hidden(inputs, wji, bias_j)
        netk,outk = Forward_Hidden_Output(outj, wkj, bias_k)
        
        if np.argmax(outk) == label_value :
            accuracy +=1
    print(accuracy)
    print("Accuracy for testing with auto-segmented images: " + str(accuracy/total))





