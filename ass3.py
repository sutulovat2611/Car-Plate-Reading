from os import listdir
import os
import cv2
import numpy as np
from skimage.util import random_noise
from imutils import grab_contours

import matplotlib.pyplot as plt

import numpy as np
import math


# OUTPUT_NEURONS = 20
# INPUT_NEURONS = 28* 28
# HIDDEN_NEURONS = 16

def Weight_Initialization():
    # Initializing of the Weights. Random float number between -0.5 to 0.5 for weights.
    np.random.seed(1)
    wji= np.random.uniform(-0.5, 0.5, size=(HIDDEN_NEURONS, INPUT_NEURONS))
    wkj = np.random.uniform(-0.5, 0.5, size=(OUTPUT_NEURONS, HIDDEN_NEURONS))
    bias_j = np.random.uniform(0, 1, size=(HIDDEN_NEURONS))
    bias_k = np.random.uniform(0, 1, size=(OUTPUT_NEURONS))
    return  wji,wkj,bias_j,bias_k

# def Read_Files():
#     # Reading of Segmented Training Files, and Target Files.
def Forward_Input_Hidden(inputs,wji, bias_j):
    # Forward Propagation from Input -> Hidden Layer.
    # Obtain the results at each neuron in the hidden layer.
    # Calculate 𝑁𝑒𝑡𝑗and 𝑂𝑢𝑡𝑗
    
    Netj = np.dot(inputs,wji.T) 
    # print("Netj :")
    # print(Netj)
    Outj = 1/(1 + math.e**-(Netj + np.transpose(bias_j)))
    # print("Outj :")
    # print(Outj)
    return Netj,Outj

def Forward_Hidden_Output(Netj,wkj, bias_k):
    # Forward Propagation from Input -> Hidden Layer.
    # Obtain the results at each neuron in the hidden layer.
    # Calculate 𝑁𝑒𝑡kand 𝑂𝑢𝑡k
    Netk = np.dot(Netj,wkj.T) 
    # print("Netk :")
    # print(Netk)
    Outk = 1/(1 + math.e**-(Netk + np.transpose(bias_k)))
    # print("Outk :")
    # print(Outk)
    return Netk, Outk

def Check_for_End(Outk, targets, user_set):
    # Check whether the total error is less than the error set by the user or the number of iterations is reached.
    # returns true or false
    def Error_Correction(outs, targets):
        total_error= np.sum(((outs - targets)**2))/OUTPUT_NEURONS
        # print(total_error)
        return total_error
    
    if Error_Correction(Outk, targets)< user_set:
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
    # print(dwkkj)
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
    n = 0.5
    wkjj = wkj - n*dwkkj
    bkkj = bias_k - n*dbkkj
    # print("wk+")
    # print(wkjj)
    # print("bias_k+")
    # print(bkkj)

    # Calculate 𝑤𝑗𝑗𝑖+ and 𝑏𝑗𝑗𝑖+
    wjji = wji - n *dwjji
    bjji = bias_j - n* dbjii
    # print("wj+")
    # print(wjji[0][300])
    # print("bias_j+")
    # print(bjji)
    return wkjj,bkkj,wjji,bjji
# def Saving_Weights_Bias(wkjj,bkkj,wjji,bjji):
#     # Save 𝑤𝑘𝑘𝑗 and 𝑏𝑘𝑘𝑗
#     # Save 𝑤𝑗𝑗𝑖 and 𝑏𝑗𝑗𝑖

# wji,wkj,bias_j,bias_k,target = Weight_Initialization()
# inputs= np.array([0.2,0.5])
# wji= np.array([[0.1,0.2],[0.3,0.4]])
# wkj= np.array([[0.5,0.6],[0.7,0.8]])
# bias_j = np.array([0.2,0.2])
# bias_k = np.array([0.4,0.4])
# target = np.array([0.2,0.8])
# # inputs= np.array([0.5,0.8])
# # wji= np.array([[-0.8,0.2],[-0.5,-0.2],[0.4,0.4]])
# # wkj= np.array([[-0.3,0.15,0.4],[0.4,-0.5,0.1]])
# # bias_j = np.array([0.4,0.2,0.1])
# # bias_k = np.array([0.3,0.15])
# # target = np.array([0.7,0.4])
# netj,outj = Forward_Input_Hidden(inputs, wji, bias_j)
# netk,outk = Forward_Hidden_Output(outj, wkj, bias_k)

# dwkkj,dbkkj = Weight_Bias_Correction_Output(outk,target, outj)

# dwjji, dbjii = Weight_Bias_Correction_Hidden(outj,outk,inputs,target,wkj)

# Weight_Bias_Update(wkj,dwkkj, bias_k, dbkkj, wji, dwjji,bias_j,dbjii)

# Error_Correction(outk, target)

import random

OUTPUT_NEURONS =6
INPUT_NEURONS = 28* 28
HIDDEN_NEURONS = 16
ITTERATIONS = 200
ERROR = 0.005
i= 0 
j= 0



target_fd = "./Car-Plate-Reading/character_image/train_case2"
file_list = os.listdir(target_fd)
random.seed(1)
random.shuffle(file_list)
for name in file_list:
    if name.startswith("0") :
        # Read the image file using OpenCV
        targets1 = [1,0,0,0,0,0]
    elif name.startswith('1') :
        targets1 = [0,1,0,0,0,0]
    elif name.startswith('2') :
        targets1 = [0,0,1,0,0,0]
    elif name.startswith("3"):
        targets1 = [0,0,0,1,0,0]
    elif name.startswith("4"):
        targets1 = [0,0,0,0,1,0]
    elif name.startswith("5"):
        targets1 = [0,0,0,0,0,1]
    image =cv2.imread(os.path.join(target_fd,name))
    resized = cv2.resize(image, (28,28))
    # convert picture to gray scale
    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # _,img = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
    x_flattend = img_gray.reshape(1, 28*28)
    # plt.matshow(img_gray)
    x_flattend = np.squeeze(x_flattend)
    x_flattend = x_flattend/255
    inputs  = x_flattend
    
    if(j == 0):
        wji,wkj,bias_j,bias_k = Weight_Initialization()
        j+=1
        
    for i in range(ITTERATIONS):
        netj,outj = Forward_Input_Hidden(inputs, wji, bias_j)
        netk,outk = Forward_Hidden_Output(outj, wkj, bias_k)
        if(Check_for_End(outk, targets1, ERROR)):
            print("x")
            # print(name)
            print(np.argmax(outk))
            print(outk)
            break
        else:
            dwkkj,dbkkj = Weight_Bias_Correction_Output(outk,targets1, outj)
            dwjji, dbjii = Weight_Bias_Correction_Hidden(outj,outk,inputs,targets1,wkj)
            wkj,bias_k,wji,bias_j = Weight_Bias_Update(wkj,dwkkj, bias_k, dbkkj, wji, dwjji,bias_j,dbjii)


        
#test
target_fd = "./Car-Plate-Reading/character_image/test_case"
for name in listdir(target_fd):
    image =cv2.imread(os.path.join(target_fd,name))
    resized = cv2.resize(image, (28,28))
    # convert picture to gray scale
    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
    x_flattend = img.reshape(1, 28*28)

    x_flattend = np.squeeze(x_flattend)
    x_flattend = x_flattend/255
    inputs  = x_flattend
    netj,outj = Forward_Input_Hidden(inputs, wji, bias_j)
    netk,outk = Forward_Hidden_Output(outj, wkj, bias_k)
    print(np.argmax(outk))
    # print(outk)

# #result not good zz 
# #a
