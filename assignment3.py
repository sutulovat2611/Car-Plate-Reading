from os import listdir
import cv2
import numpy as np
import math


def Weight_Initialization(HIDDEN_NEURONS, INPUT_NEURONS, OUTPUT_NEURONS):
    # Initialization of the Weights. Random float number between -0.5 to 0.5 for weights.
    np.random.seed(1)
    wji= np.random.uniform(-0.5, 0.5, size=(HIDDEN_NEURONS, INPUT_NEURONS))
    wkj = np.random.uniform(-0.5, 0.5, size=(OUTPUT_NEURONS, HIDDEN_NEURONS))
    bias_j = np.random.uniform(0, 1, size=(HIDDEN_NEURONS, 1))
    bias_k = np.random.uniform(0, 1, size=(OUTPUT_NEURONS, 1))

    wji = [[0.1, 0.2], [0.3, 0.4]]
    wkj = [[0.5, 0.6],[0.7, 0.8]]
    bias_j = [[0.2], [0.2]]
    bias_k = [[0.4], [0.4]]

    return wji, wkj, bias_j, bias_k

def Read_Files():
    # Reading of Segmented Training Files, and Target Files.
    target_fd = "./target"
    test_train_fd = "./test_train"

    test_train = []
    target = []

    for image in listdir(target_fd):
        target.append(cv2.imread(target_fd + "/"+ image))

    for image in listdir(test_train_fd):
        test_train.append(cv2.imread(test_train_fd + "/"+ image))

    return test_train, target

def Forward_Input_Hidden(inputs, weights, biases):
    # Forward Propagation from Input -> Hidden Layer.
    # Obtain the results at each neuron in the hidden layer.
    # Calculate 𝑁𝑒𝑡𝑗 and 𝑂𝑢𝑡𝑗
    NetJ_array = []
    for i in range(len(weights)):
        NetJ = 0
        for x in range(len(weights[i])):
            NetJ += (weights[i][x]*inputs[x])
        NetJ_array.append(NetJ)

    OutJ_array = []
    for i in range(len(NetJ_array)):
        OutJ = 1/(1 + math.e**-(NetJ_array[i] + biases[i][0]))
        OutJ_array.append(OutJ)
    return OutJ_array


def Forward_Hidden_Output(OutJ, wkj, bias_k):
    # Forward Propagation from Hidden -> Output Layer.
    # Calculate 𝑁𝑒𝑡𝑘 and 𝑂𝑢𝑡𝑘
    NetK_array = []
    for i in range(len(wkj)):
        NetK = 0
        for x in range(len(wkj[i])):
            NetK += (wkj[i][x]*OutJ[x])
        NetK_array.append(NetK)

    OutK_array = []
    for i in range(len(NetK_array)):
        OutK = 1/(1 + math.e**-(NetK_array[i] + bias_k[i][0]))
        OutK_array.append(OutK)
    return OutK_array

def Check_for_End(OutK_array, targets):
    # Check whether the total error is less than the error set by the user or the number of iterations is reached.
    # returns true or false
    errors_array = []
    error_total = 0
    for i in range(len(OutK_array)):
        result = ((OutK_array[i] - targets[i])**2)/2
        errors_array.append(result)
        error_total +=result
    return error_total, errors_array

def Weight_Bias_Correction_Output(OutK_array, targets, OutJ_array):
    # Correction of Weights and Bias between Hidden and Output Layer.
    # Calculate 𝑑𝑤𝑘𝑘𝑗 and 𝑑𝑏𝑘𝑘𝑗
    dwkkj = []
    dbkkj = []
    for j in range(len(OutK_array)):
        temp=[]
        for i in range(len(OutJ_array)):
            temp.append(((OutK_array[j]-targets[j])*OutK_array[j]*(1-OutK_array[j])*OutJ_array[i]))
        dwkkj.append(temp)
        dbkkj.append(((OutK_array[j]-targets[j])*OutK_array[j]*(1-OutK_array[j])))
        
    return dwkkj, dbkkj

def Weight_Bias_Correction_Hidden(OutJ_array, OutK_array, targets, inputs, wkj, HIDDEN_LAYER):
    # Correction of Weights and Bias between Input and Hidden Layer.
    # Calculate 𝑑𝑤𝑗𝑗𝑖 and 𝑑𝑏𝑗𝑗𝑖
    dwjji = []
    dbjji = []
    deltas = []
    sums = []
    for j in range(len(OutK_array)):
        deltas.append((OutK_array[j] - targets[j])* OutK_array[j] * (1-OutK_array[j]))
    
    for i in range(HIDDEN_LAYER):
        sum = 0
        for j in range(len(wkj[j])):
            sum += (deltas[j]*wkj[j][i])
        sums.append(sum)

    
    for j in range(len(OutJ_array)):
        variable = OutJ_array[j]*(1-OutJ_array[j])*sums[j]
        dbjji.append(variable)
        temp = []
        for i in range(len(inputs)):
            temp.append(variable*inputs[i])
        dwjji.append(temp)
    return dwjji, dbjji

def Weight_Bias_Update(wji, wkj, dwkkj, dwjji, bias_j, bias_k, dbjji, dbkkj ):
    # Saving_Weights_Bias() implemented inside
    # Update Weights and Bias.
    # Calculate 𝑤𝑘𝑘𝑗+ and 𝑏𝑘𝑘𝑗+

    for i in range(len(wji)):
        for j in range(len(wji[i])):
            wji[i][j] -= (0.5*dwjji[i][j])
    
    for i in range(len(wkj)):
        for j in range(len(wkj[i])):
            wkj[i][j] -= (0.5*dwkkj[i][j])

    for i in range(len(bias_j)):
        bias_j[i][0] -= (0.5*dbjji[i])
    
    for i in range(len(bias_k)):
        bias_k[i][0] -= (0.5*dbkkj[i])
    
    return wji, wkj, bias_j, bias_k


# i) Set the number of input features,
# ii) Number of hidden neurons.
# iii) Number of output neurons.
# iv) Global error or number of iterations.
# v) Any other para meters that you may want to initialize in your program.

def resize(image):
    h, w, *_ = image.shape
    if ( h > 100 or w > 100):
        scale_percent = 80 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image 



if __name__ == "__main__":
    # # Reading Image
    # image = cv2.imread("./character_image/M/001198.jpg")
    # # Resizing in case the picture is too big
    # image = resize(image)

    # # Input is the total number of pixels in an image
    # INPUT_NEURONS = image.shape[1] * image.shape[0]
    # # Set the number of hidden neurons
    # HIDDEN_NEURONS = 15
    # # Set the number of output neuron: total 20 different classes (0-9, U, V, W, B, F, L, M, P, Q, T)
    # OUTPUT_NEURONS = 20

    # # wji, wkj, bias_j, bias_k = Weight_Initialization(HIDDEN_NEURONS, INPUT_NEURONS, OUTPUT_NEURONS)
    # # test_train, target = Read_Files()
    # # NetJ, OutJ = Forward_Input_Hidden()

    # # n = 20 # NUMBER OF INPUTS??

    # # NetK, OutK = Forward_Hidden_Output(wkj, OutJ, bias_k, n)
    # # dwkkj, dbkkj = Weight_Bias_Correction_Output(OutK, OutJ)
    # # Weight_Bias_Correction_Hidden(OutK, Targetj, OutJ, wklj)

    # image = cv2.imread("./character_image/M/001198.jpg")

    # Input is the total number of pixels in an image
    INPUT_NEURONS = 2
    # Set the number of hidden neurons
    HIDDEN_NEURONS = 2
    # Set the number of output neuron: total 20 different classes (0-9, U, V, W, B, F, L, M, P, Q, T)
    OUTPUT_NEURONS = 2

    wji, wkj, bias_j, bias_k = Weight_Initialization(HIDDEN_NEURONS, INPUT_NEURONS, OUTPUT_NEURONS)

    inputs = [0.2, 0.5]
    OutJ_array = Forward_Input_Hidden(inputs, wji, bias_j)
    OutK_array = Forward_Hidden_Output(OutJ_array, wkj, bias_k)

    targets = [0.2, 0.8]
    error_total, errors_array = Check_for_End(OutK_array, targets)

    dwkkj, dbkkj = Weight_Bias_Correction_Output(OutK_array, targets, OutJ_array)
    # dwkkj is wk00, wk01, wk10, wk11

    dwjji, dbjji  = Weight_Bias_Correction_Hidden(OutJ_array, OutK_array, targets, inputs, wkj, HIDDEN_NEURONS)
    # print(dwjji)
    wji, wkj, bias_j, bias_k = Weight_Bias_Update(wji, wkj, dwkkj, dwjji, bias_j, bias_k, dbjji, dbkkj)