import math
import numpy as np
import cv2
import os



def Weight_Initialization():
    # Initializing of the Weights. Random float number between -0.5 to 0.5 for weights.

    np.random.seed(1)
    wji = np.random.uniform(-0.5, 0.5, size=(HIDDEN_NEURONS, INPUT_NEURONS))
    wkj = np.random.uniform(-0.5, 0.5, size=(OUTPUT_NEURONS, HIDDEN_NEURONS))
    bias_j = np.random.uniform(0, 1, size=HIDDEN_NEURONS)
    bias_k = np.random.uniform(0, 1, size=OUTPUT_NEURONS)

    return wji, wkj, bias_j, bias_k

def Read_Files():
    # Reading of Segmented Training Files, and Target Files.
    img = cv2.imread("./0/0-000700.jpg", cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img, (32,32))
    img_array = np.array(img_resize).flatten()
    return img_array


def Forward_Input_Hidden(inputs, wji, bias_j):
    # Forward Propagation from Input -> Hidden Layer.
    # Obtain the results at each neuron in the hidden layer.
    # Calculate 𝑁𝑒𝑡𝑗and 𝑂𝑢𝑡�

    net_j = np.dot(inputs, np.transpose(wji))
    out_j = 1 / (1 + math.e ** -(net_j + bias_j))

    return net_j, out_j


def Forward_Hidden_Output(out_j, wkj, bias_k):
    # Forward Propagation from Hidden -> Output Layer.
    # Calculate 𝑁𝑒𝑡𝑘and 𝑂𝑢𝑡�

    net_k = np.dot(out_j, np.transpose(wkj))

    out_k = 1 / (1 + math.e ** -(net_k + bias_k))

    return net_k, out_k

def Check_for_End(out_k, targets, user_set):
    # Check whether the total error is less than the error set by the user or the number of iterations is reached.
    # returns true or false
    # If TRUE, proceed to Step 10
    def Error_Correction(outs, targets):
        total_error = np.sum(((outs - targets) ** 2) / OUTPUT_NEURONS)
        # print(total_error)
        return total_error

    if Error_Correction(out_k, targets) < user_set:
        return True
    else:
        return False

def Weight_Bias_Correction_Output(Outk, targets, Outj):
    # Correction of Weights and Bias between Hidden and Output Layer.
    # Calculate 𝑑𝑤𝑘𝑘𝑗 and 𝑑𝑏𝑘𝑘𝑗
    dwkkj = np.empty((0, len(Outk)))
    for i in range(len(Outj)):
        temp = (Outk - targets) * Outk * (1 - Outk) * Outj[i]
        dwkkj = np.vstack([dwkkj, temp])
    dbkkj = (Outk - targets) * Outk * (1 - Outk)
    dwkkj = dwkkj.T

    return dwkkj, dbkkj


def Weight_Bias_Correction_Hidden(inputs, outj, wkj, outk, targets):
    # Correction of Weights and Bias between Input and Hidden Layer.
    # Calculate 𝑑𝑤𝑗𝑗𝑖 and 𝑑𝑏𝑗𝑗�
    skl = (outk - targets) * outk * (1 - outk)
    dwjji = np.multiply.outer(outj * (1 - outj) * np.dot(skl, wkj), inputs)
    dbjii = outj * (1 - outj) * np.dot(skl, wkj)

    return dwjji, dbjii


def Weight_Bias_Update(wkj, dwkkj, bias_k, dbkkj, wji, dwjji, bias_j, dbjii):
    # Update Weights and Bias.
    # Calculate 𝑤𝑘𝑘𝑗+ and 𝑏𝑘𝑘𝑗+
    # Calculate 𝑤𝑗𝑗𝑖+ and 𝑏𝑗𝑗𝑖+
    n = 0.5
    wkjj = wkj - n * dwkkj
    bkkj = bias_k - n * dbkkj

    # Calculate 𝑤𝑗𝑗𝑖+ and 𝑏𝑗𝑗𝑖+
    wjji = wji - n * dwjji
    bjji = bias_j - n * dbjii

    return wkjj, bkkj, wjji, bjji

def Saving_Weights_Bias(wjji, bjji, wkkj, bkkj):
    # Save 𝑤𝑘𝑘𝑗 and 𝑏𝑘𝑘𝑗
    # Save 𝑤𝑗𝑗𝑖 and 𝑏𝑗𝑗�
    np.savez("weights_bias_values.npz", weights_ji=wjji, bias_j=bjji, weights_kj=wkkj, bias_k = bkkj)
    print("Saved")

def loading_Weights_Bias():
    data = np.load('Weights_bias_values.npz')

    wjji = data['weights_ji']
    bjji = data['bias_j']
    wkkj = data['weights_kj']
    bkkj = data['bias_k']

    return wjji, bjji, wkkj, bkkj


if __name__ == "__main__":
    """
    # Example Testing
    Input_Neurons = 2
    Hidden_Neurons = 2
    Output_Neurons = 2
    Global_error = 0.1

    # wji, wkj, bias_j, bias_k = Weight_Initialization()
    wji = np.array([[0.1, 0.2], [0.3, 0.4]])
    bias_j = np.array([0.2, 0.2])
    wkj = np.array([[0.5, 0.6], [0.7, 0.8]])
    bias_k = np.array([0.4, 0.4])
    targets = np.array([0.2,0.8])

    input_nodes= np.array([0.2, 0.5])

    net_j, out_j = Forward_Input_Hidden(input_nodes, wji, bias_j)
    net_k, out_k = Forward_Hidden_Output(out_j, wkj, bias_k)
    Check_for_End(out_k, targets)
    dwkkj, dbkkj = Weight_Bias_Correction_Output(out_j, out_k, wkj, bias_k, targets)
    Weight_Bias_Update(wkj, bias_k, dwkkj, dbkkj)
    """

    # Using real images
    INPUT_NEURONS = 28*28
    HIDDEN_NEURONS = 516
    OUTPUT_NEURONS = 20
    ERROR = 0.05
    ITERATIONS = 50
    i = 0
    j = 0

    target_fd = "./training"

    alphabets_targets = {'B':10, 'F':11, 'L':12, 'M':13, 'P':14, 'Q':15, 'T':16, 'U':17, 'V':18, 'W':19}

    for name in os.listdir(target_fd):
        label = name[0]
        try:
            label_value = int(label)
        except ValueError:
            label_value = alphabets_targets.get(label)
            assert label_value is not None, "label_value is None"

        targets = np.zeros(20)
        targets[label_value] = 1

        image = cv2.imread(os.path.join(target_fd, name))
        resized = cv2.resize(image, (28, 28))
        # convert picture to gray scale
        img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # _,img = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
        x_flattend = img_gray.reshape(1, 28 * 28)
        # plt.matshow(img_gray)
        x_flattend = np.squeeze(x_flattend)
        x_flattend = x_flattend / 255
        inputs = x_flattend

        # wji, wkj, bias_j, bias_k = None
        if (j == 0):
            wji, wkj, bias_j, bias_k = Weight_Initialization()
            j += 1
        for i in range(ITERATIONS):
            netj, outj = Forward_Input_Hidden(inputs, wji, bias_j)
            netk, outk = Forward_Hidden_Output(outj, wkj, bias_k)
            if Check_for_End(outk, targets, ERROR):
                break
            else:
                dwkkj, dbkkj = Weight_Bias_Correction_Output(outk, targets, outj)
                dwjji, dbjii = Weight_Bias_Correction_Hidden(inputs, outj, wkj, outk, targets)
                wkjj, bkkj, wjji, bjji = Weight_Bias_Update(wkj, dwkkj, bias_k, dbkkj, wji, dwjji, bias_j, dbjii)
            wji = wjji
            wkj = wkjj
            bias_j = bjji
            bias_k = bkkj

    print("Training completed, saving networks' configuration values")

    Saving_Weights_Bias(wji, bias_j, wkj, bias_k)

    loading_Weights_Bias()

