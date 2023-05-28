import numpy as np
import math

OUTPUT_NEURONS = 20
INPUT_NEURONS = 28 * 28
HIDDEN_NEURONS = 16

def Weight_Initialization():
    # Initializing of the Weights. Random float number between -0.5 to 0.5 for weights.
    np.random.seed(1)
    wji= np.random.uniform(-0.5, 0.5, size=(HIDDEN_NEURONS, INPUT_NEURONS))
    wkj = np.random.uniform(-0.5, 0.5, size=(OUTPUT_NEURONS, HIDDEN_NEURONS))
    bias_j = np.random.uniform(0, 1, size=(HIDDEN_NEURONS, 1))
    bias_k = np.random.uniform(0, 1, size=(OUTPUT_NEURONS, 1))

# def Read_Files():
#     # Reading of Segmented Training Files, and Target Files.


def Forward_Input_Hidden(inputs,weights, biases):
    # Forward Propagation from Input -> Hidden Layer.
    # Obtain the results at each neuron in the hidden layer.
    # Calculate 𝑁𝑒𝑡𝑗and 𝑂𝑢𝑡𝑗
    Netj = np.dot(inputs,weights.T) 
    Outj = 1/(1 + math.e**-(Netj + np.transpose(biases)))

def Forward_Hidden_Output(inputs,weights, biases):
#     # Forward Propagation from Hidden -> Output Layer.
#     # Calculate 𝑁𝑒𝑡𝑘and 𝑂𝑢𝑡𝑘
    Netk = np.dot(inputs,weights.T) 
    Outk = 1/(1 + math.e**-(Netk + np.transpose(biases)))


def Check_for_End(user_set, outs, targets):
    # Check whether the total error is less than the error set by the user or the number of iterations is reached.
    # returns true or false
    def Error_Correction(outs, targets):
        result = ((outs - targets)**2)/2
        return result
    
    if Error_Correction(outs, targets) < user_set:
        return True
    else:
        return False


# def Weight_Bias_Correction_Output():
#     # Correction of Weights and Bias between Hidden and Output Layer.
#     # Calculate 𝑑𝑤𝑘𝑘𝑗 and 𝑑𝑏𝑘𝑘𝑗

# def Weight_Bias_Correction_Hidden():
# # Correction of Weights and Bias between Input and Hidden Layer.
# # Calculate 𝑑𝑤𝑗𝑗𝑖 and 𝑑𝑏𝑗𝑗𝑖

# def Weight_Bias_Update():
# # Update Weights and Bias.
# # Calculate 𝑤𝑘𝑘𝑗+ and 𝑏𝑘𝑘𝑗+
# # Calculate 𝑤𝑗𝑗𝑖+ and 𝑏𝑗𝑗𝑖


# def Saving_Weights_Bias()
# # Save 𝑤𝑘𝑘𝑗 and 𝑏𝑘𝑘𝑗
# # Save 𝑤𝑗𝑗𝑖 and 𝑏𝑗𝑗𝑖


# i) Set the number of input features,
# ii) Number of hidden neurons.
# iii) Number of output neurons.
# iv) Global error or number of iterations.
# v) Any other parameters that you may want to initialize in your program.