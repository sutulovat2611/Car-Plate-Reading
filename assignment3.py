

def Weight_Initialization():
    # Initializing of the Weights. Random float number between -0.5 to 0.5 for weights.
    np.random.seed(1)
    wji= np.random.uniform(-0.5, 0.5, size=(Hidden_Neurons, Input_Neurons))
    wkj = np.random.uniform(-0.5, 0.5, size=(Output_Neurons, Hidden_Neurons))
    bias_j = np.random.uniform(0, 1, size=(Hidden_Neurons, 1))
    bias_k = np.random.uniform(0, 1, size=(Output_Neurons, 1))

# def Read_Files():
#     # Reading of Segmented Training Files, and Target Files.


# def Forward_Input_Hidden():
#     # Forward Propagation from Input -> Hidden Layer.
#     # Obtain the results at each neuron in the hidden layer.
#     # Calculate 𝑁𝑒𝑡𝑗and 𝑂𝑢𝑡𝑗


# def Check_for_End():
#     # Check whether the total error is less than the error set by the user or the number of iterations is reached.
#     # returns true or false


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