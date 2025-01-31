# Practical 02: Implement the following.

# A. Generate AND/NOT function using McCulloch Pitts neural net.


def mc_culloch_pitts_neuron(inputs, weights, threshold):
    # Calculate the weighted sum of inputs
    net_input = sum(i * w for i, w in zip(inputs, weights))

    # Apply threshold function: if net_input >= threshold, output 1, else output 0
    if net_input >= threshold:
        return 1
    else:
        return 0

# AND function using McCulloch-Pitts model
def and_function(x1, x2):
    # Define weights and threshold
    weights = [1, 1]  # weights for AND function
    threshold = 2  # Threshold for AND function (2 because both inputs need to be 1)
    inputs = [x1, x2]
    return mc_culloch_pitts_neuron(inputs, weights, threshold)

# NOT function using McCulloch-Pitts model
def not_function(x1):
    # Define weight and threshold for NOT function
    weights = [-1]  # weight for NOT function (inverting input)
    threshold = 0  # Threshold for NOT function (to flip the input)
    inputs = [x1]
    return mc_culloch_pitts_neuron(inputs, weights, threshold)

# Test the AND function
print("AND Function:")
print("0 AND 0 =", and_function(0, 0))  # Expected output: 0
print("0 AND 1 =", and_function(0, 1))  # Expected output: 0
print("1 AND 0 =", and_function(1, 0))  # Expected output: 0
print("1 AND 1 =", and_function(1, 1))  # Expected output: 1

# Test the NOT function
print("\nNOT Function:")
print("NOT 0 =", not_function(0))  # Expected output: 1
print("NOT 1 =", not_function(1))  # Expected output: 0


# B. Generate XOR function using McCulloch-Pitts neural net.

import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights and biases for the network
input_size = 2
hidden_size = 2
output_size = 1

# Weights for the input layer to the hidden layer
w_input_hidden = np.array([[20, -20], [20, -20]])

# Weights for the hidden layer to the output layer
w_hidden_output = np.array([[20], [20]])

# Bias for the hidden layer
b_hidden = np.array([[-10, 30]])

# Bias for the output layer
b_output = np.array([[-30]])

# Define the XOR function
def xor(x1, x2):
    input_data = np.array([[x1, x2]])
    # Calculate the hidden layer output
    hidden_input = np.dot(input_data, w_input_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)
    # Calculate the output
    output_input = np.dot(hidden_output, w_hidden_output) + b_output
    final_output = sigmoid(output_input)
    return final_output[0][0]

# Test the XOR function
print("XOR(0, 0) =", xor(0, 0))
print("XOR(0, 1) =", xor(0, 1))
print("XOR(1, 0) =", xor(1, 0))
print("XOR(1, 1) =", xor(1, 1))

