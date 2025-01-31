# Practical 01: Implement the following.

# A. Design a simple linear neural network model

x1 = float(input("Enter Value of X1: "))
x2 = float(input("Enter Value of X2: "))

# Weight values
w1 = float(input("Enter Value of W1: "))
w2 = float(input("Enter Value of W2: "))

# Calculate the net input
yin = x1 * w1 + x2 * w2

print("Net Input to the Output Neuron =", yin)

# Apply the step function to determine the output
if yin <= 0:
    yout = 0
else:
    yout = 1

print("The Output of the given neural network =", yout)


# B. Neural net using both binary and bipolar sigmoidal function.

import numpy as np

# Input values
x1 = float(input("Enter Value of X1: "))
x2 = float(input("Enter Value of X2: "))

# Weight values
w1 = float(input("Enter Value of W1: "))
w2 = float(input("Enter Value of W2: "))

# Bias term
b = 1

# Calculate the net input
yin = b + (x1 * w1 + x2 * w2)

print("Net Input to the Output Neuron =", yin)

# Apply the binary sigmoid activation function
binary_output = 1 / (1 + np.exp(-yin))
print("Binary Output =", round(binary_output, 4))

# Apply the bipolar sigmoid activation function
bipolar_output = (2 / (1 + np.exp(-yin))) - 1
print("Bipolar Output =", round(bipolar_output, 4))
