# Practical 03: Implement the following.

# A. Write a program to implement Hebb's rule.

w1 = 0
w2 = 0
b = 0

x1 = [-1, -1, 1, 1]
x2 = [-1, 1, -1, 1]
y = [-1, 1, 1, 1]

def train_perceptron(x1, x2, y):
    global w1, w2, b
    for i in range(len(x1)):
        w1 += x1[i] * y[i]
        w2 += x2[i] * y[i]
        b += y[i]
        print("Epoch {}: w1_new={}, w2_new={}, b_new={}".format(i+1, w1, w2, b))

# Training with OR examples
print("Training with OR examples:")
train_perceptron(x1, x2, y)

# Display the final weights and bias
print("\nThe Final Weights are:")
print("w1_new =", w1)
print("w2_new =", w2)
print("b_new =", b)


# B. Write a program to implement of Delta rule.

import numpy as np

# Input data for OR function
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target values for OR function
Y = np.array([0, 1, 1, 1])

# Initialize weights and bias
weights = np.random.rand(2)  # Initialize with random values
bias = np.random.rand(1)

# Learning rate
learning_rate = 0.1

# Number of epochs
epochs = 200

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def delta_rule_train(X, Y, weights, bias, learning_rate, epochs):
    for epoch in range(epochs):
        for i in range(len(X)):
            # Forward pass
            output = sigmoid(np.dot(X[i], weights) + bias)
            # Compute error
            error = Y[i] - output
            # Update weights and bias using the Delta rule
            weights += learning_rate * error * X[i]
            bias += learning_rate * error
        # Display the error for every 100 epochs
        if epoch % 100 == 0:
            total_error = np.mean(np.square(Y - sigmoid(np.dot(X, weights) + bias)))
            print("Epoch {}: Total Error: {}".format(epoch, total_error))
    return weights, bias

# Train the perceptron using the Delta rule
trained_weights, trained_bias = delta_rule_train(X, Y, weights, bias, learning_rate, epochs)

# Display the final weights and bias rounded to two decimal places
print("\nThe Final Weights are:")
print("w1_new =", round(trained_weights[0], 2))
print("w2_new =", round(trained_weights[1], 2))
print("b_new =", round(trained_bias[0], 2))

