# Practical 07: Implement the following.

# A. Write a program for Linear separation.

import numpy as np
import matplotlib.pyplot as plt

# Function to plot the data points and the decision boundary
def plot_data(X, y, weights):
    plt.figure(figsize=(8, 6))
    # Plot the data points
    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X[i][0], X[i][1], color='blue', marker='o', label='Class 1' if i == 0 else "")
        else:
            plt.scatter(X[i][0], X[i][1], color='red', marker='x', label='Class -1' if i == 0 else "")
    # Plot decision boundary: w0 + w1*x1 + w2*x2 = 0 -> x2 = -(w0 + w1*x1)/w2
    x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2 = -(weights[0] + weights[1] * x1) / weights[2]
    plt.plot(x1, x2, color='green', label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Data points and Decision Boundary')
    plt.grid(True)
    plt.show()

# Perceptron learning algorithm
def perceptron(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    for epoch in range(epochs):
        for idx, x_i in enumerate(X):
            linear_output = np.dot(x_i, weights) + bias
            y_predicted = np.sign(linear_output)
            if y_predicted != y[idx]:
                # Update rule
                update = learning_rate * y[idx]
                weights += update * x_i
                bias += update
    return weights, bias

# Generate linearly separable data
X = np.array([[2, 3], [3, 4], [4, 5], [5, 6], [3, 8], [2, 7], [2, 5], [1, 7]])
y = np.array([1, 1, 1, 1, -1, -1, -1, -1])  # Class labels (-1 and 1)

# Train the Perceptron model
weights, bias = perceptron(X, y)

# Print the weights and bias
print(f"Weights: {weights}, Bias: {bias}")

# Add bias to weights for plotting
weights = np.insert(weights, 0, bias)

# Plot the data points and the decision boundary
plot_data(X, y, weights)
