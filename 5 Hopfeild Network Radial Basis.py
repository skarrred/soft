# Practical 05: Implement the following.

# A. Write a program for Hopfield Network.

import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.reshape(pattern, (self.size, 1))
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def predict(self, input_pattern, max_iter=100):
        input_pattern = np.reshape(input_pattern, (self.size, 1))
        for _ in range(max_iter):
            output_pattern = np.sign(np.dot(self.weights, input_pattern))
            if np.array_equal(input_pattern, output_pattern):
                return output_pattern.flatten()
            input_pattern = output_pattern
        raise RuntimeError("Hopfield Network did not converge within the maximum number of iterations.")

# Example usage:
# Define patterns for training
pattern1 = np.array([1, -1, 1, -1])
pattern2 = np.array([-1, -1, -1, 1])
patterns = [pattern1, pattern2]

# Create a Hopfield Network and train it
hopfield_net = HopfieldNetwork(size=len(pattern1))
hopfield_net.train(patterns)

# Test the Hopfield Network with a noisy input pattern
noisy_pattern = np.array([1, 1, 1, 1])
predicted_pattern = hopfield_net.predict(noisy_pattern)

# Display results
print("Noisy Pattern:", noisy_pattern)
print("Predicted Pattern:", predicted_pattern)



# B. Write a program for Radial Basis function.

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=200, n_features=2, n_classes=2, n_clusters_per_class=1,
    n_informative=2, n_redundant=0, n_repeated=0, random_state=42
)

# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("Generated Dataset")
plt.show()

# RBF Network Parameters
num_centers = 10  # Number of RBF centers
learning_rate = 0.01
num_iterations = 100

# Define the RBF network class
class RBFNetwork:
    def __init__(self, num_centers):
        self.num_centers = num_centers
        self.centers = None
        self.sigmas = None
        self.weights = None

    def _rbf(self, x, center, sigma):
        """Gaussian RBF activation function."""
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * sigma ** 2))

    def _calculate_activations(self, X):
        """Calculate the activation of each RBF neuron for all data points."""
        activations = np.zeros((X.shape[0], self.num_centers))
        for i, center in enumerate(self.centers):
            activations[:, i] = np.array([self._rbf(x, center, self.sigmas[i]) for x in X])
        return activations

    def fit(self, X, y):
        """Train the RBF Network."""
        # Step 1: Use KMeans to find the RBF centers
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42).fit(X)
        self.centers = kmeans.cluster_centers_
        # Step 2: Calculate the sigma for each RBF neuron
        d_max = np.max([np.linalg.norm(c1 - c2) for c1 in self.centers for c2 in self.centers])
        self.sigmas = np.full(self.num_centers, d_max / np.sqrt(2 * self.num_centers))
        # Step 3: Calculate the activations for all data points
        activations = self._calculate_activations(X)
        # Step 4: Solve for weights using linear regression
        self.weights = np.linalg.pinv(activations).dot(y)

    def predict(self, X):
        """Predict class labels for input data."""
        activations = self._calculate_activations(X)
        y_pred = activations.dot(self.weights)
        return np.round(y_pred).astype(int)

# Train the RBF Network
rbf_net = RBFNetwork(num_centers=num_centers)
rbf_net.fit(X, y)

# Predict on the same dataset
y_pred = rbf_net.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = rbf_net.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title("RBF Network Decision Boundary")
plt.show()

