# Practical 06: Implement the following.

# A. Kohonen Self organizing map.

import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters
grid_size = (10, 10)  # SOM grid size (10x10 neurons)
input_dim = 3  # Input dimension (e.g., 3 for RGB color data)
learning_rate = 0.5  # Initial learning rate
num_iterations = 1000  # Number of training iterations
decay_rate = 0.1  # Learning rate decay
neighborhood_radius = 3  # Initial neighborhood radius

# Generate some random input data (e.g., RGB colors)
data = np.random.rand(300, input_dim)

# Initialize the SOM grid with random weights
som_grid = np.random.rand(grid_size[0], grid_size[1], input_dim)

# Define functions to help with SOM training
def get_bmu(data_point, som_grid):
    """Find the Best Matching Unit (BMU) in the SOM grid for a data point."""
    distances = np.linalg.norm(som_grid - data_point, axis=2)
    return np.unravel_index(np.argmin(distances), distances.shape)

def update_weights(som_grid, data_point, bmu, learning_rate, radius):
    """Update weights in the neighborhood of the BMU."""
    for x in range(som_grid.shape[0]):
        for y in range(som_grid.shape[1]):
            distance = np.sqrt((x - bmu[0])**2 + (y - bmu[1])**2)
            if distance <= radius:
                influence = np.exp(-distance**2 / (2 * (radius**2)))
                som_grid[x, y] += influence * learning_rate * (data_point - som_grid[x, y])

# Train the SOM
for i in range(num_iterations):
    # Select a random data point
    data_point = data[np.random.randint(0, data.shape[0])]
    # Find the BMU for this data point
    bmu = get_bmu(data_point, som_grid)
    # Update weights of SOM grid within the neighborhood
    update_weights(som_grid, data_point, bmu, learning_rate, neighborhood_radius)
    # Decay learning rate and neighborhood radius over time
    learning_rate *= (1 - decay_rate)
    neighborhood_radius *= (1 - decay_rate)

# Visualize the SOM grid
plt.imshow(som_grid)
plt.title("Kohonen Self-Organizing Map")
plt.show()
