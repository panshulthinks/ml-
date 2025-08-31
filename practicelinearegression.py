import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([600, 800, 1000, 1200, 1400])
y = np.array([150, 200, 250, 300, 350])

# Feature scaling
x_mean = x.mean()
x_std = x.std()
x_norm = (x - x_mean) / x_std

# Hyperparameters
w = 0.0
b = 0.0
L = 0.01      # learning rate
epochs = 1000 # number of iterations

def compute_cost(x, y, w, b):
    """
    Computes the MSE cost J(w,b) = (1/2m) * sum((w*x[i] + b - y[i])^2)
    """
    m = len(y)
    total_cost = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        total_cost += (f_wb - y[i]) ** 2
    return total_cost / (2 * m)  # divide after the loop

def compute_gradient(x, y, w, b):
    """
    Computes the gradients dj_dw and dj_db for linear regression.
    """
    m = len(y)
    dj_dw = 0.0
    dj_db = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        dj_dw += error * x[i]  # partial derivative wrt w
        dj_db += error         # partial derivative wrt b
    return dj_dw / m, dj_db / m  # average over all examples

# Gradient descent loop
for i in range(epochs):
    dj_dw, dj_db = compute_gradient(x_norm, y, w, b)
    w -= L * dj_dw
    b -= L * dj_db
    if i % 100 == 0:
        cost = compute_cost(x_norm, y, w, b)
        print(f"Iteration {i:4d}: Cost {cost:.4f}, w = {w:.4f}, b = {b:.4f}")

# Final parameters
print(f"\nFinal parameters: w = {w:.4f}, b = {b:.4f}")

# Plot
plt.scatter(x, y, label="Data")
plt.plot(x, w * ((x - x_mean) / x_std) + b, color='red', label="Model")
plt.xlabel('Size (sqft)')
plt.ylabel('Price (1000s)')
plt.legend()
plt.show()




