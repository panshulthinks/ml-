import numpy as np

# Dataset: three points (x, y)
x = np.array([1.0, 2.0, 4.0])
y = np.array([2.0, 3.0, 6.0])
m = x.shape[0]   # m = 3

# Initialize parameters
w = 1.0
b = 0.0
alpha = 0.1      # learning rate
iterations = 1   # you can increase for multiple steps

def compute_cost(x, y, w, b):
    """
    Computes J(w,b) = (1/(2m)) * Σ (w*x[i] + b - y[i])^2
    """
    m = x.shape[0]
    total = 0.0
    for i in range(m):
        pred = w * x[i] + b
        total += (pred - y[i])**2
    return total / (2 * m)

def compute_gradient(x, y, w, b):
    """
    Computes gradients:
      dj_dw = (1/m) * Σ (w*x[i] + b - y[i]) * x[i]
      dj_db = (1/m) * Σ (w*x[i] + b - y[i])
    """
    m = x.shape[0]
    dj_dw = 0.0
    dj_db = 0.0
    for i in range(m):
        pred = w * x[i] + b
        error = pred - y[i]
        dj_dw += error * x[i]
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# Before update
print("Initial cost:", compute_cost(x, y, w, b))
print("Initial gradients:", compute_gradient(x, y, w, b))

# Perform one gradient descent step
for it in range(iterations):
    dj_dw, dj_db = compute_gradient(x, y, w, b)
    w -= alpha * dj_dw
    b -= alpha * dj_db

# After update
print("Updated w, b:", w, b)
print("Cost after update:", compute_cost(x, y, w, b))
