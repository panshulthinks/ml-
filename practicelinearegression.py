import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

## here we are just making a simple linear regression model to predict the house price based on size
## lets load the data
x = np.array([600, 800, 1000, 1200, 1400])
y = np.array([150, 200, 250, 300, 350])

## we will use feature scaling beacause the values are small
x_mean = x.mean()
x_std = x.std()
x_norm = (x - x_mean) / x_std

## lets initialis parameters
w = 0
b = 0
L = 0.01
epochs = 1

## lets compute the cost function
def compute_cost(x, y, w, b):
    m = len(y)
    total_cost = 0
    for i in range(m):
        f_wb = w * x[] + b
        total_cost += (f_wb - y[i]) ** 2
        total_cost = total_cost / (2 * m)
        return total_cost

## lets compute gradient descent
def compute_gradient(x, y, w, b):
    m = len(y)
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_dw += (f_wb - y[i])
        dj_dw = dj_dw/m
        dj_db = dj_db/m
        return dj_dw, dj_db
    
## lets do gradient descent
for i in range(epochs):
    dj_dw, dj_db = compute_gradient(x_norm, y, w, b)
    w = w - L * dj_dw
    b = b - L * dj_db
    if (i%1000 == 0):
        cost = compute_cost(x_norm, y, w, b)
        print(f"Iteration {i}: Cost {cost}, w: {w}, b: {b}")    
        print(f"Final parameters: w: {w}, b: {b}")

## lets plot
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, w*x + b, color='red')
plt.xlabel('Size (sqft)')
plt.ylabel('Price (1000s)')
plt.show()




