import numpy as np
import matplotlib.pyplot as plt

# Given functions and parameters
def f(x):
    return x + np.sin(1.5 * x)

# Random noise generator from N(0, 0.3)
def generate_y(x):
    epsilon = np.random.normal(0, np.sqrt(0.3), size=x.shape)
    # epsilon = np.random.randn * (0.3**(1/2))
    return x + np.sin(1.5 * x) + epsilon

# Generate dataset of size 20
np.random.seed(42)  # For reproducibility
x_data = np.random.uniform(0, 10, 20)
y_data = generate_y(x_data)

# Plotting the scatter plot for y_data and smooth line for f(x)
x_smooth = np.linspace(0, 10, 1000)  # Smooth x values for f(x)
f_values = f(x_smooth)

plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, label="y(x) = x + sin(1.5x) + N(0, 0.3)", color='blue')
plt.plot(x_smooth, f_values, label="f(x) = x + sin(1.5x)", color='red', linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("20 Point Sampled Dataset and f(x)")
plt.legend()
plt.grid(True)
plt.show()



# random.seed(0)
# np.random.seed(0)  # For reproducibility
# x_range = np.arange(0,10,0.1)
# std = 0.3**(1/2)
# x_fit = np.random.uniform(0, 10, 20)
# def func(x):
#     return np.sin(1.5*x) + x
# def addNoisetoFunc(f, shape, std):
#     return f + np.random.randn(*shape) * std
# def plotFunc(f):
#     #plot func
#     plt.plot(x_range, f(x_range), label='f(x)')
    
#     #scatter points
#     y = addNoisetoFunc(func(x_fit), x_fit.shape, std)
#     plt.scatter(x_fit, y, label='y')
    
# plotFunc(func)
# plt.legend()
plt.show()