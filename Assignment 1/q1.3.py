import numpy as np
import matplotlib.pyplot as plt

# Recreate x_data and y_data from previous step
def generate_y(x):
    epsilon = np.random.normal(0, np.sqrt(0.3), size=x.shape)
    return x + np.sin(1.5 * x) + epsilon

np.random.seed(42)
x_data = np.random.uniform(0, 10, 20)
y_data = generate_y(x_data)

# Given function for f(x)
def f(x):
    return x + np.sin(1.5 * x)

# Define polynomial degrees
degrees = [1, 3, 10]
polynomial_fits = []

# Fit the polynomial models using np.polyfit
for degree in degrees:
    coefficients = np.polyfit(x_data, y_data, degree)  # Fit polynomial of given degree
    polynomial_fits.append((degree, coefficients))

# Prepare for plotting
x_smooth = np.linspace(0, 10, 1000)

# Plot the original data and the fitted polynomials
plt.figure(figsize=(10, 8))

# Original function f(x)
plt.plot(x_smooth, f(x_smooth), label="f(x) = x + sin(1.5x)", color='red', linewidth=2)

# Original y(x) data
plt.scatter(x_data, y_data, label="y(x) = x + sin(1.5x) + N(0, 0.3)", color='blue')

# Plot the polynomial estimators using np.polyval
colors = ['green', 'orange', 'purple']
for i, (degree, coefficients) in enumerate(polynomial_fits):
    y_smooth_poly = np.polyval(coefficients, x_smooth)
    plt.plot(x_smooth, y_smooth_poly, label=f"g_{degree}(x) Polynomial Degree {degree}", color=colors[i], linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Estimators for Polynomial Degrees 1, 3, and 10")
plt.legend()
plt.grid(True)
plt.show()

# Display the coefficients of each polynomial
for degree, coefficients in polynomial_fits:
    print(f"g_{degree}(x) Coefficients: {coefficients}")





# np.random.seed(0)  # For reproducibility
# x_range = np.arange(0,10,0.1)
# std = 0.3**(1/2)
# x_fit = np.random.uniform(0, 10, 20)
# polynomial_degrees = [1, 3, 10]
# theta = {}
# fit = {}
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
#     for i, degree in enumerate(polynomial_degrees):
#         theta[degree] = np.polyfit(x_fit, y, degree)
#         fit[degree] = np.polyval(theta[degree], x_grid)
#         plt.plot(x_grid, fit[degree], label="g_(" + str(degree) + ")")


# x_grid = np.linspace(0, 10, 100)
# plotFunc(func)

# plt.legend()
# # plt.xlim([0, 10])
# plt.ylim([0, 12])
# plt.show()