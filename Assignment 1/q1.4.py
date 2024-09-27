import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# Hyperparameters
np.random.seed(42)
dsize = 50        # Size of each dataset
n_dataset = 100   # Number of datasets
n_trainset = int(np.ceil(dsize * 0.8))  # Size of training set
polynomial_degrees = range(1, 16)  # Model complexities

# Fixed x values
x = np.linspace(0, 5, dsize)
x = np.random.permutation(x)
x_train = x[:n_trainset]
x_test = x[n_trainset:]

# Variables to store results
coefficient = []
pred_train = defaultdict(list)
pred_test = defaultdict(list)
train_errors = defaultdict(list)
test_errors = defaultdict(list)

# Recreate x_data and y_data from previous step
def generate_y(x):
    epsilon = np.random.normal(0, np.sqrt(0.3), size=x.shape)
    return x + np.sin(1.5 * x) + epsilon

# Given function for f(x)
def f(x):
    return x + np.sin(1.5 * x)

# Error calculation function
def error(pred, actual):
    return (pred - actual) ** 2

# Train models over datasets and polynomial degrees
def train_over_polynomial_degrees():
    for dataset in range(n_dataset):
        # Simulate training/testing targets with noise
        y_train = generate_y(x_train)
        y_test = generate_y(x_test)

        # Loop over model complexities (polynomial degrees)
        for degree in polynomial_degrees:
            # Train model
            tmp_coefficient = np.polyfit(x_train, y_train, degree)

            # Make predictions on train set
            tmp_pred_train = np.polyval(tmp_coefficient, x_train)
            pred_train[degree].append(tmp_pred_train)

            # Test predictions
            tmp_pred_test = np.polyval(tmp_coefficient, x_test)
            pred_test[degree].append(tmp_pred_test)

            # Mean Squared Error for train and test sets
            train_errors[degree].append(np.mean(error(tmp_pred_train, y_train)))
            test_errors[degree].append(np.mean(error(tmp_pred_test, y_test)))

# Calculate squared bias
def calculate_estimator_bias_squared(pred_test):
    pred_test = np.array(pred_test)
    average_model_prediction = pred_test.mean(0)  # E[g(x)]
    return np.mean((average_model_prediction - f(x_test)) ** 2)

# Calculate variance
def calculate_estimator_variance(pred_test):
    pred_test = np.array(pred_test)
    average_model_prediction = pred_test.mean(0)  # E[g(x)]
    return np.mean((pred_test - average_model_prediction) ** 2)

# Perform the training and evaluation
train_over_polynomial_degrees()

# Initialize lists to store results for bias, variance, and error
complexity_train_error = []
complexity_test_error = []
bias_squared = []
variance = []

# Loop over each degree and compute bias, variance, and errors
for degree in range(1, 16):
    complexity_train_error.append(np.mean(train_errors[degree]))
    complexity_test_error.append(np.mean(test_errors[degree]))
    bias_squared.append(calculate_estimator_bias_squared(pred_test[degree]))
    variance.append(calculate_estimator_variance(pred_test[degree]))

# Identify the best model based on test error
best_model_degree = polynomial_degrees[np.argmin(complexity_test_error)]

# Plot the results: Bias, Variance, and Testing Set Error
plt.figure(figsize=(10, 8))
plt.plot(polynomial_degrees, bias_squared, label='$bias^2$', color='red', linewidth=2)
plt.plot(polynomial_degrees, variance, label='Variance', color='blue', linewidth=2)
plt.plot(polynomial_degrees, complexity_test_error, label='Testing Set Error', linewidth=3, color='green')
plt.axvline(best_model_degree, linestyle='--', color='black', label=f'Best Model (degree={best_model_degree})')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylim([0, 1])
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.grid(True)
plt.show()
