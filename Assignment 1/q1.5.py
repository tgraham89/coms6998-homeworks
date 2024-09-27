import numpy as np
from collections import defaultdict
from sklearn.linear_model import Ridge

# Hyperparameters
np.random.seed(42)
dsize = 50        # Size of each dataset
n_dataset = 100   # Number of datasets
n_trainset = int(np.ceil(dsize * 0.8))  # Size of training set
polynomial_degrees = range(1, 16)  # Model complexities
regularization_rate = 5.0  # Regularization rate for Ridge regression

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

            # Train Ridge regression model for degree 10
            if degree == 10:
                poly_features = np.vander(x_train, N=degree + 1, increasing=True)
                ridge_model = Ridge(alpha=regularization_rate)
                ridge_model.fit(poly_features, y_train)
                ridge_pred_train = ridge_model.predict(poly_features)
                ridge_pred_test = ridge_model.predict(np.vander(x_test, N=degree + 1, increasing=True))
                pred_train['ridge_10'].append(ridge_pred_train)
                pred_test['ridge_10'].append(ridge_pred_test)
                train_errors['ridge_10'].append(np.mean(error(ridge_pred_train, y_train)))
                test_errors['ridge_10'].append(np.mean(error(ridge_pred_test, y_test)))

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

# Calculate bias, variance, and errors for Ridge regression model
ridge_bias_squared = calculate_estimator_bias_squared(pred_test['ridge_10'])
ridge_variance = calculate_estimator_variance(pred_test['ridge_10'])
ridge_train_error = np.mean(train_errors['ridge_10'])
ridge_test_error = np.mean(test_errors['ridge_10'])

# Print the results for Ridge regression model
print(f"Ridge Regression (degree 10) Bias^2: {ridge_bias_squared}")
print(f"Ridge Regression (degree 10) Variance: {ridge_variance}")
print(f"Ridge Regression (degree 10) Train Mean Squared Error: {ridge_train_error}")
print(f"Ridge Regression (degree 10) Test Mean Squared Error: {ridge_test_error}")

# Print the results for unregularized 10th-degree polynomial model
print(f"Unregularized (degree 10) Bias^2: {bias_squared[9]}")
print(f"Unregularized (degree 10) Variance: {variance[9]}")
print(f"Unregularized (degree 10) Train Mean Squared Error: {complexity_train_error[9]}")
print(f"Unregularized (degree 10) Test Mean Squared Error: {complexity_test_error[9]}")
