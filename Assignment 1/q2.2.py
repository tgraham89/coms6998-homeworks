import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


file_path = 'csv_result-dataset_31_credit-g.csv'
data = pd.read_csv(file_path)


# Cleaning up the column names by removing extra quotation marks and spaces
data.columns = data.columns.str.replace("'", "").str.strip()

# The target variable is 'class', and the features are the remaining columns (excluding 'id')
X = data.drop(columns=['class', 'id'])
y = data['class'].replace({'good': 1, 'bad': 0})  # Converting target to binary (1 = good, 0 = bad)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Some of the features are categorical, so we will need to encode them properly
# Using OneHotEncoder to handle categorical variables

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Create a column transformer to handle categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Keep numerical columns as they are
)

# Pipeline to include preprocessing and the classifiers
# AdaBoost pipeline
ada_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', AdaBoostClassifier(random_state=0))])

# Logistic Regression pipeline
logreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(random_state=0, max_iter=1000))])

# Training both pipelines
ada_pipeline.fit(X_train, y_train)
logreg_pipeline.fit(X_train, y_train)

# Generate ROC curve data for both classifiers
fpr_adaboost, tpr_adaboost, _ = roc_curve(y_test, ada_pipeline.predict_proba(X_test)[:, 1])
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, logreg_pipeline.predict_proba(X_test)[:, 1])

# Generate Precision-Recall curve data for both classifiers
precision_adaboost, recall_adaboost, _ = precision_recall_curve(y_test, ada_pipeline.predict_proba(X_test)[:, 1])
precision_logistic, recall_logistic, _ = precision_recall_curve(y_test, logreg_pipeline.predict_proba(X_test)[:, 1])

# Define the point for an all-positive classifier in ROC and PR spaces
pos_ratio = np.sum(y_test) / len(y_test)
all_positive_roc = (1, 1)
all_positive_pr = (pos_ratio, 1)

print("pos_ratio: ", pos_ratio)

# Plotting the ROC and PR curves
plt.figure(figsize=(10, 5))

# ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr_adaboost, tpr_adaboost, label="AdaBoost ROC")
plt.plot(fpr_logistic, tpr_logistic, label="Logistic Regression ROC")
plt.scatter(*all_positive_roc, color='red', label='All Positive Classifier', zorder=5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

# PR Curve
plt.subplot(1, 2, 2)
plt.plot(recall_adaboost, precision_adaboost, label="AdaBoost PR")
plt.plot(recall_logistic, precision_logistic, label="Logistic Regression PR")
plt.scatter(*all_positive_pr, color='red', label='All Positive Classifier', zorder=5)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curves')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()

# Calculating AUROC for both classifiers
auroc_adaboost = auc(fpr_adaboost, tpr_adaboost)
auroc_logistic = auc(fpr_logistic, tpr_logistic)

# Calculating AUPR for both classifiers
aupr_adaboost = auc(recall_adaboost, precision_adaboost)
aupr_logistic = auc(recall_logistic, precision_logistic)

# To calculate AUPRG (Area under PR Gain Curve), we need to compute Precision Gain and Recall Gain
def calculate_prg(recall, precision, pos_ratio):
    # Precision Gain and Recall Gain as defined in the PRG paper
    precision_gain = np.where(precision > pos_ratio, (precision - pos_ratio) / ((1 - pos_ratio) * precision), 0)
    recall_gain = np.where(recall > pos_ratio, (recall - pos_ratio) / ((1 - pos_ratio) * recall), 0)
    return recall_gain, precision_gain

# Calculating PRG for AdaBoost
recall_gain_adaboost, precision_gain_adaboost = calculate_prg(recall_adaboost, precision_adaboost, pos_ratio)
auprg_adaboost = auc(recall_gain_adaboost, precision_gain_adaboost)

# Calculating PRG for Logistic Regression
recall_gain_logistic, precision_gain_logistic = calculate_prg(recall_logistic, precision_logistic, pos_ratio)
print("recall_gain_logistic: ", recall_gain_logistic)
print("precision_gain_logistic: ", precision_gain_logistic)
auprg_logistic = auc(recall_gain_logistic, precision_gain_logistic)

results = {
    'Classifier': ['AdaBoost', 'Logistic Regression'],
    'AUROC': [auroc_adaboost, auroc_logistic],
    'AUPR': [aupr_adaboost, aupr_logistic],
    'AUPRG': [auprg_adaboost, auprg_logistic]
}

results_df = pd.DataFrame(results)

print(results_df)