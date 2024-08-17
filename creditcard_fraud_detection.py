import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv('fraudtrain.csv')
test_data = pd.read_csv('fraudtest.csv')

# Display basic information about the datasets
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print("Train data columns:", train_data.columns)
print("Test data columns:", test_data.columns)

# Check for missing values
print("Missing values in train data:\n", train_data.isnull().sum())
print("Missing values in test data:\n", test_data.isnull().sum())

# Separate features and target variable from training data
X = train_data.drop('Class', axis=1)  # 'Class' is the target column
y = train_data['Class']

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_data)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = model.predict(X_val)

# Evaluate the model
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# Predict on the test set
y_test_pred = model.predict(X_test)

# Optional: Displaying the results
print("Test Predictions:\n", y_test_pred)

# If needed, save the predictions to a CSV file
output = pd.DataFrame({'Id': test_data.index, 'Class': y_test_pred})
output.to_csv('submission.csv', index=False)
