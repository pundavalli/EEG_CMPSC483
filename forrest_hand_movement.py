import numpy as np
import pandas as pd
from matplotlib.pyplot import clf
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import glob  # To load multiple files
import time
import joblib

# Start time
start_time = time.time()

# Load all CSV files in the left and right directories
# left_files = glob.glob('data_left/left*.csv')
# right_files = glob.glob('data_right/right*.csv')
left_files = glob.glob('p_left*/preprocessed_left*.csv')
right_files = glob.glob('p_right*/preprocessed_right*.csv')

#exg_columns = [f'EXG Channel {i}' for i in range(16)]
exg_columns = [f'Channel_{i}' for i in range(2, 18)]

all_data = []

# Process left hand data
for file in left_files:
    df = pd.read_csv(file)
    df['class'] = 1  # Left hand movement class 1

    # Keep only columns from 'EXG Channel 0' to 'EXG Channel 15'
    df = df[exg_columns + ['class']]  # Keep the EXG channels and the 'class' column

    all_data.append(df)

# Process right hand data
for file in right_files:
    df = pd.read_csv(file)
    df['class'] = 0  # Right hand movement class 0

    # Keep only columns from 'EXG Channel 0' to 'EXG Channel 15'
    df = df[exg_columns + ['class']]  # Keep the EXG channels and the 'class' column

    all_data.append(df)

# Combine all data into one dataset
eeg_data = pd.concat(all_data, ignore_index=True)

#print(eeg_data)

# Separate features and class
X = eeg_data.drop('class', axis=1) #Predictors
y = eeg_data['class'] #Target
'''
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
#classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, bootstrap=False, max_depth=32, max_features='sqrt', min_samples_leaf=1, min_samples_split=2)

# param_grid = {
#     'n_estimators': [10, 20, 30, 40, 50, 100],  # Number of trees
#     'max_depth': [None, 10, 20, 30],             # Depth of trees
#     'min_samples_split': [2, 5, 10],             # Minimum samples to split node
#     'min_samples_leaf': [1, 2, 4],               # Minimum samples at leaf node
#     'max_features': ['auto', 'sqrt', 'log2'],    # Number of features to consider for split
#     'bootstrap': [True, False]                   # Whether to use bootstrap samples
# }
# # Perform grid search with cross-validation
# classifier = GridSearchCV(classifier2, param_grid, cv=5, n_jobs=-1, verbose=2)
# Best hyperparameters found by GridSearchCV: {'bootstrap': False, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

classifier.fit(X_train, y_train)
'''
# # Print the best hyperparameters found by GridSearchCV
# print("Best hyperparameters found by GridSearchCV:", classifier.best_params_)

# Load serialized model
classifier = joblib.load("classifier.pkl")

# Make predictions
y_pred = classifier.predict(X)

# Print confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classifier.classes_)
disp.plot()
plt.show()

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy * 100}")
# Print classification report
print(classification_report(y, y_pred))

# End time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time: {elapsed_time:.2f} seconds")

'''
# #New Code for testing unseen data
file_test = 'preprocessed_both1.csv'
df_test = pd.read_csv(file_test)

X_test = df_test[exg_columns]
y_new = classifier.predict(X_test)

label_map = {1: "left", 0: "right"}
df_test["class"] = [label_map[pred] for pred in y_new]

# Display first 10 predictions
#print(df_test[exg_columns + ['class']].head())
print(df_test['class'].value_counts())
'''