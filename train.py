import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump

from preprocess import load_dataset

def train_svm_model(base_dir, class_mapping, normalize_len=True):
    # Load the dataset
    print("Loading and processing dataset...")
    X, y = load_dataset(base_dir, class_mapping, normalize_len)
    
    # Check if we have enough data
    if len(X) == 0:
        print("Error: No data found or all data processing failed!")
        return None
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.unique(y, return_counts=True)}")
    
    # Create a pipeline with scaling and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])
    
    # Define hyperparameter grid for tuning
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1],
        'svm__kernel': ['linear', 'rbf']
    }
    
    # Perform grid search with cross-validation
    print("Training model with grid search...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print("Best parameters:", grid_search.best_params_)
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Print evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    classes = np.unique(y)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Save the model
    dump(best_model, 'eeg_svm_model.joblib')
    print("Model saved as 'eeg_svm_model.joblib'")
    
    return best_model

if __name__ == "__main__":
    # Set the base directory containing the EEG data folders
    base_dir = "eeg_data"
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found!")
        exit(1)
    
    # Ask if length normalization should be applied
    normalize_input = input("Normalize recording lengths to 4 seconds? (y/n, default=y): ").lower()
    normalize_len = normalize_input != 'n'
    
    # Class mappings (folder name to target class)
    class_mapping = {
        # 'left_hand': 'left_hand',
        # 'left_hand_w_finger_movement': 'left_hand',
        # 'right_hand': 'right_hand',
        # 'right_hand_w_finger_movement': 'right_hand',
        # 'relaxed_state': 'resting',
        # 'meditative_state': 'resting',
        'both_hand': 'flexing',
        'left_hand': 'flexing',
        # 'left_hand_w_finger_movement': 'flexing',
        'right_hand': 'flexing',
        # 'right_hand_w_finger_movement': 'flexing',
        'relaxed_state': 'resting',
        'meditative_state': 'resting'
    }

    # Train the SVM model
    model = train_svm_model(base_dir, class_mapping, normalize_len)
    
    if model:
        print("Model training completed successfully!")
    else:
        print("Model training failed.")