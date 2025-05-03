import os
import numpy as np
import joblib

# Import the preprocessing module
from preprocess import process_single_file

def predict_single_file(file_path, model, normalize_len=True):
    # Process the file to extract features
    features = process_single_file(file_path, normalize_len)
    
    if features is None:
        return None, None
    
    # Reshape for single sample prediction
    features_array = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features_array)
    probability = model.predict_proba(features_array)
    
    # Get the class probabilities
    class_probabilities = {class_name: prob for class_name, prob in zip(model.classes_, probability[0])}
    
    return prediction[0], class_probabilities

if __name__ == "__main__":
    # Import the model
    model_path = 'eeg_svm_model.joblib'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        exit(1)
    
    model = joblib.load(model_path)
    
    # Get the file path from user or use a default
    file_path = input("Enter the path to the EEG file to test (or press Enter to use 'right_hand_example.txt'): ")
    
    if not file_path:
        file_path = "right_hand_example.txt"  # Default file
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        exit(1)
    
    # Ask if length normalization should be applied
    normalize_input = input("Normalize recording length to 4 seconds? (y/n, default=y): ").lower()
    normalize_len = normalize_input != 'n'
    
    # Make prediction
    prediction, probabilities = predict_single_file(file_path, model, normalize_len)
    
    if prediction:
        print(f"\nPredicted class: {prediction}")
        print("\nClass probabilities:")
        for class_name, probability in probabilities.items():
            print(f"  {class_name}: {probability:.4f} ({probability*100:.2f}%)")
    else:
        print("Failed to make a prediction.")
