import os
import numpy as np
import joblib

from preprocess import process_single_file

def predict_single_file(file_path, model, normalize_len=True):
    features = process_single_file(file_path, normalize_len)
    
    if features is None:
        return None, None
    
    features_array = np.array(features).reshape(1, -1)
    
    prediction = model.predict(features_array)
    probability = model.predict_proba(features_array)
    
    class_probabilities = {class_name: prob for class_name, prob in zip(model.classes_, probability[0])}
    
    return prediction[0], class_probabilities

if __name__ == "__main__":
    model_path = 'eeg_svm_model.joblib'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        exit(1)
    
    model = joblib.load(model_path)
    
    file_path = input("Enter the path to the EEG file to test (or press Enter to use 'right_hand_example.txt'): ")
    
    if not file_path:
        file_path = "right_hand_example.txt" # just for testing
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        exit(1)
    
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