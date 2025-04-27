import os
import numpy as np
import pandas as pd
import glob
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# Function to load and preprocess a single EEG file
def load_eeg_file(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    
    # If the file starts with "%OpenBCI Raw EXG Data", it has the header
    if first_line.startswith("%OpenBCI"):
        # Read the file, skipping the first 4 lines (header)
        data = pd.read_csv(file_path, skiprows=4)
    else:
        data = pd.read_csv(file_path)
    
    # Select only EXG Channels 0-14 (dropping channel 15 as it's noise)
    column_names = data.columns
    eeg_columns = [col for col in column_names if "EXG Channel" in col and not "EXG Channel 15" in col]
    
    # If column names don't have "EXG Channel", assume columns 1-15 are the EEG channels
    if not eeg_columns:
        eeg_data = data.iloc[:, 1:16]  # Columns 1-15 correspond to EXG Channel 0-14
    else:
        eeg_data = data[eeg_columns]
    
    return eeg_data

def extract_activity_segment(eeg_data, sample_rate=125):
    start_idx = 5 * sample_rate
    end_idx = len(eeg_data) - 1 * sample_rate
    
    # If recording is shorter than expected, adjust accordingly
    if end_idx <= start_idx:
        # If recording is too short, use the middle section
        start_idx = len(eeg_data) // 4
        end_idx = len(eeg_data) * 3 // 4
    
    # Extract the activity segment
    activity_segment = eeg_data.iloc[start_idx:end_idx, :]
    
    return activity_segment

def normalize_length(eeg_segment, target_length=4*125):
    current_length = len(eeg_segment)
    
    if current_length == target_length:
        return eeg_segment
    elif current_length > target_length:
        # If longer, crop to target length
        middle_idx = current_length // 2
        half_target = target_length // 2
        start_idx = middle_idx - half_target
        end_idx = middle_idx + half_target
        return eeg_segment.iloc[start_idx:end_idx, :]
    else:
        # If shorter, pad with zeros to target length
        padding_length = target_length - current_length
        padding_before = padding_length // 2
        padding_after = padding_length - padding_before
        
        pad_before_df = pd.DataFrame(np.zeros((padding_before, eeg_segment.shape[1])), 
                                     columns=eeg_segment.columns)
        pad_after_df = pd.DataFrame(np.zeros((padding_after, eeg_segment.shape[1])), 
                                    columns=eeg_segment.columns)
        
        normalized = pd.concat([pad_before_df, eeg_segment, pad_after_df], ignore_index=True)
        return normalized

def extract_frequency_features(eeg_segment, sample_rate=125):
    features = []
    
    for channel in range(eeg_segment.shape[1]):
        channel_data = eeg_segment.iloc[:, channel].values
        
        # Compute PSD using Welch's method
        freqs, psd = welch(channel_data, fs=sample_rate, nperseg=sample_rate)
        
        # Extract power in different frequency bands
        # Delta (0.5-4 Hz)
        delta_mask = (freqs >= 0.5) & (freqs <= 4)
        delta_power = np.mean(psd[delta_mask]) if np.any(delta_mask) else 0
        
        # Theta (4-8 Hz)
        theta_mask = (freqs >= 4) & (freqs <= 8)
        theta_power = np.mean(psd[theta_mask]) if np.any(theta_mask) else 0
        
        # Alpha (8-13 Hz)
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        alpha_power = np.mean(psd[alpha_mask]) if np.any(alpha_mask) else 0
        
        # Beta (13-30 Hz)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        beta_power = np.mean(psd[beta_mask]) if np.any(beta_mask) else 0
        
        # Add features for this channel
        features.extend([delta_power, theta_power, alpha_power, beta_power])
    
    return features

def extract_time_features(eeg_segment):
    features = []
    
    for channel in range(eeg_segment.shape[1]):
        channel_data = eeg_segment.iloc[:, channel].values
        
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        skewness = skew(channel_data)
        kurt = kurtosis(channel_data)
        
        features.extend([mean, std, skewness, kurt])
    
    return features

def extract_features(eeg_segment):
    time_features = extract_time_features(eeg_segment)
    freq_features = extract_frequency_features(eeg_segment)
    
    # combine all features
    all_features = time_features + freq_features
    
    return all_features

def process_single_file(file_path, normalize_len=True):
    try:
        # Load the EEG data
        eeg_data = load_eeg_file(file_path)
        # Extract the activity segment
        activity_segment = extract_activity_segment(eeg_data)
        
        # Normalize length if needed
        if normalize_len:
            activity_segment = normalize_length(activity_segment)
        
        # Extract features
        features = extract_features(activity_segment)
        
        return features
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def load_dataset(base_dir, class_mapping, normalize_len=True):    
    features_list = []
    labels_list = []
    
    for folder_name, target_class in class_mapping.items():
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found. Skipping.")
            continue
        
        file_pattern = os.path.join(folder_path, "*.txt")
        files = glob.glob(file_pattern)
        
        print(f"Processing {len(files)} files from {folder_name}...")
        
        for file_path in files:
            features = process_single_file(file_path, normalize_len)
            
            if features:
                features_list.append(features)
                labels_list.append(target_class)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    return X, y