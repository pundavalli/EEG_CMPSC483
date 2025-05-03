import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams, LogLevels
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations, NoiseTypes, \
    WaveletDenoisingTypes, ThresholdTypes, WaveletTypes
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import extract_features, extract_activity_segment, normalize_length

BoardShim.enable_dev_board_logger()


ID = BoardIds.CYTON_DAISY_BOARD
model = joblib.load("eeg_svm_model.joblib")
proj_matrix = joblib.load("csp_filters.pkl")

def preprocess(eeg_data):
    #print(type(eeg_data))
    #print(eeg_data[1])
    #print(eeg_data.dtype)
    #np.savetxt("before_from_file.txt", eeg_data, fmt="%.5f")
    eeg_data = np.ascontiguousarray(eeg_data)

    sampling_rate = BoardShim.get_sampling_rate(ID)
    eeg_channels = BoardShim.get_eeg_channels(ID)
    num_channels = len(eeg_channels)  # top row is time

    # 1. Remove power line noise
    # print("Step 1: Removing power line noise...")
    for channel in range(num_channels):
        # Apply 50Hz notch filter (use NoiseTypes.SIXTY for 60Hz)
        DataFilter.remove_environmental_noise(
            eeg_data[channel],
            sampling_rate,
            NoiseTypes.FIFTY
        )

    # 2. Apply bandpass filter
    # print("Step 2: Applying bandpass filter (1-50Hz)...")
    for channel in range(num_channels):
        DataFilter.perform_bandpass(
            eeg_data[channel],
            sampling_rate,
            1.0,
            50.0,
            4,
            FilterTypes.BUTTERWORTH,
            0
        )

    # 3. Detrend the signal
    # print("Step 3: Detrending signals...")
    for channel in range(num_channels):
        DataFilter.detrend(
            eeg_data[channel],
            DetrendOperations.LINEAR
        )

    #np.savetxt("after_from_file.txt", eeg_data, fmt="%.5f")
    #4. Wavelet denoising
    # print("Step 4: Performing wavelet denoising...")
    for channel in range(num_channels):
        DataFilter.perform_ica(
            eeg_data,#?[channel],
            8
        )

    #return eeg_data

    # Trying another step
    for channel in range(num_channels):
        DataFilter.perform_wavelet_denoising(
            eeg_data[channel],
            WaveletTypes.DB4,  # Wavelet name (e.g., 'db4' for Daubechies 4)
            3,  # Decomposition level (typical: 3–5)
            WaveletDenoisingTypes.SURESHRINK,  # Denoising method
            ThresholdTypes.SOFT  # Threshold type: HARD or SOFT
        )

    return eeg_data

def predict(features):
    if features is None:
        return None, None
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    probability = model.predict_proba(features_array)
    class_probabilities = {class_name: prob for class_name, prob in zip(model.classes_, probability[0])}
    return prediction[0], class_probabilities



if __name__ == "__main__":

    params = BrainFlowInputParams()
    # Adjust parameters if you have custom setup
    params.serial_port = "/dev/cu.usbserial-DP04W4EN"

    board = BoardShim(ID, params)

    board.prepare_session()
    board.start_stream()
    print("Started stream")

    #Temp code to instead get data from csv file
    # filename = "/Users/praneelundavalli/Documents/OpenBCI_GUI/Recordings/recordingsv2/left_hand/OpenBCI-RAW-2025-04-17_18-37-56.txt"
    # df = pd.read_csv(filename, skiprows=4)
    # df.drop('Sample Index', axis=1, inplace=True)

    time.sleep(10)

    eeg_channels = BoardShim.get_eeg_channels(ID)
    try:
        while True:
            eeg_data = board.get_current_board_data(1250)
            #eeg_data = board.get_board_data(1250)
            '''
            print(eeg_data.shape)
            if eeg_data.shape[1] == 0:
                print("No data received. Check your connection and serial port.")
            else:
                print("Data received.")
                eeg_channels = BoardShim.get_eeg_channels(ID)
                timestamp_channel = eeg_data[-1]
                print("Timestamps (last 10):", timestamp_channel[-10:])
                print("First EEG channel sample (last 10):", eeg_data[eeg_channels[0]][-10:])

            timestamps = eeg_data[-1, :]
            print("Newest timestamp:", timestamps[-1])
            '''
            #print(eeg_data[1][-10:])
            eeg_data = eeg_data[1:16]
            eeg_data = pd.DataFrame(eeg_data.T, columns=[f' EXG Channel {i}' for i in range(15)])

            activity_segment = extract_activity_segment(eeg_data)
            activity_segment = normalize_length(activity_segment)
            features = extract_features(activity_segment)
            print("Features:", features)
            print("\n", np.std(features))
            prediction, probabilities = predict(features)

            print(f"Prediction: {prediction}\n")
            print(f"Probabilities: {probabilities}")
            if prediction:
                prediction_state = prediction
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        print("Cleaning up...")
        board.stop_stream()
        board.release_session()
    #board.stop_stream()
    #board.release_session()