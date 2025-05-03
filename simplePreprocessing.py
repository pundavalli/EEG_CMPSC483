import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations, NoiseTypes


def preprocess_eeg_from_csv(input_csv_path, output_csv_path=None, sampling_rate=128, board_id=BoardIds.CYTON_DAISY_BOARD, visualize=True):
    # Getting channel information
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    num_channels = len(eeg_channels) + 1 #top row is time

    # Load data
    try:
        # Try loading with BrainFlow but doesnt work but I think has to do with file formatting
        data = DataFilter.read_file(input_csv_path)
        if len(data) >= num_channels:
            eeg_data = data[0:num_channels]
        else:
            print("Error: Not enough channels found in the data.")
            return None, None
    except Exception as e:
        #print(f"BrainFlow loading failed: {e}. Trying pandas instead.")
        try:
            df = pd.read_csv(input_csv_path, skiprows=4)
            # 8 columns
            eeg_data = df.iloc[:, 0:num_channels].values.T
        except Exception as e2:
            print(f"Error loading data: {e2}")
            return None, None

    # Store original data for comparison
    original_eeg_data = eeg_data.copy()

    # Print data information
    print(f"Data shape: {eeg_data.shape}")
    print(f"Using {num_channels} EEG channels")
    print(f"Sampling rate: {sampling_rate} Hz")

    # 1. Remove power line noise
    print("Step 1: Removing power line noise...")
    for channel in range(num_channels):
        # Apply 50Hz notch filter (use NoiseTypes.SIXTY for 60Hz)
        DataFilter.remove_environmental_noise(
            eeg_data[channel],
            sampling_rate,
            NoiseTypes.FIFTY
        )

    # 2. Apply bandpass filter
    print("Step 2: Applying bandpass filter (1-50Hz)...")
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
    print("Step 3: Detrending signals...")
    for channel in range(num_channels):
        DataFilter.detrend(
            eeg_data[channel],
            DetrendOperations.LINEAR
        )

    #4. Wavelet denoising
    print("Step 4: Performing wavelet denoising...")
    for channel in range(num_channels):
        DataFilter.perform_ica(
            eeg_data,#?[channel],
            8
        )

    # Visualize data before and after preprocessing
    if visualize:
        plt.figure(figsize=(30, 24))

        # Plot original data
        for i in range(num_channels):
            plt.subplot(num_channels, 2, 2 * i + 1)
            plt.plot(original_eeg_data[i])
            plt.title(f'Original Channel {i + 1}')
            if i == num_channels - 1:
                plt.xlabel('Samples')
            plt.ylabel('Amplitude')

        # Plot preprocessed data
        for i in range(num_channels):
            plt.subplot(num_channels, 2, 2 * i + 2)
            plt.plot(eeg_data[i])
            plt.title(f'Preprocessed Channel {i + 1}')
            if i == num_channels - 1:
                plt.xlabel('Samples')

        plt.tight_layout()
        plt.show()

    # Saving
    print(f"Saving preprocessed data")
    df_output = pd.DataFrame(eeg_data.T)
    df_output.columns = [f'Channel_{i + 1}' for i in range(num_channels)]
    df_output.to_csv(output_csv_path, index=False)

    print("Preprocessing complete")
    return original_eeg_data, eeg_data


if __name__ == "__main__":
    #change path based on your files
    path = "data_left/left11.csv"

    directory_prefix = "/Users/praneelundavalli/Documents/OpenBCI_GUI/Recordings/recordings v2/"
    directory= directory_prefix+"left_hand"
    left_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    directory = directory_prefix + "left_hand_w_finger_movement"
    left_files.extend([os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

    directory = directory_prefix + "right_hand"
    right_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    directory = directory_prefix + "right_hand_w_finger_movement"
    right_files.extend([os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


    #path = "data_right/right3.csv"

    #OUTPUT_FILE = "p_left/preprocessed_left11.csv"
    #OUTPUT_FILE = "p_right/preprocessed_right3.csv"

    '''original, preprocessed = preprocess_eeg_from_csv(
        input_csv_path=path,
        output_csv_path=OUTPUT_FILE,
        visualize=True
    )'''

    os.makedirs('p_left2', exist_ok=True)
    for i in range(len(left_files)):
        output_path = "p_left2/preprocessed_left" + str(i) + ".csv"
        original, preprocessed = preprocess_eeg_from_csv(input_csv_path=left_files[i], output_csv_path=output_path, visualize=False)

    os.makedirs('p_right2', exist_ok=True)
    for i in range(len(right_files)):
        output_path = "p_right2/preprocessed_right" + str(i) + ".csv"
        original, preprocessed = preprocess_eeg_from_csv(input_csv_path=right_files[i], output_csv_path=output_path,
                                                         visualize=False)

