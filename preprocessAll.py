import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes, \
    WaveletDenoisingTypes, ThresholdTypes, WaveletTypes
import glob
import os


def preprocess_eeg_from_csv(input_csv_path, output_csv_path = None, sampling_rate=128,
                            board_id=BoardIds.CYTON_DAISY_BOARD, visualize=True):
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    num_channels = len(eeg_channels)  # 16 channels for Cyton Daisy
    # Load data
    try:
        df = pd.read_csv(input_csv_path,delimiter='\t', header=None)
        print(f"Loaded file: {input_csv_path}")
        print(f"Data shape: {df.shape}")
        print("First few rows:")
        print(df.head())
        if df.shape[1] < num_channels + 1:
            raise ValueError("CSV file does not contain enough columns for EEG data.")
        eeg_data = df.iloc[:, 1:num_channels + 1].values.T  # Exclude the first column (timestamps)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    original_eeg_data = eeg_data.copy()

    print(f"Data shape: {eeg_data.shape}")
    print(f"Using {num_channels} EEG channels")
    print(f"Sampling rate: {sampling_rate} Hz")

    print("Step 1: Removing power line noise...")
    for channel in range(num_channels):
        DataFilter.remove_environmental_noise(
            eeg_data[channel],
            sampling_rate,
            NoiseTypes.FIFTY
        )

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

    print("Step 3: Detrending signals...")
    for channel in range(num_channels):
        DataFilter.detrend(
            eeg_data[channel],
            DetrendOperations.LINEAR
        )

    print("Step 4: Performing wavelet denoising...")
    for channel in range(num_channels):
        DataFilter.perform_wavelet_denoising(
            eeg_data[channel],
            WaveletTypes.DB4,
            4,  # Decomposition level
            WaveletDenoisingTypes.SURESHRINK,
            ThresholdTypes.SOFT
        )

    if visualize:
        plt.figure(figsize=(15, 6 * num_channels))

        for i in range(num_channels):
            plt.subplot(num_channels, 2, 2 * i + 1)
            plt.plot(original_eeg_data[i])
            plt.title(f'Original Channel {i + 1}')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')

            plt.subplot(num_channels, 2, 2 * i + 2)
            plt.plot(eeg_data[i])
            plt.title(f'Preprocessed Channel {i + 1}')
            plt.xlabel('Samples')

        plt.tight_layout()
        plt.show()

    if output_csv_path:
        print(f"Saving preprocessed data to {output_csv_path}")
        df_output = pd.DataFrame(eeg_data.T)
        df_output.columns = [f'Channel_{i + 1}' for i in range(num_channels)]
        df_output.to_csv(output_csv_path, index=False)

    print("Preprocessing complete")
    return original_eeg_data, eeg_data


def preprocess_all_files(left_dir='data_left', right_dir='data_right', output_dir='output', sampling_rate=128,
                         board_id=BoardIds.CYTON_DAISY_BOARD, visualize=True):
    # Find all CSV files in the left and right directories
    left_files = glob.glob('C:/path/*.csv')
    right_files = glob.glob('C:/path/*.csv')

    os.makedirs(output_dir, exist_ok=True)

    # Process left files
    for file in left_files:
        print(f"Processing left-hand file: {file}")
        left_output_dir = "C:/path"
        output_file = os.path.join(left_output_dir, f'preprocessed_{os.path.basename(file)}')
        #output_file = os.path.join(output_dir, f'preprocessed_{os.path.basename(file)}')
        preprocess_eeg_from_csv(
            input_csv_path=file,
            output_csv_path=output_file,
            sampling_rate=sampling_rate,
            board_id=board_id,
            visualize=visualize
            #visualize = 0
        )

    # Process right files
    for file in right_files:
        print(f"Processing right-hand file: {file}")
        right_output_dir = "C:/path"
        output_file = os.path.join(right_output_dir, f'preprocessed_{os.path.basename(file)}')
        #output_file = os.path.join(output_dir, f'preprocessed_{os.path.basename(file)}')
        preprocess_eeg_from_csv(
            input_csv_path=file,
            output_csv_path=output_file,
            sampling_rate=sampling_rate,
            board_id=board_id,
            visualize=visualize
            #visualize=0
        )


if __name__ == "__main__":
    # Process all files in the left and right directories
    preprocess_all_files(
        left_dir="data_left_and_right",
        right_dir="data_right_and_left",
        output_dir="C:/path",
        sampling_rate=128,
        board_id=BoardIds.CYTON_DAISY_BOARD,
        visualize=True
    )