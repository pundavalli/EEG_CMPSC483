## Getting Started

This is a capstone project completed in `CMPSC 483W` at The Pennsylvania State University.

The EEG data is compressed in the `eeg_data.zip` file. Simply unzip the file and everything shouild already be in the correct directories.

## Demo

1. Prepare the EEG and wear it according to the following pin-out:

> Note: We did not use electrode wire `Pin 16` (Daisy channel `N8P`)

![Pinout](https://github.com/user-attachments/assets/f8b96069-e796-4626-b563-2b705b84cf8d)

2. Once the EEG is properly worn, complete an impedance test on each connected channel to verify that each electrode is under **15~20 kΩ**. If any electron is over 20 kΩ, disconnect, reapply saline solution, and reconnect. If electrode impedance is very low, e.g., 0~3 kΩ, restart the hardware; this may be a calibration problem.

3. Run `main.py` using the following command
   
		python3 -m uvicorn main:app --reload

4. The correct model `eeg_svm_model_csp_only.joblib` (Model V2) should automatically load.

5. Navigate to `http://localhost:8000`, and you should be presented with a game to test the EEG signal.

6. To directly access the EEG classifier data, the websocket endpoint can be found at `http://localhost:3000/ws` 

## Data Preparation using OpenBCI Cyton

- **make sure all records are 4 seconds**
	- first 4s are rest
	- the remaining  6s are flexing (-1s at the start & -1s at the end to account for human error)

- aim for three classes (LEFT_HAND, RIGHT_HAND, RELAXED)
	- meditative state is closed eyes with no noise
	- relaxed state is eyes open just looking around

- DO NOT use channel 16 (it is not in use, garbage noise)

# Model Info

A slice of the current SVM model boundary:

![Boundary](https://github.com/pundavalli/EEG_CMPSC483/blob/574fc0f7655e44f814706de01c162d564a495509/svm_decision_boundary.png)
