## Getting Started

The EEG data is compressed in the `eeg_data.zip` file. Simply unzip the file and everything shouild already be in the correct directories.

## Data Preparation

- **make sure all records are 4 seconds**
	- first 4s are rest
	- the remaining  6s are flexing (-1s at the start & -1s at the end to account for human error)

- aim for three classes (LEFT_HAND, RIGHT_HAND, RELAXED)
	- meditative state is closed eyes with no noise
	- relaxed state is eyes open just looking around

- DO NOT use channel 16 (it is not in use, garbage noise)
