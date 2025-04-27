from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
	params = BrainFlowInputParams()
	params.serial_port = "/dev/cu.usbserial-DP04W4EN"
	
	board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
	board.prepare_session()
	board.start_stream()
	print("Starting Streaming")

	input("Enter anything to stop streaming: ")

	board.stop_stream()
	data = board.get_board_data()
	board.release_session()
	print("Streaming stopped")

	eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD)
	
	eeg_data = data[eeg_channels, :]
	
	plt.plot(eeg_data[0, :1000])
	plt.title("Channel 3")
	plt.xlabel("Samples")
	plt.ylabel("Voltage (microV)")
	'''
	fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12,8))
	for i in range(4):
		for j in range(4):
			currax = axes[i][j]
			currax.plot(eeg_data[4*i+j, :1000])
			currax.set_title("Channel {}".format(4*i+j))
			currax.set_xlabel("Samples")
			currax.set_ylabel("Voltage (microV)")	
    '''
	plt.show()
