import socket
import json
import pandas as pd
from realtimetrial import preprocess
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
from brainflow.board_shim import BoardIds

ID = BoardIds.CYTON_DAISY_BOARD
model = joblib.load("eeg_svm_model.joblib")

TCP_IP = "127.0.0.1"
TCP_PORT = 8002 # Use the same port as in GUI

window_size = 500  # total number of samples to keep


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_IP, TCP_PORT))
sock.listen(1)

print(f"Listening for TCP packets on {TCP_IP}:{TCP_PORT}...")

conn, addr = sock.accept()

print("Connected by", addr)

def predict(features):
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

with conn:
    while True:
        data = conn.recv(4096)  # Receive up to 4096 bytes
        try:
            packet = json.loads(data.decode('utf-8'))
            print(packet)
            #data = np.array(packet['data'])
        except json.JSONDecodeError:
            continue
