import socket
import json
import time

import pandas as pd
import numpy as np
import joblib
import threading
import pygame
from brainflow.board_shim import BoardIds
from preprocess import extract_activity_segment, normalize_length, extract_features
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio
import random  # simulate prediction

app = FastAPI()
clients = set()

# Model & Board Setup
ID = BoardIds.CYTON_DAISY_BOARD
model = joblib.load("eeg_svm_model_csp_only.joblib")

UDP_IP = "127.0.0.1"
UDP_PORT = 8002
window_size = 10 * 125  # total number of samples to keep

buffer = np.empty((16, 0))
fill_level = 0.0  # from 0.0 (empty) to 1.0 (full)
prediction_state = { "state": "resting", "probability": 0.0 }

# Pygame setup
WIDTH, HEIGHT = 300, 400
BAR_WIDTH = 100
BAR_HEIGHT = 300
FPS = 30

def update_buffer(buffer, new_data):
    buffer = np.hstack((buffer, new_data))
    if buffer.shape[1] > window_size:
        buffer = buffer[:, -window_size:]
    return buffer

def predict(features):
    if features is None:
        return None, None
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    probability = model.predict_proba(features_array)
    class_probabilities = {class_name: prob for class_name, prob in zip(model.classes_, probability[0])}
    return prediction[0], class_probabilities

def run_udp_listener():
    global buffer, fill_level, prediction_state

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}...")

    time.sleep(3)

    while True:
        data, _ = sock.recvfrom(4096)
        try:
            packet = json.loads(data.decode('utf-8'))
            data = np.array(packet['data'])
            #print(data)
            buffer = update_buffer(buffer, data)

            if buffer.shape[1] >= window_size:
                buffer = np.ascontiguousarray(buffer)
                eeg_data = buffer[0:15]
                #print(eeg_data[1][-10:])
                eeg_data = pd.DataFrame(eeg_data.T, columns=[f' EXG Channel {i}' for i in range(15)])

                activity_segment = extract_activity_segment(eeg_data)
                activity_segment = normalize_length(activity_segment)
                features = extract_features(activity_segment)
                #print(features)

                prediction, probabilities = predict(features)

                #print(f"Prediction: {prediction}")
                #print(f"Probabilities: {probabilities}")
                if prediction:
                    prediction_state = {
                        "state": prediction,
                        "probability": probabilities,
                    }
        except json.JSONDecodeError:
            continue

def run_pygame_meter():
    global fill_level, prediction_state

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("EEG Flex Meter")
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update fill_level based on prediction state

        if prediction_state['state'] == "flexing":
            fill_level = min(1.0, fill_level + 0.01)
        else:
            fill_level = max(0.0, fill_level - 0.01)

        '''
        if prediction_state['state'] == "right_hand":
            fill_level = min(1.0, fill_level + 0.01)
        elif prediction_state['state'] == "left_hand":
            fill_level = max(0.0, fill_level - 0.01)
        '''
        # Clear screen
        screen.fill((30, 30, 30))

        # Draw meter background
        pygame.draw.rect(screen, (70, 70, 70), ((WIDTH - BAR_WIDTH) // 2, HEIGHT - BAR_HEIGHT - 20, BAR_WIDTH, BAR_HEIGHT))

        # Draw fill
        fill_px = int(BAR_HEIGHT * fill_level)
        pygame.draw.rect(screen, (0, 200, 0), ((WIDTH - BAR_WIDTH) // 2, HEIGHT - fill_px - 20, BAR_WIDTH, fill_px))

        # Draw text
        font = pygame.font.SysFont(None, 36)
        label = font.render(prediction_state['state'].upper(), True, (255, 255, 255))
        screen.blit(label, ((WIDTH - label.get_width()) // 2, 20))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

# Start both threads
threading.Thread(target=run_udp_listener, daemon=True).start()
#run_pygame_meter()

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("message.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(0.1)
    except:
        clients.remove(websocket)

# Example function to push predictions
async def send_predictions():
    global prediction_state

    while True:
        #prediction = random.choice(["flexing", "relaxed"])  # replace with your real prediction
        for client in clients:
            await client.send_text(json.dumps(prediction_state))
        await asyncio.sleep(0.5)  # adjust to your real-time speed

# Start the background task
import uvicorn
threading.Thread(target=lambda: asyncio.run(send_predictions()), daemon=True).start()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)