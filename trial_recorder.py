import pygame
import random
import time
import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import random
import os
import pandas as pd

ID = BoardIds.CYTON_DAISY_BOARD.value
TIME_BEFORE_STREAMING = 3
TIME_BEFORE_ARROW = 4
TIME_BEFORE_STOP = 6

def convert_ts(ts):
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def save_data(board, filename="session.txt", side='left', i=0):
    df = pd.DataFrame(board.get_board_data()).T
    column_names = ['Sample Index', 'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'EXG Channel 8', 'EXG Channel 9', 'EXG Channel 10', 'EXG Channel 11', 'EXG Channel 12', 'EXG Channel 13', 'EXG Channel 14', 'EXG Channel 15', 'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Not Used', 'Digital Channel 0 (D11)', 'Digital Channel 1 (D12)', 'Digital Channel 2 (D13)', 'Digital Channel 3 (D17)', 'Not Used', 'Digital Channel 4 (D18)', 'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2', 'Timestamp', 'Marker Channel']

    df.columns = column_names
    df.set_index('Sample Index', inplace=True)

    df['Timestamp (Formatted)'] = df['Timestamp'].apply(convert_ts)

    metadata = "%OpenBCI Raw EXG Data\n%Number of channels = 16\n%Sample Rate = 125 Hz\n%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n"
    
    with open("Recordings/BrainflowRecorded/recordings_v5/"+filename+side+"{:02}".format(i)+".txt", 'w') as f:
        f.write(metadata)
        df.to_csv(f)

# Helper function to draw cross
def draw_cross():
    screen_width, screen_height = screen.get_size()
    
    # Define cross size
    cross_size = 10  # You can adjust this size based on your preference

    # Calculate center
    center_x = screen_width // 2
    center_y = screen_height // 2
    
    # Calculate cross endpoints
    left = center_x - cross_size
    right = center_x + cross_size
    top = center_y - cross_size
    bottom = center_y + cross_size

    screen.fill((0, 0, 0))
    pygame.draw.line(screen, (255, 255, 255), (left, center_y), (right, center_y), 4)
    pygame.draw.line(screen, (255, 255, 255), (center_x, top), (center_x, bottom), 4)
    pygame.display.flip()

# Helper function to draw arrow
def draw_arrow(side):
    screen_width, screen_height = screen.get_size()
    center_x = screen_width // 2
    center_y = screen_height // 2
    arrow_length = 60  # Length of the arrow (triangle base)
    arrow_height = 40  # Height of the arrow triangle

    if side == 'rest':
        return None
    elif side == "left":
        # Triangle pointing left
        arrow_tip = (center_x - 150, center_y)  # tip of arrow further left from center
        base_top = (arrow_tip[0] + arrow_length, center_y - arrow_height // 2)
        base_bottom = (arrow_tip[0] + arrow_length, center_y + arrow_height // 2)
        pygame.draw.polygon(screen, (255, 255, 255), [arrow_tip, base_top, base_bottom])

    elif side == "right":
        # Triangle pointing right
        arrow_tip = (center_x + 150, center_y)  # tip of arrow further right from center
        base_top = (arrow_tip[0] - arrow_length, center_y - arrow_height // 2)
        base_bottom = (arrow_tip[0] - arrow_length, center_y + arrow_height // 2)
        pygame.draw.polygon(screen, (255, 255, 255), [arrow_tip, base_top, base_bottom])

    elif side == "both":
        arrow_tip = (center_x - 150, center_y)  # tip of arrow further left from center
        base_top = (arrow_tip[0] + arrow_length, center_y - arrow_height // 2)
        base_bottom = (arrow_tip[0] + arrow_length, center_y + arrow_height // 2)
        pygame.draw.polygon(screen, (255, 255, 255), [arrow_tip, base_top, base_bottom])

        arrow_tip = (center_x + 150, center_y)  # tip of arrow further right from center
        base_top = (arrow_tip[0] - arrow_length, center_y - arrow_height // 2)
        base_bottom = (arrow_tip[0] - arrow_length, center_y + arrow_height // 2)
        pygame.draw.polygon(screen, (255, 255, 255), [arrow_tip, base_top, base_bottom])

    pygame.display.flip()

if __name__=='__main__':
    filename = input("Input the name of the file this recording will be stored in: ");

    # Initialize BrainFlow
    params = BrainFlowInputParams()
    # Adjust parameters if you have custom setup
    params.serial_port = "/dev/cu.usbserial-DP04W4EN"
    
    board = BoardShim(ID, params)
    
    # Initialize Pygame

    os.environ['SDL_VIDEO_WINDOW_POS'] = 'center'
    os.environ['SDL_VIDEO_ALLOW_SCREENSAVER'] = '0'

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("EEG Visual Task")
    font = pygame.font.SysFont(None, 48)

    trials = []
    for i in range(5):
        trials.append('left')
        trials.append('right')
        trials.append('both')
        trials.append('rest')

    #random.shuffle(trials)
    random.shuffle(trials)
    draw_cross()

    # Draw cross

    # Start streaming
    print("Preparing BrainFlow streaming...")
    board.prepare_session()
    time.sleep(TIME_BEFORE_STREAMING)

    for i, trial in enumerate(trials):
        board.start_stream()
        # Wait a short period
        time.sleep(TIME_BEFORE_ARROW)

        # Show arrow either left or right
        draw_arrow(trial)

        # Keep arrow on screen for some duration (e.g., 5 seconds) while streaming data
        time.sleep(TIME_BEFORE_STOP)

        # Stop streaming
        print("Stopping BrainFlow streaming...")
        board.stop_stream()
        draw_cross()

        '''
        # Show "done" message
        screen.fill((0, 0, 0))
        done_text = font.render("Done!", True, (255, 255, 255))
        screen.blit(done_text, (350, 280))
        pygame.display.flip()
        '''

        # Save data to file
        save_data(board, filename, trial, i)

        time.sleep((TIME_BEFORE_STREAMING))


    board.release_session()

    # Keep window open for a few seconds before quitting
    pygame.quit()
    print("Program complete.")
