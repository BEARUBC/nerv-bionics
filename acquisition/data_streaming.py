import os
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import logging
import PyQt6
import pyqtgraph as pg

import numpy as np
import time
import scipy.io
import argparse
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore
import pandas as pd

import visualization

frequency = 250

eeg_csv_file = []

parsed_event = []
parsed_pos = []
parsed_event_pos = []

def import_to_mne(file):

    csv = pd.read_csv(file, header=None)
    stream_duration = (len(csv) / frequency) + 1
    print("Stream Duration: ", stream_duration)
    print("Stream Length: ", len(csv))
    

    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.file = file
    params.master_board = BoardIds.SYNTHETIC_BOARD
    board = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)
    print("# Cols: ", BoardShim.get_num_rows(BoardIds.SYNTHETIC_BOARD.value))
    print("EEG Channels: ", BoardShim.get_exg_channels(BoardIds.SYNTHETIC_BOARD.value))
    print("EEG Names: ", BoardShim.get_eeg_names(BoardIds.SYNTHETIC_BOARD.value))
    print("Timestamp Channel: ", BoardShim.get_timestamp_channel(BoardIds.SYNTHETIC_BOARD.value))
    #print("Marker Channel: ", BoardShim.get_marker_channel(BoardIds.SYNTHETIC_BOARD.value))

    parser = argparse.ArgumentParser()
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    args = parser.parse_args()
    
    try:
        board.prepare_session()
        board.start_stream(450000, args.streamer_params)
        visualization.Graph(board)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board.is_prepared():
            logging.info('Releasing session')
            board.release_session()

    # board.prepare_session()
    
    # board.start_stream()
    
    # BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    #time.sleep(stream_duration)
    # data = board.get_board_data()
    # #get_emg_channels (board_id)
    # board.stop_stream()
    # board.release_session()

    
    # eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    # eeg_data = data[eeg_channels, :]

    # return data
    
def create_csv(file):

    mat = scipy.io.loadmat(f'data/{file}.mat')
    print(mat.keys())

    # only getting EEG data, getting rid of EOG channels
    mat['s'] = np.delete(mat['s'], [22, 23, 24], 1)

    num_cols = mat['s'].shape[1]

    # adding cols cause dataformat is weird
    if num_cols < 32:
        extra_cols = np.zeros((mat['s'].shape[0], 32 - num_cols))
        mat['s'] = np.hstack((mat['s'], extra_cols))
    
    # add time column at index 30
    time_column = np.arange(mat['s'].shape[0]) / frequency
    mat['s'][:, 30] = time_column

    print("Time:", mat['s'][:, 30])
    data = mat['s']
    print("CSV # Rows: ", len(data))
    np.savetxt(f'data/{file}.csv', data, delimiter=",")

def get_events(file):
    mat = scipy.io.loadmat(f'./data/{file}.mat')
    parsed_event_pos = parse_event(mat['EVENTTYP'], mat['EVENTPOS'])


def parse_event(event_type, event_pos):
    
    event_index = {'Idling EEG (eyes open)': 0,
                   'Idling EEG (eyes closed)': 0,
                   'Cue onset left (class 1)': 0,
                   'Cue onset right (class 2)': 0,
                   'Cue onset foot (class 3)': 0,
                   'Cue onset tongue (class 4)': 0,
                   'Cue unknown': 0,
                   'Rejected trial': 0,
                   'Eye movements': 0,
                   'Parse Error': 0}

    for i in range(len(event_type)):
        match event_type[i]:
            case 276:
                event_index['Idling EEG (eyes open)'] += 1
                parsed_event.append(f"Idling EEG (eyes open) #{event_index['Idling EEG (eyes open)']}")
            case 277:
                event_index['Idling EEG (eyes closed)'] += 1
                parsed_event.append(f"Idling EEG (eyes closed) #{event_index['Idling EEG (eyes closed)']}")
            case 768:
                parsed_event.append(f"Start of a trial")
            case 769:
                event_index['Cue onset left (class 1)'] += 1
                parsed_event.append(f"Cue onset left (class 1) #{event_index['Cue onset left (class 1)']}")
            case 770:
                event_index['Cue onset right (class 2)'] += 1
                parsed_event.append(f"Cue onset right (class 2) #{event_index['Cue onset right (class 2)']}")
            case 771:
                event_index['Cue onset foot (class 3)'] += 1
                parsed_event.append(f"Cue onset foot (class 3) #{event_index['Cue onset foot (class 3)']}")
            case 772:
                event_index['Cue onset tongue (class 4)'] += 1
                parsed_event.append(f"Cue onset tongue (class 4) #{event_index['Cue onset tongue (class 4)']}")
            case 783:
                event_index['Cue unknown'] += 1
                parsed_event.append(f"Cue unknown #{event_index['Cue unknown']}")
            case 1023:
                event_index['Rejected trial'] += 1
                parsed_event.append(f"Rejected trial #{event_index['Rejected trial']}")
            case 1072:
                event_index['Eye movements'] += 1
                parsed_event.append(f"Eye movements #{event_index['Eye movements']}")
            case 32766:
                parsed_event.append(f"Start of a new run")
            case _:
                event_index['Parse Error'] += 1
                parsed_event.append(f"Parse Error #{event_index['Parse Error']}")

    for i in range(len(event_pos)):
        parsed_pos.append(int(event_pos[i][0]))

    for i in range(len(parsed_pos)):
        parsed_event_pos.append([parsed_event[i], parsed_pos[i]])

    return parsed_event_pos

def trim_desired_data(desired_trials, csv):
    desired_trial_pos = []
    desired_trial_start_pos = []
    desired_trial_end_pos = []
    trimmed_data = []

    for i in range(len(desired_trials)):
        desired_trial_pos.append(parsed_event_pos[desired_trials[i]][1])
        desired_trial_start_pos.append(parsed_event_pos[desired_trials[i] - 1][1])
        desired_trial_end_pos.append(parsed_event_pos[desired_trials[i] + 1][1])

    print("Start: ", desired_trial_start_pos)
    print("Pos: ", desired_trial_pos)
    print("End: ", desired_trial_end_pos)

    for i in range(len(desired_trials)):
        for j in range(desired_trial_end_pos[i] - desired_trial_start_pos[i]):
            trimmed_data.append(csv.iloc[j + desired_trial_start_pos[i]])
    
    np.savetxt(f'data/stream.csv', trimmed_data, delimiter=",")
    


# wishlist:
# clean up code
# UI
# make it so mlx script is run in python

def show_data(data):
    df = pd.DataFrame()
    for i in range(22):
        df["node ", (i + 1)] = data[i]
    print(df)

def table_of_contents():

    df = pd.DataFrame()
    df['EVENT'] = parsed_event
    df['Position'] = parsed_pos
    print(df.to_string())
        
def launch_board_ds2a(file, stream_duration):

    desired_trials = []

    if not(os.path.isfile(f'./data/{file}.csv')):
        print("Creating CSV file")
        create_csv(file)

    eeg_csv_file = pd.read_csv(f'./data/{file}.csv', header=None)
    print("eeg_csv_file: ", eeg_csv_file)
    get_events(file)
    print("Launching playback board with file ", file)
    table_of_contents()
    
    user_input = input("Enter Desired Trial Numbers (2 3 12 ... 4): ").split(' ')
    
    for i in range(len(user_input)):
        desired_trials.append(int(user_input[i]))

    trim_desired_data(desired_trials, eeg_csv_file)


    import_to_mne(f'./data/stream.csv')
    #data = import_to_mne(f'./data/stream.csv')
    #show_data(data)

def main():
    launch_board_ds2a('A04T', 0)

if __name__ == '__main__':
    main()