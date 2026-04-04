import os
import mne
import matplotlib
matplotlib.use('Agg')  # no popup plots
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

import visualization  # our live graph module

frequency = 250  # how many samples per second the EEG records

eeg_csv_file = []

parsed_event = []      # event names like "Cue onset left #1"
parsed_pos = []        # where each event happens in the data
parsed_event_pos = []  # pairs of [event name, position]

# plays back a CSV file as if it were a live EEG stream and shows the graph
def import_to_mne(file):
    """Replay a BrainFlow-formatted CSV file through the live graph.

    Parameters
    ----------
    file : str
        Path to the BrainFlow-compatible CSV file to replay.

    Returns
    -------
    None
        This function only runs the playback session and prints diagnostics.

    Side Effects
    ------------
    - Starts a BrainFlow playback board session.
    - Opens the live visualization window.
    - Prints stream metadata to the console.
    - Releases the BrainFlow session when playback finishes or fails.
    """

    # figure out how long the recording is
    csv = pd.read_csv(file, header=None)
    stream_duration = (len(csv) / frequency) + 1
    print("Stream Duration: ", stream_duration)
    print("Stream Length: ", len(csv))

    BoardShim.enable_dev_board_logger()

    # fake EEG board that reads from our file instead of real hardware
    params = BrainFlowInputParams()
    params.file = file
    params.master_board = BoardIds.SYNTHETIC_BOARD
    board = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)

    # just printing info for debugging
    print("# Cols: ", BoardShim.get_num_rows(BoardIds.SYNTHETIC_BOARD.value))
    print("EEG Channels: ", BoardShim.get_exg_channels(BoardIds.SYNTHETIC_BOARD.value))
    print("EEG Names: ", BoardShim.get_eeg_names(BoardIds.SYNTHETIC_BOARD.value))
    print("Timestamp Channel: ", BoardShim.get_timestamp_channel(BoardIds.SYNTHETIC_BOARD.value))
    #print("Marker Channel: ", BoardShim.get_marker_channel(BoardIds.SYNTHETIC_BOARD.value))

    parser = argparse.ArgumentParser()
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    args = parser.parse_args()

    # start streaming and show live graph
    try:
        board.prepare_session()
        board.start_stream(450000, args.streamer_params)  # big buffer
        visualization.Graph(board)  # opens the live graph window
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        # always clean up even if it crashes
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

# turns a .mat file into a CSV that brainflow understands
def create_csv(file):
    """Convert a MATLAB EEG recording into a BrainFlow-compatible CSV.

    Parameters
    ----------
    file : str
        Base dataset name without extension, for example 'A01T'.

    Returns
    -------
    None
        The converted CSV is written to disk.

    Side Effects
    ------------
    - Loads data/<file>.mat.
    - Removes EOG channels.
    - Pads the matrix to the 32-column BrainFlow layout.
    - Writes data/<file>.csv.
    - Prints dataset keys and the resulting row count.
    """

    mat = scipy.io.loadmat(f'data/{file}.mat')
    print(mat.keys())

    # ditch the eye movement channels (not brain data)
    mat['s'] = np.delete(mat['s'], [22, 23, 24], 1)

    num_cols = mat['s'].shape[1]

    # brainflow wants 32 columns so pad with zeros if needed
    if num_cols < 32:
        extra_cols = np.zeros((mat['s'].shape[0], 32 - num_cols))
        mat['s'] = np.hstack((mat['s'], extra_cols))

    # stick a time column in at position 30
    time_column = np.arange(mat['s'].shape[0]) / frequency
    mat['s'][:, 30] = time_column

    print("Time:", mat['s'][:, 30])
    data = mat['s']
    print("CSV # Rows: ", len(data))
    np.savetxt(f'data/{file}.csv', data, delimiter=",")

# grabs event markers from the .mat file
def get_events(file):
    """Load raw event markers and convert them into parsed event labels.

    Parameters
    ----------
    file : str
        Base dataset name without extension, for example 'A01T'.

    Returns
    -------
    None
        The parsed event table is stored in module-level state.

    Side Effects
    ------------
    - Loads ./data/<file>.mat.
    - Populates parsed_event_pos via parse_event().
    """

    mat = scipy.io.loadmat(f'./data/{file}.mat')
    parsed_event_pos = parse_event(mat['EVENTTYP'], mat['EVENTPOS'])


# turns event codes (numbers) into readable names
def parse_event(event_type, event_pos):
    """Convert raw event codes into readable event labels and positions.

    Parameters
    ----------
    event_type : array-like
        Raw event code values from the .mat file.
    event_pos : array-like
        Sample positions aligned with event_type.

    Returns
    -------
    list[list[object]]
        Parsed event entries of the form [event_name, sample_position].

    Side Effects
    ------------
    - Appends parsed labels to the module-level parsed_event list.
    - Appends sample positions to the module-level parsed_pos list.
    - Appends paired entries to the module-level parsed_event_pos list.
    """

    # keeps count so we can label them like "left hand #1", "left hand #2", etc
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
            case 276:  # eyes open baseline
                event_index['Idling EEG (eyes open)'] += 1
                parsed_event.append(f"Idling EEG (eyes open) #{event_index['Idling EEG (eyes open)']}")
            case 277:  # eyes closed baseline
                event_index['Idling EEG (eyes closed)'] += 1
                parsed_event.append(f"Idling EEG (eyes closed) #{event_index['Idling EEG (eyes closed)']}")
            case 768:  # trial starting
                parsed_event.append(f"Start of a trial")
            case 769:  # imagine moving left hand
                event_index['Cue onset left (class 1)'] += 1
                parsed_event.append(f"Cue onset left (class 1) #{event_index['Cue onset left (class 1)']}")
            case 770:  # imagine moving right hand
                event_index['Cue onset right (class 2)'] += 1
                parsed_event.append(f"Cue onset right (class 2) #{event_index['Cue onset right (class 2)']}")
            case 771:  # imagine moving foot
                event_index['Cue onset foot (class 3)'] += 1
                parsed_event.append(f"Cue onset foot (class 3) #{event_index['Cue onset foot (class 3)']}")
            case 772:  # imagine moving tongue
                event_index['Cue onset tongue (class 4)'] += 1
                parsed_event.append(f"Cue onset tongue (class 4) #{event_index['Cue onset tongue (class 4)']}")
            case 783:  # unknown cue
                event_index['Cue unknown'] += 1
                parsed_event.append(f"Cue unknown #{event_index['Cue unknown']}")
            case 1023:  # bad trial, toss it
                event_index['Rejected trial'] += 1
                parsed_event.append(f"Rejected trial #{event_index['Rejected trial']}")
            case 1072:  # eye movement noise
                event_index['Eye movements'] += 1
                parsed_event.append(f"Eye movements #{event_index['Eye movements']}")
            case 32766:  # new run starting
                parsed_event.append(f"Start of a new run")
            case _:  # no idea what this code is
                event_index['Parse Error'] += 1
                parsed_event.append(f"Parse Error #{event_index['Parse Error']}")

    # grab where each event sits in the data
    for i in range(len(event_pos)):
        parsed_pos.append(int(event_pos[i][0]))

    # pair up names with positions
    for i in range(len(parsed_pos)):
        parsed_event_pos.append([parsed_event[i], parsed_pos[i]])

    return parsed_event_pos

# cuts out only the trials the user wants from the full recording
def trim_desired_data(desired_trials, csv):
    """Trim selected trials from the full recording and save them for replay.

    Parameters
    ----------
    desired_trials : list[int]
        Indices into parsed_event_pos that identify the desired cue trials.
    csv : pandas.DataFrame
        The full BrainFlow-formatted EEG recording.

    Returns
    -------
    None
        The trimmed trial data is written to data/stream.csv.

    Side Effects
    ------------
    - Prints trial start, cue, and end positions.
    - Writes a new data/stream.csv file.
    """

    desired_trial_pos = []
    desired_trial_start_pos = []
    desired_trial_end_pos = []
    trimmed_data = []

    # find where each selected trial starts and ends
    for i in range(len(desired_trials)):
        desired_trial_pos.append(parsed_event_pos[desired_trials[i]][1])
        desired_trial_start_pos.append(parsed_event_pos[desired_trials[i] - 1][1])
        desired_trial_end_pos.append(parsed_event_pos[desired_trials[i] + 1][1])

    print("Start: ", desired_trial_start_pos)
    print("Pos: ", desired_trial_pos)
    print("End: ", desired_trial_end_pos)

    # grab all rows between start and end for each trial
    for i in range(len(desired_trials)):
        for j in range(desired_trial_end_pos[i] - desired_trial_start_pos[i]):
            trimmed_data.append(csv.iloc[j + desired_trial_start_pos[i]])

    # save it so we can replay it later
    np.savetxt(f'data/stream.csv', trimmed_data, delimiter=",")



# wishlist:
# clean up code
# UI
# make it so mlx script is run in python

# prints all 22 EEG channels as a table
def show_data(data):
    """Print the first 22 EEG channels as a pandas table.

    Parameters
    ----------
    data : numpy.ndarray
        BrainFlow data matrix where each row is a channel.

    Returns
    -------
    None
        The channel table is printed to the console.
    """

    df = pd.DataFrame()
    for i in range(22):
        df["node ", (i + 1)] = data[i]
    print(df)

# shows a menu of all events so user can pick which trials to look at
def table_of_contents():
    """Print the parsed event table with labels and sample positions.

    Returns
    -------
    None
        The event table is printed to the console.
    """

    df = pd.DataFrame()
    df['EVENT'] = parsed_event
    df['Position'] = parsed_pos
    print(df.to_string())

# the main flow: load data, pick trials, visualize
def launch_board_ds2a(file, stream_duration):
    """Load a dataset, let the user choose trials, and replay them.

    Parameters
    ----------
    file : str
        Base dataset name without extension.
    stream_duration : float
        Kept for compatibility with the original script; not used directly.

    Returns
    -------
    None
        This function coordinates the full interactive playback workflow.

    Side Effects
    ------------
    - Creates data/<file>.csv if missing.
    - Prints the event table.
    - Prompts the user for trial indices.
    - Writes data/stream.csv.
    - Starts live playback on the trimmed stream.
    """

    desired_trials = []

    # make a csv if we don't have one yet
    if not(os.path.isfile(f'./data/{file}.csv')):
        print("Creating CSV file")
        create_csv(file)

    # load the csv and get all the events
    eeg_csv_file = pd.read_csv(f'./data/{file}.csv', header=None)
    print("eeg_csv_file: ", eeg_csv_file)
    get_events(file)
    print("Launching playback board with file ", file)

    # show whats in the recording
    table_of_contents()

    # let user pick which trials they want
    user_input = input("Enter Desired Trial Numbers (2 3 12 ... 4): ").split(' ')

    for i in range(len(user_input)):
        desired_trials.append(int(user_input[i]))

    # keep only the chosen trials
    trim_desired_data(desired_trials, eeg_csv_file)

    # play it back with the live graph
    import_to_mne(f'./data/stream.csv')
    #data = import_to_mne(f'./data/stream.csv')
    #show_data(data)

# kicks everything off with the A04T dataset
def main():
    """Program entry point for the playback workflow.

    Returns
    -------
    None
        Delegates to launch_board_ds2a().
    """

    launch_board_ds2a('A04T', 0)

if __name__ == '__main__':
    main()
