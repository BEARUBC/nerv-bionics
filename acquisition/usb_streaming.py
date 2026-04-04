from mne_lsl.lsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    local_clock,
    resolve_streams,
)
import numpy as np
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

import visualization

params = BrainFlowInputParams() #get default parameters
params.master_board = BoardIds.CYTON_BOARD
params.serial_port = "COM3" #can be found in windows Device Manager under Ports when the Cyton board is connected
board = BoardShim(BoardIds.CYTON_BOARD, params) #Boardshim is the module that gets data from the board. For the Cyton board, we need to specify the board ID and the input port
#BoardShim.enable_dev_board_logger()

board.prepare_session()
board.start_stream ()
visualization.Graph(board)
# data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
data = board.get_board_data()  # get all data and remove it from internal buffer
board.stop_stream()
board.release_session()


# # arrays for each channel
# ch1=[]
# ch2=[]
# timestamps=[]
# for i in range(0,100):
#     # get a new sample
#     sample, timestamp = inlet.pull_sample()
#     print(sample)

#     #Sometimes the sample is none, tends to happen if connection is lost and re-established
#     if type(sample) != None:
#         ch1.append(sample[0])
#         ch2.append(sample[1])
#         timestamps.append(timestamp)
#     else:
#         print("aaaaaa none type :(")

# # free up resources
# inlet.close_stream()
# del inlet

# #plot what we collected
# def singlechannelgraph(sf,chdata,chno):
# # Plot the signal
#     fig, ax = plt.subplots(1, 1, figsize=(12, 4))
#     plt.plot(timestamps, chdata, lw=1.5, color='k')
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Voltage')
#     #plt.xlim([timestamps.min(), timestamps.max()])
#     plt.title('Channel %d EEG data'%(chno))
#     plt.show()

#singlechannelgraph(250, ch1, 1)