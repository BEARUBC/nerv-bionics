import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import logging
import PyQt6
import pyqtgraph as pg

import numpy as np
import argparse
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore
import pandas as pd

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        # Use GraphicsLayoutWidget instead of GraphicsWindow
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle('BrainFlow Plot')
        self.win.resize(800, 600)

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        self.app.exec()
        self.board_id = board_shim.get_board_id()
 

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()

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