import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtGui, QtCore


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtGui.QApplication([]) # UI processing events
        self.win = pg.GraphicsWindow(title='BrainFlow Plot', size=(800, 600)) # displays EEG

        self._init_timeseries()

        timer = QtCore.QTimer() # schedules when functions run
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms) # every 50 milliseconds, new data gets displayed on ui
        QtGui.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = list() # graphics, axes, layout
        self.curves = list() # eeg signal plots
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0) # PlotItem
            p.showAxis('left', False) # hides axis and menu
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot') # only add title to first plot
            self.plots.append(p) # add plot for channel
            curve = p.plot() # PlotDataItem, empty line chart
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points) 
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            # preprocessing EEG data
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.curves[count].setData(data[channel].tolist()) # adds eeg data to respective curve and plots

        self.app.processEvents()