from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

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
