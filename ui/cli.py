import argparse
import configparser

# ##########################################################
# ### Argparse & Config Parser
# ##########################################################

ap = argparse.ArgumentParser()
config = configparser.ConfigParser()
config.read("config.ini") # Location of config.ini file.

ap.add_argument("-i", "--input", type=str, default=configuration.DEFAULT_INPUT,
	help="lokasi file input")
ap.add_argument("-i", "--input", type=str, default=configuration.DEFAULT_INPUT,
	help="lokasi file input")

args = vars(ap.parse_args())

try:
	while True:



except:


finally:





# ##########################################################