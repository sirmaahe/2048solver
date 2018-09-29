"""Class that represents the network to be evolved."""
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import logging


class Network:
    def __init__(self, nn_param_ranges=None):
        self.score = 0
        self.human_score = 0
        self.nn_param_ranges = nn_param_ranges
        self.network = []

    def create_random(self):
        network = Sequential()
        # network.add(InputLayer(batch_input_shape=(16,)))
        network.add(Dense(22, input_shape=(16,), activation='sigmoid'))
        network.add(Dense(4, activation='sigmoid'))
        network.compile('adam', loss='mse')
        self.network = network

    def create_set(self, network):
        self.network = network

    def print_network(self):
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.score * 100))
