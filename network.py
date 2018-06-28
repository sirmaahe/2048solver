"""Class that represents the network to be evolved."""
import random
import logging


class Network:
    def __init__(self, nn_param_ranges=None):
        self.score = 0
        self.human_score = 0
        self.nn_param_ranges = nn_param_ranges
        self.network = []

    def create_random(self):
        network = [[[random.uniform(*[-1, 1]) for _ in range(16)] for _ in range(24)]]

        network.append([
            [random.uniform(*[-1, 1]) for _ in range(24)]
            for _ in range(4)
        ])
        self.network = network

    def create_set(self, network):
        self.network = network

    def print_network(self):
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.score * 100))
