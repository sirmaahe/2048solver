"""Class that represents the network to be evolved."""
import random
import logging
from run import score


class Network:
    def __init__(self, nn_param_ranges=None):
        self.score = None
        self.nn_param_ranges = nn_param_ranges
        self.network = []

    def create_random(self):
        network = []
        prev_length = 16

        for i in range(random.randint(*self.nn_param_ranges['layers'])):
            new_layer = [
                [random.uniform(*self.nn_param_ranges['weight']) for _ in range(prev_length)]
                for _ in range(random.randint(*self.nn_param_ranges['neurons']))
            ]

            network.append(new_layer)
            prev_length = len(new_layer)

        network.append([
            [random.uniform(*self.nn_param_ranges['weight']) for _ in range(prev_length)]
            for _ in range(4)
        ])

        self.network = network

    def create_set(self, network):
        self.network = network

    def print_network(self):
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.score * 100))
