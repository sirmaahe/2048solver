"""Class that represents the network to be evolved."""
import random
import logging


class Network:
    def __init__(self, neurons=32):
        self.score = 0
        self.human_score = 0
        self.neurons = neurons
        self.network = []

    def create_random(self):
        network = [[[random.uniform(*[-1, 1]) for _ in range(16)] for _ in range(self.neurons)]]

        network.append([
            [random.uniform(*[-1, 1]) for _ in range(self.neurons)]
            for _ in range(4)
        ])
        self.network = network

    def create_set(self, network):
        self.network = network

    def print_network(self):
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.score * 100))
