import math
import time
import numpy as np
from functools import reduce
from operator import add


DIRECTIONS = ['up', 'down', 'left', 'right']


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def calculate_neuron(prev, weights):
    elems = [x * y for x, y in zip(prev, weights)]
    signal = reduce(add, elems)
    neuron = sigmoid(signal)
    return neuron


def score(network, game):
    prev_elements = []
    steps = 1
    stime = time.time()
    while True:
        prev_layer = game.elements
        current_elements = game.elements

        if prev_elements == current_elements:
            break
        else:
            prev_elements = current_elements
        prev_layer = np.asarray(prev_layer)
        prev_layer = np.asarray([prev_layer])
        scoring = network.predict(prev_layer, batch_size=1028)
        max_neuron = max(enumerate(scoring[0]), key=lambda x: x[1])
        direction = DIRECTIONS[max_neuron[0]]
        game.move(direction)
        steps += 1
    print(time.time() - stime)
    return game.score, game.score
