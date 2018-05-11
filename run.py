import math
from functools import reduce
from operator import add
from game_interface import game


DIRECTIONS = ['up', 'down', 'left', 'right']


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def calculate_neuron(prev, weights):
    elems = [x * y for x, y in zip(prev, weights)]
    signal = reduce(add, elems)
    neuron = sigmoid(signal)
    return neuron


def score(network):
    game.restart()

    prev_elements = []
    attempt = 0
    while not game.is_over() and attempt <= 3:
        prev_layer = game.elements
        current_elements = game.elements

        if prev_elements == current_elements:
            attempt += 1
        else:
            attempt = 0
            prev_elements = current_elements

        prev_layer.append(attempt)

        for layer in network:
            prev_layer = [calculate_neuron(prev_layer, n) for n in layer]

        max_neuron = max(enumerate(prev_layer), key=lambda x: x[1])
        direction = DIRECTIONS[max_neuron[0]]
        game.move(direction)

    return game.score
