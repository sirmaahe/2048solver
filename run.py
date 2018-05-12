import math
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
    game.restart()

    prev_elements = []
    while not game.is_over():
        prev_layer = game.elements
        current_elements = game.elements

        if prev_elements == current_elements:
            break
        else:
            prev_elements = current_elements

        for layer in network:
            prev_layer = [calculate_neuron(prev_layer, n) for n in layer]

        max_neuron = max(enumerate(prev_layer), key=lambda x: x[1])
        direction = DIRECTIONS[max_neuron[0]]
        game.move(direction)

    return game.score
