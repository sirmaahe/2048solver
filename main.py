"""Entry point to evolving the neural network. Start here."""
import json
import time
import random
from math import e
import numpy as np
from multiprocessing import Process, Manager, Pool
from functools import reduce
from game_interface import Game
from optimizer import Optimizer
from run import score


def get_avg(net):
    return reduce(lambda x, y: x + y, (n.human_score for n in net)) / len(net)


def pool_score(network, game):
    try:
        return score(network, game)
    except OverflowError:
        return 0, 0


def island(args):
    networks, optimizer, i, return_dict = args
    for p in range(50):
        for n in networks:
            n.human_score, n.score = pool_score(n.network, Game())

        if p < 49:
            networks = optimizer.evolve(networks)
    return_dict[i] = networks

pool = Pool(4)

def generate(generations, population, nn_param_choices, n_range, global_network):
    optimizer = Optimizer(nn_param_choices, n_range=n_range)
    processes_manager = Manager()
    return_dict = processes_manager.dict()

    # Evolve the generation.
    i = 1
    res = 0
    for _ in range(generations):
        pool.map(island, [
            (global_network[int((r * population / 4)): int(((r + 1) * population / 4))],
            optimizer, r, return_dict)
        for r in range(4)])

        new_global_network = []
        for r in range(4):
            new_global_network.extend(sorted(return_dict[r], key=lambda x: x.score, reverse=True)[:int(population / 4)])

        # if i % 10 == 0:
        average_accuracy = get_avg(new_global_network)
        res += average_accuracy
        if i % 2 == 0:
            print("Generation average: {}".format(average_accuracy))

        if i < population - 1:
            global_network = optimizer.evolve(new_global_network)
        i += 1

    with open('./checkpoint.json', 'w') as checkpoint:
        print("Writing checkpoint")
        json.dump([n.network for n in global_network], checkpoint)
    # Sort our final population.
    # networks = sorted(networks, key=lambda x: x.human_score, reverse=True)

    # Print out the top 5 networks.
    # print([n.network for n in networks[:5]])
    return res/generations, global_network


def main():
    """Evolve a network."""
    generations = 5  # Number of times to evole the population.
    population = 40  # Number of networks in each generation.

    nn_param_choices = {
        # 'neurons': [1, 64],
        # 'layers': [1, 4],
        'weight': [-1, 1]
    }
    optimizer = Optimizer(nn_param_choices, n_range=(-5, 5))
    global_network = optimizer.create_population(population)
    score, global_network = generate(generations, population, nn_param_choices, n_range=(-2, 2), global_network=global_network)
    for i in reversed(np.arange(2, 500, 1)):
        new_score, new_network = generate(generations, population, nn_param_choices, n_range=(-2, 2), global_network=global_network)
        delta_score = new_score - score
        if delta_score > 0:
            global_network = new_network
            score = new_score
        elif pow(e, delta_score/i) > random.random():
            global_network = new_network
            score = new_score
        print('-'*80)
        print("{}: {}".format(i, score))
        print('-'*80)


if __name__ == '__main__':
    main()
