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


def get_avg(n):
    return reduce(lambda x, y: x + y, (n.human_score for n in n)) / len(n)


def pool_score(args):
    network, game, j, return_dict = args
    try:
        return_dict[j] = score(network, game)
    except OverflowError:
        return_dict[j] = 0, 0


def island(args):
    networks, optimizer, i, return_dict = args
    for p in range(50):
        for n in networks:
            n.human_score, n.score = pool_score(n.network, Game())

            networks = optimizer.evolve(networks)
    return_dict[i] = networks

pool = Pool(4)

def generate(generations, population, nn_param_choices, n_range, global_network):
    optimizer = Optimizer(nn_param_choices, n_range=n_range)
    processes_manager = Manager()
    return_dict = processes_manager.dict()

    # Evolve the generation.
    i = 1
    while True:
        pool.map(pool_score, [(global_network[j].network, Game(), j, return_dict) for j in range(0, len(global_network))])

        for k, v in return_dict.items():
            global_network[k].human_score, global_network[k].score = v

        if i != generations - 1:
            global_network = optimizer.evolve(global_network)
            print("{}: {}".format(i, get_avg(global_network)))
        i += 1

    with open('./checkpoint.json', 'w') as checkpoint:
        print("Writing checkpoint")
        json.dump([n.network for n in global_network], checkpoint)
    # Sort our final population.
    # networks = sorted(networks, key=lambda x: x.human_score, reverse=True)

    # Print out the top 5 networks.
    # print([n.network for n in networks[:5]])
    return global_network


def main():
    """Evolve a network."""
    generations = 100  # Number of times to evole the population.
    population = 40  # Number of networks in each generation.

    nn_param_choices = {
        # 'neurons': [1, 64],
        # 'layers': [1, 4],
        'weight': [-1, 1]
    }
    optimizer = Optimizer(nn_param_choices, n_range=(-5, 5))
    global_network = optimizer.create_population(population)
    global_network = generate(generations, population, nn_param_choices, n_range=(-1, 1), global_network=global_network)
    for i in reversed(np.arange(0, 2, 0.1)):
        global_network = generate(generations, population, nn_param_choices, n_range=(-1, 1), global_network=global_network)
        # delta_score = get_avg(new_network) - get_avg(global_network)
        # if delta_score > 0:
        #     global_network = new_network
        # elif pow(e, delta_score/i) > random.random():
        #     global_network = new_network
        print('-'*80)
        print("{}: {}".format(i, get_avg(global_network)))
        print('-'*80)


if __name__ == '__main__':
    main()
