"""Entry point to evolving the neural network. Start here."""
import json
import time
from multiprocessing import Process, Manager, Pool
from functools import reduce
from game_interface import Game
from optimizer import Optimizer
from run import score


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


def generate(generations, population, nn_param_choices):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    global_network = optimizer.create_population(population)
    processes_manager = Manager()
    return_dict = processes_manager.dict()

    # Evolve the generation.
    i = 1
    pool = Pool(4)
    while True:
        start = time.time()
        if i % 1 == 0:
            print("**Doing generation %d of %d**" % (i + 1, generations))

        pool.map(island, [
            (global_network[int((r * population / 4)): int(((r + 1) * population / 4))],
            optimizer, r, return_dict)
        for r in range(4)])

        new_global_network = []
        for r in range(4):
            new_global_network.extend(sorted(return_dict[r], key=lambda x: x.score, reverse=True)[:int(population / 4)])

        # if i % 10 == 0:
        average_accuracy = reduce(lambda x, y: x + y, (n.human_score for n in new_global_network)) / population
        if i % 1 == 0:
            print('-'*80)
            print("Generation average: {}".format(average_accuracy))
            print(time.time() - start)
            print('-'*80)

        global_network = optimizer.evolve(new_global_network)

        if i % 100 == 0:
            with open('./checkpoint.json', 'w') as checkpoint:
                print("Writing checkpoint")
                json.dump([n.network for n in global_network], checkpoint)
        i += 1
    # Sort our final population.
    # networks = sorted(networks, key=lambda x: x.human_score, reverse=True)

    # Print out the top 5 networks.
    # print([n.network for n in networks[:5]])


def main():
    """Evolve a network."""
    generations = 2000  # Number of times to evole the population.
    population = 40  # Number of networks in each generation.

    nn_param_choices = {
        # 'neurons': [1, 64],
        # 'layers': [1, 4],
        'weight': [-1, 1]
    }

    print("**Evolving %d generations with population %d**" %
                 (generations, population))

    generate(generations, population, nn_param_choices)


if __name__ == '__main__':
    main()
