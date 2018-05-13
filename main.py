"""Entry point to evolving the neural network. Start here."""
import json
import time
from multiprocessing import Process, Manager
from functools import reduce
from game_interface import Game
from optimizer import Optimizer
from run import score


def pool_score(network, game, i, return_dict):
    return_dict[i] = score(network, game)


def generate(generations, population, nn_param_choices):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)
    games = [Game() for _ in range(4)]
    processes_manager = Manager()
    return_dict = processes_manager.dict()

    # Evolve the generation.
    for i in range(generations):
        print("**Doing generation %d of %d**" %
                     (i + 1, generations))

        for j in range(0, len(networks), 4):
            p1 = Process(target=pool_score, args=(networks[j].network, games[0], j, return_dict))
            p2 = Process(target=pool_score, args=(networks[j + 1].network, games[1], j + 1, return_dict))
            p3 = Process(target=pool_score, args=(networks[j + 2].network, games[2], j + 2, return_dict))
            p4= Process(target=pool_score, args=(networks[j + 3].network, games[3], j + 3, return_dict))

            p1.start()
            p2.start()
            p3.start()
            p4.start()

            p1.join()
            p2.join()
            p3.join()
            p4.join()

        for k, v in return_dict.items():
            networks[k].score = v

        # Get the average accuracy for this generation.
        average_accuracy = reduce(lambda x, y: x + y, (n.score for n in networks)) / population

        # Print out the average accuracy each generation.
        print("Generation average: {}".format(average_accuracy))
        print('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)
            for network in networks:
                network._score = None

        if i % 10 == 0:
            with open('./checkpoint.json', 'w') as checkpoint:
                print("Writing checkpoint")
                json.dump([n.network for n in networks], checkpoint)
    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.score, reverse=True)

    # Print out the top 5 networks.
    print([n.network for n in networks[:5]])


def main():
    """Evolve a network."""
    generations = 200  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.

    nn_param_choices = {
        'neurons': [1, 64],
        'layers': [1, 4],
        'weight': [-1, 1]

    }

    print("**Evolving %d generations with population %d**" %
                 (generations, population))

    generate(generations, population, nn_param_choices)


if __name__ == '__main__':
    main()
