"""Entry point to evolving the neural network. Start here."""
import json
from multiprocessing import Process, Manager
from functools import reduce
from game_interface import Game
from optimizer import Optimizer
from run import score


def pool_score(network, game, i, return_dict):
    return_dict[i] = score(network, game)


processes = 10
games = [Game() for _ in range(processes)]


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
    processes_manager = Manager()
    return_dict = processes_manager.dict()

    # Evolve the generation.
    for i in range(generations):
        print("**Doing generation %d of %d**" % (i + 1, generations))

        for j in range(0, len(networks), processes):
            jobs = []
            for k in range(processes):
                jobs.append(
                    Process(target=pool_score, args=(networks[j + k].network, games[k], j + k, return_dict))
                )

            [p.start() for p in jobs]
            [p.join() for p in jobs]

        for k, v in return_dict.items():
            networks[k].human_score, networks[k].score = v

        # Get the average accuracy for this generation.
        average_accuracy = reduce(lambda x, y: x + y, (n.human_score for n in networks)) / population

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
    networks = sorted(networks, key=lambda x: x.human_score, reverse=True)

    # Print out the top 5 networks.
    print([n.network for n in networks[:5]])


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
    try:
        main()
    except Exception as e:
        [g.close() for g in games]
        raise e
