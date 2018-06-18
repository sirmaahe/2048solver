"""Entry point to evolving the neural network. Start here."""
import json
from multiprocessing import Process, Manager
from functools import reduce
from game_interface import Game
from optimizer import Optimizer, NeuronOptimizer
from run import score


def pool_score(network, game, i, return_dict):
    try:
        return_dict[i] = score(network, game)
    except OverflowError:
        print('OverflowError')
        return_dict[i] = (0, 0)


def pool_gen(neurons, i, return_dict):
    return_dict[i] = score_param(10000, 40, neurons['count'])


processes = 5


def score_param(generations, population, neurons):
    optimizer = Optimizer(neurons)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        for j in range(0, len(networks)):
            try:
                networks[j].human_score, networks[j].score = score(networks[j].network, Game())
            except OverflowError:
                networks[j].human_score, networks[j].score = 0, 0

        networks = optimizer.evolve(networks)
        for network in networks:
            network._score = None

        if i % 10 == 0:
            print("**Doing generation %d of %d**" % (i + 1, generations))
            average_accuracy = reduce(lambda x, y: x + y, (n.human_score for n in networks)) / population
            print("Generation average: {}".format(average_accuracy))
            print('-'*80)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.human_score, reverse=True)

    return networks


def generate(generations, population):
    optimizer = NeuronOptimizer()
    neurons = optimizer.create_population(population)
    processes_manager = Manager()
    return_dict = processes_manager.dict()

    # Evolve the generation.
    for i in range(200):
        print("**Doing generation %d of %d**" % (i + 1, 200))

        for j in range(0, len(neurons), processes):
            jobs = []
            for k in range(processes):
                jobs.append(
                    Process(target=pool_gen, args=(neurons[j + k], j + k, return_dict))
                )

            [p.start() for p in jobs]
            [p.join() for p in jobs]

        for k, v in return_dict.items():
            neurons[k]['score'] = sum([n.score for n in v[:5]]) / 5

        # Get the average accuracy for this generation.

        # Print out the average accuracy each generation.

        if i % 1 == 0:
            average_accuracy = reduce(lambda x, y: x + y, (n['score'] for n in neurons)) / population
            print("Generation average: {}".format(average_accuracy))
            print('-'*80)

        print(neurons[:5])
        neurons = optimizer.evolve(neurons)


def main():
    """Evolve a network."""
    generations = 2000  # Number of times to evole the population.
    population = 10  # Number of networks in each generation.

    generate(generations, population)


if __name__ == '__main__':
    main()
