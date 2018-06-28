"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
import os.path
import json
import random
from itertools import zip_longest
from network import Network


def zip_longest_wrapper(one, two):
    one = one if one else []
    two = two if two else []
    return zip_longest(one, two)


class Optimizer:
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, nn_param_ranges, retain=0.15,
                 random_select=0.05, random_create=1, mutate_chance=0.15, n_range=(-1, 1)):
        """Create an optimizer.

        Args:
            nn_param_ranges (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated

        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.random_create = random_create
        self.retain = retain
        self.n_range = n_range
        self.nn_param_ranges = nn_param_ranges

    def create_population(self, count):
        """Create a population of random networks.

        Args:
            count (int): Number of networks to generate, aka the
                size of the population

        Returns:
            (list): Population of network objects

        """
        if os.path.isfile('checkpoint.json'):
            with open('checkpoint.json') as checkpoint:
                networks = json.load(checkpoint)

            pop = [Network(self.nn_param_ranges) for _ in range(0, count)]

            for n, loaded in zip_longest(pop, networks):
                if loaded is None:
                    n.create_random()
                if n is None:
                    continue
                n.create_set(loaded)
            return pop

        pop = []
        for _ in range(0, count):
            # Create a random network.
            network = Network(self.nn_param_ranges)
            network.create_random()

            # Add the network to our population.
            pop.append(network)

        return pop

    def breed(self, mother, father):
        children = []
        for _ in range(2):
            child = []
            for m_l, f_l in zip(mother.network, father.network):
                layer = []
                for m_n, f_n in zip(m_l, f_l):
                    neuron = []
                    for m_w, f_w in zip(m_n, f_n):
                        neuron.append(random.choice([m_w, f_w]))
                    layer.append(neuron)
                child.append(layer)

            network = Network(self.nn_param_ranges)
            network.create_set(child)

            # Randomly mutate some of the children.
            network = self.mutate(network)

            children.append(network)

        return children

    def mutate(self, n):
        """Randomly mutate all parts of the network.

        Args:
            n (instance): The network parameters to mutate

        Returns:
            (Network): A randomly mutated network object

        """
        network = n.network
        for i, layer in enumerate(network):
            for j, neuron in enumerate(layer):
                for k, weight in enumerate(neuron):
                    if random.random() <= self.mutate_chance:
                        neuron[k] = weight + weight * random.uniform(-1, 1)
        return n

    def evolve(self, pop):
        """Evolve a population of networks.

        Args:
            pop (list): A list of network parameters

        Returns:
            (list): The evolved population of networks

        """
        # Sort on the scores.
        graded = [x for x in sorted(pop, key=lambda x: x.score, reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep.
        if len(graded) < 40:
            parents = graded[:]
        else:
            parents = graded[:retain_length]
        result = []
        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                result.append(individual)
        if self.random_create > random.random():
            network = Network(self.nn_param_ranges)
            network.create_random()
            result.append(network)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = 40 - len(result)
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        result.extend(children)

        return result
