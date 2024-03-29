from abc import ABCMeta, abstractmethod
from operator import attrgetter
import random
import math
import copy

class SelectionProtocol(object):
    """
        Base class for the different selection protocols
    """

    def __init__(self, size, population):
        self.size = size
        self.population = population

    def do(self):
        pass

    def reduce_pool(self, adults):
        temp_pop = adults[:]
        temp_pop.sort(key=attrgetter('fitness_value'), reverse=True)
        return temp_pop[:self.size]


class FullReplacement(SelectionProtocol):

    def do(self):
        self.population.adults = self.population.children[:]
        self.population.children = []


class OverProduction(FullReplacement):

    def do(self):
        super(OverProduction, self).do()
        self.population.adults = self.reduce_pool(self.population.adults)

class GenerationalMixing(SelectionProtocol):

    def do(self):
        self.population.adults += self.population.children[:]
        self.population.children = []
        self.population.adults = self.reduce_pool(self.population.adults)




class SelectionMechanism(object):
    """
        Base class for the different selection mechanisms 
    """
    def __init__(self, size, population):
        self.population = population
        self.size = size

    def do(self):
        pass

    def probability_func (self, fitness):
        pass

    def roulette_wheel(self):

        # Calculate size of sectors
        tmp_population = self.population.adults[:]
        wheel = []
        accumulated = 0.0

        for adult in tmp_population:
            accumulated += self.probability_func(adult.fitness_value)
            wheel.append((adult, accumulated))

        new_population = []
        while (len(new_population) < min([self.size, len(tmp_population)])):
            limit = random.uniform(0, accumulated);
            # Find a 
            for adult, probability in wheel:
                if probability > limit:
                    new_population.append(copy.deepcopy(adult))
                    break

        self.population.parents = new_population[:]
        return self.population.parents


class FitnessProportionate(SelectionMechanism):
    """
        Fitness values are scaled by avg population fitness. 
        Normalized
    """

    def set_values(self):
        self.total = sum([i.fitness_value for i in self.population.adults[:]])

    def probability_func (self, fitness):
        return (fitness) / float(self.total)

    def do(self):
        self.set_values();
        return self.roulette_wheel()


class SigmaScaling(SelectionMechanism):
    """
        Selection Mechanism with selection pressure inherent in the raw fitness value using
        the population's scaling and variance. 

        Formula: 1.0 + ((fitness - avg_fitness)/ 2*std_deviation)
    """

    def set_values(self):
        fitness = [i.fitness_value for i in self.population.adults[:]]
        self.pop_length = float(len(fitness))
        self.avg_fitness = sum(fitness) / self.pop_length

        # Var
        e_square = sum(math.pow(i, 2) for i in fitness) / self.pop_length
        self.std_deviation = math.sqrt(e_square - math.pow(self.avg_fitness, 2.0))

    def probability_func (self, fitness):
        if self.std_deviation == 0:
            return 1.0
        
        return ((1.0 + ((fitness - self.avg_fitness) / (2*self.std_deviation))) / self.pop_length)

    def do(self):
        self.set_values();
        return self.roulette_wheel()

class Tournament(SelectionMechanism):

    def probability_func (self, fitness):
        pass

    def do(self, k=2, e=0.8):
        tmp_population = self.population.adults[:]

        new_population = []
        while (len(new_population) < min([self.size, len(tmp_population)])):
            adults = random.sample(tmp_population, k)
            adults.sort(key=attrgetter('fitness_value'), reverse=True)
            new_population.append(copy.deepcopy(adults[int(random.random() > e)]))
        
        self.population.parents = new_population[:]
        return self.population.parents
        

class HighestFitness(SelectionMechanism):

    def do(self):
        pop_sorted = sorted(self.population.adults, key=attrgetter('fitness_value'), reverse=True)
        self.population.parents = pop_sorted[:self.size]
        return self.population.parents


class SelectionStrategy(object):

    def __init__(self, size, protocol = GenerationalMixing, mechanism=None, elitism=None, truncation=None, output_size = None):

        self.size = size;
        self.protocol_name = protocol
        self.protocol = None
        self.mechanism_name = mechanism
        self.mechanism = None
        
        self.output_size = output_size
        if output_size is None:
            self.output_size = self.size

        # Elitism
        self.elitism = elitism
        if self.elitism is None:
            self.elitism = 0

        # Truncation
        self.truncation = truncation
        if self.truncation is None:
            self.truncation = 0
        
    # IMPLEMENT ELITISM?
        
    def select(self, population):

        # Check if we have protocol defined for the strategy
        if self.protocol_name is not None:
            orig_size = len(population.children)
            if self.protocol is None:
                self.protocol = self.protocol_name(self.output_size, population)

            # Do selection protocol
            self.protocol.do()

            # Need to check if we have enough indeviduals
            if len(population.adults) < orig_size:
                population.adults.sort(key=attrgetter('fitness_value'), reverse=True)
                for i in range(self.size - len(population.adults)):
                    population.adults.append(copy.deepcopy(population.adults[i % len(population.adults)]))


        # Check if we have mechanism defined for the strategy
        if self.mechanism_name is not None:
            tmp_size = self.size
            population.parents = []
            if self.mechanism is None:
                self.mechanism = self.mechanism_name(self.size, population)

            
            # Elitism - Select a few to go directly to the next generation.
            # Do not mutate or recombine.
            new_population = population.adults[:]
            new_population.sort(key=attrgetter('fitness_value'), reverse=True)
            
            if self.elitism > 0:
                E = self.elitism
                if E < 1: # a fraction
                    E = int(len(new_population)*E)
                    population.elitsm = copy.deepcopy(new_population[:E])

                for i in population.elitsm:
                    print "Elitism: %s" % i.fitness_value
                self.size -= E

            # Truncation - Removal of parents as parents.
            population.adults.sort(key=attrgetter('fitness_value'), reverse=True)
            if self.truncation > 0:
                F = int(len(population.adults)*self.truncation)
                del population.adults[-F:]

            # Do selection mechanism
            self.mechanism.do()

            # Need to check if we have enough indeviduals
            if len(population.parents) < self.size:
                population.parents.sort(key=attrgetter('fitness_value'), reverse=True)
                for i in range(self.size - len(population.parents)):
                    population.parents.append(copy.deepcopy(population.parents[i % len(population.parents)]))

            self.size = tmp_size

