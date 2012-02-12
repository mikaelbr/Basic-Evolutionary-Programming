#!/usr/bin/env python

# Testing - init vars
from lib.EA import *
from lib.Indevidual import *
from lib.Selection import *
from lib.Reproduction import *
from lib.Mutation import *
from lib.Population import *
from lib.Plotter import *

pop_size = 20
generations = 100
output_size = int(pop_size - (pop_size/5))
birth_probability = 1.0
mutation_probability = 0.25
geno_size = 40


# Genotype subclass
class BinaryVector(Indevidual):

    def __init__(self, gene_size, fitness_func, value = None):
        if value is None:
            value = ''.join(random.choice(('0', '1')) for i in range(gene_size))

        self.gene_size = gene_size
        super(BinaryVector, self).__init__(value, fitness_func)

    def toPhenotype(self):
        self.phenotype = list(map(int, self.value))

    def create_child(self, value):
        return BinaryVector(len(value), self.fitness_func, value)


# Fitness function
def fitness_test(phenotype_value):
    """
        Get the sum of the phenotype value.
        For this fitness test we need to count
        all the 1's in the list.
    """
    return sum(phenotype_value)



# A closure solution to pass arguments in the inner function.
def create_binary_vector(fitness_test, gene_size):

    def inner_closure ():
        return BinaryVector(gene_size, fitness_test)

    return inner_closure

population = Population(pop_size, create_binary_vector(fitness_test, geno_size))

adult_selection = SelectionStrategy(output_size, OverProduction)
parent_selection = SelectionStrategy(pop_size, None, SigmaScaling)

reproduction = BinaryUniformCrossover(birth_probability) # Birth probability
mutation = BinaryStringInversion(mutation_probability) # Mutation probability

plotter = Plotter()

ea = EA(population, adult_selection, parent_selection, reproduction, mutation, generations, plotter)
ea.loop()


print "Length: %d" % len(ea.population.children)
for item in ea.population.children[:]:

    item.toPhenotype()
    item.fitness()
    fitness = item.fitness_value;

    if fitness is None:
        fitness = 0
    print "Value %s, fitness: %d" % (item.value, fitness)


plotter.plot()
