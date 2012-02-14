#! /usr/bin/env ipython
# -*- coding: utf-8 -*-

"""
---------------------------------------------------------------------------
*               ONE MAX Solution - Evolutionary Algorithms                *
---------------------------------------------------------------------------

    Project 1.A – IT3708 - Subsymbolic Methods in AI, Spring 2012
    Programmed by Mikael Brevik

    A basic EA for solving the One Max Problem
    Should be modular enough to solve other evolutionary problems 
    aswell. Very object oriented, and based on polymorphy and 
    inheritance. 

---------------------------------------------------------------------------
*                         // EXAMPLE RUN //                               *
---------------------------------------------------------------------------
[Will run with all standard options]
$ ipython main.py 


To use different selection strategy and reproduction/mutation, you have to
do this by hand (code)
---------------------------------------------------------------------------
*               ONE MAX Solution - Evolutionary Algorithms                *
---------------------------------------------------------------------------
"""

# Testing - init vars
from lib.EA import *
from lib.Indevidual import *
from lib.Selection import *
from lib.Reproduction import *
from lib.Mutation import *
from lib.Population import *
from lib.Plotter import *

# Genotype/Phenotype subclass
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



def create_binary_vector(fitness_test, gene_size):
    """
        A closure solution to pass arguments in the inner function.
        This way we can initiate the Indevidual subclass with 
        arbitrary number of arguments
    """

    def inner_closure ():
        return BinaryVector(gene_size, fitness_test)

    return inner_closure


if __name__ == "__main__":

    import argparse
    from IPython.config import loader


    parser = loader.ArgumentParser(version='0.1', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-o', 
        action="store", 
        dest="output_file",
        type=str, 
        default="onemax",
        help='The threshold, if not optimized')

    parser.add_argument(
        '-ps', 
        dest="pop_size",
        type=float,
        action="store", 
        default=100,
        help='Population size')

    parser.add_argument(
        '-m', 
        action="store", 
        dest="mutation_probability",
        type=float, 
        default=0.0,
        help='Mutation probability')

    parser.add_argument(
        '-b', 
        action="store", 
        dest="birth_probability",
        type=float, 
        default=0.91,
        help='Birth probability')

    parser.add_argument(
        '-s', 
        action="store", 
        dest="geno_size",
        type=int,
        default=40,
        help='Geno size. Number of bits.')

    parser.add_argument(
        '-g', 
        action="store", 
        dest="generations",
        type=int,
        default=100,
        help='Number of generations')


    args = parser.parse_args()

    pop_size = args.pop_size
    generations = args.generations
    output_size = int(pop_size - (pop_size/5))
    birth_probability = args.birth_probability
    mutation_probability = args.mutation_probability
    geno_size = args.geno_size
    output_file = args.output_file

    print args

    population = Population(pop_size, create_binary_vector(fitness_test, geno_size))

    adult_selection = SelectionStrategy(output_size, FullReplacement)
    parent_selection = SelectionStrategy(pop_size, None, FitnessProportionate)

    reproduction = BinaryUniformCrossover(birth_probability) #, 0.3) # Birth probability
    mutation = BinaryStringInversion(mutation_probability, 2) # Mutation probability

    if output_file is not None:
        plotter = Plotter("./plots", output_file)
    else:
        plotter = None

    ea = EA(population, adult_selection, parent_selection, reproduction, mutation, generations, plotter)
    ea.loop()

    if plotter is not None:
        plotter.plot()

