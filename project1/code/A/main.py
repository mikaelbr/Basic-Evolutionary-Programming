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

# Closure function for deliverable 5 in project A
def fitness_test_target_bitstring(gene_size):
    bitstring_target = [int(random.choice(('0', '1'))) for i in range(gene_size)]
    print "BITSTRING TARGET: %s" % bitstring_target
    def fitness_test(phenotype_value):
        fitness = 0
        for i in range(len(bitstring_target)):
            if bitstring_target[i] == phenotype_value[i]:
                fitness += 1
        return fitness
    return fitness_test


def create_binary_vector(fitness_test, gene_size):
    """
        A closure solution to pass arguments in the inner function.
        This way we can initiate the Indevidual subclass with 
        arbitrary number of arguments
    """
    #fitness_func_target = fitness_test_target_bitstring(gene_size)
    fitness_func_target = fitness_test
    def inner_closure ():
        return BinaryVector(gene_size, fitness_func_target)

    return inner_closure


# Default values for all params. 
std_values = {
    'output_file': 'onemax',
    'do_plot': True,
    'pop_size':  20,
    'mutation_probability': 0.2,
    'birth_probability': 0.6,
    'geno_size': 40,
    'generations': 100,
    'protocol': 'FullReplacement',
    'mechanism': 'SigmaScaling',
    'reproduction': 'BinaryUniformCrossover',
    'elitism': 0,
    'truncation': 0,
}

if __name__ == "__main__":

    import argparse
    from IPython.config import loader


    parser = loader.ArgumentParser(version='0.1', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-o', 
        action="store", 
        dest="output_file",
        type=str, 
        default=std_values['output_file'],
        help='The threshold, if not optimized')

    parser.add_argument(
        '-noplot', 
        dest="do_plot",
        action="store_false", 
        default=std_values['do_plot'],
        help='A boolean value ')

    parser.add_argument(
        '-ps', 
        dest="pop_size",
        type=float,
        action="store", 
        default=std_values['pop_size'],
        help='Population size')

    parser.add_argument(
        '-m', 
        action="store", 
        dest="mutation_probability",
        type=float, 
        default=std_values['mutation_probability'],
        help='Mutation probability')

    parser.add_argument(
        '-b', 
        action="store", 
        dest="birth_probability",
        type=float, 
        default=std_values['birth_probability'],
        help='Birth probability')


    parser.add_argument(
        '-e', 
        action="store", 
        dest="elitism",
        type=float, 
        default=std_values['elitism'],
        help='Elitism ( e < 1 means fraction ). ')

    parser.add_argument(
        '-t', 
        action="store", 
        dest="truncation",
        type=float, 
        default=std_values['truncation'],
        help='truncation - a fraction ')

    parser.add_argument(
        '-s', 
        action="store", 
        dest="geno_size",
        type=int,
        default=std_values['geno_size'],
        help='Geno size. Number of bits.')

    parser.add_argument(
        '-g', 
        action="store", 
        dest="generations",
        type=int,
        default=std_values['generations'],
        help='Number of generations')

    parser.add_argument(
        '-protocol', 
        action="store", 
        dest="protocol",
        type=str, 
        default=std_values['protocol'],
        help='The protocol for using adult selection')

    parser.add_argument(
        '-mechanism', 
        action="store", 
        dest="mechanism",
        type=str, 
        default=std_values['mechanism'],
        help='The mechanism for using parent selection')

    parser.add_argument(
        '-reproduction', 
        action="store", 
        dest="reproduction",
        type=str, 
        default=std_values['reproduction'],
        help='The reproduction method')

    import sys
    import types
    
    def str_to_class(field):
        try:
            identifier = getattr(sys.modules[__name__], field)
        except AttributeError:
            raise NameError("%s doesn't exist." % field)
        if isinstance(identifier, (types.ClassType, types.TypeType)):
            return identifier
        raise TypeError("%s is not a class." % field)


    args = parser.parse_args()
    output_size = int(args.pop_size * 0.1)

    print args

    population = Population(args.pop_size, create_binary_vector(fitness_test, args.geno_size))

    adult_selection = SelectionStrategy(output_size, str_to_class(args.protocol))
    parent_selection = SelectionStrategy(args.pop_size, None, str_to_class(args.mechanism), args.elitism, args.truncation)

    reproduction = str_to_class(args.reproduction)(args.birth_probability) # , 0.3) # Birth probability
    mutation = BinaryStringInversion(args.mutation_probability) # Mutation probability

    if args.output_file is not None and args.do_plot is True:
        plotter = Plotter("./plots", args.output_file)
    else:
        plotter = None

    ea = EA(population, adult_selection, parent_selection, reproduction, mutation, args.generations, plotter)
    ea.loop()

    if plotter is not None:
        plotter.plot()

