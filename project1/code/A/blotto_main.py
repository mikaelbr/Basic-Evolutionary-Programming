#! /usr/bin/env ipython
# -*- coding: utf-8 -*-

"""
---------------------------------------------------------------------------
*                Blotto Solution - Evolutionary Algorithms                *
---------------------------------------------------------------------------

    Project 1.B – IT3708 - Subsymbolic Methods in AI, Spring 2012
    Programmed by Mikael Brevik

    A solution for the Blotto Competition/Problem using a Evolutionary
    Algorithm.  

    Uses iPython for data plotting. 

---------------------------------------------------------------------------
*                         // EXAMPLE RUN //                               *
---------------------------------------------------------------------------
[Will run with all standard options]
$ ipython blotto_main.py 


To use different selection strategy and reproduction/mutation, you have to
do this by hand (code)
---------------------------------------------------------------------------
*                Blotto Solution - Evolutionary Algorithms                *
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
import random

BATTLE_POINT = {
    'WIN': 2,
    'TIE': 1,
    'LOSS': 0
}

# Genotype/Phenotype subclass
class BattleStrategy(Indevidual):

    def __init__(self, battles, fitness_func, value = None):
        self.battles = battles

        if value is None:
            value = self.generate_bits()


        self.fitness_value = None
        super(BattleStrategy, self).__init__(value, fitness_func)

    def generate_bits (self):
        def get_gene():
            return '{:04b}'.format(int(random.randint(0,10)))
        return ''.join(get_gene() for x in range(self.battles) )

    def toPhenotype(self):
        weights = []
        for i in range(0, len(self.value), 4):
            weights.append(int(self.value[i:i+3], 2));

        # Need to check if the sum of weights equals zero
        if sum(weights) == 0:
            weight_len = len(weights)
            for i in range(len(weights)):
                weights[i] = 1/float(weight_len)

        # Normalize
        self.phenotype = [(i / float(sum(weights))) for i in weights]

    def create_child(self, value):
        return BattleStrategy(self.battles, self.fitness_func, ''+value)

    def entropy(self):
        """
            Calculate the entropy for the strategy. Uses formula:
            H(s) = - ∑(p_i * log_2(p_i), from: i=1, to: B)
            Where p_i is the fraction of ith battle (one element of phenotype) 
        """
        return -1.0 * sum([battle*math.log(battle, 2) for battle in self.phenotype if battle])


class RandomChangeValue(Mutation):

     def do(self, g1):
        random_position = random.randint(0, (len(g1.value)-4)/4) * 4
        new_value = '{:04b}'.format(int(random.randint(0,10)))
        g1.value = ''.join([g1.value[:random_position], new_value, g1.value[random_position+4:]])


class Colonels(Population):
    """
        Extention of the regular Population class. 
        With this we can alter the normal fitness test
        and battle two strategies with each other. 
    """
    def __init__(self, size, creation_closure, reployment_fraction = 0.0, strength_reduction = 0.0):
        super(Colonels, self).__init__(size, creation_closure)
        self.strength_reduction = strength_reduction
        self.reployment_fraction = reployment_fraction

    def test_fitness(self):
        """
            Use this to battle between two different
            colnels. 
        """
        # Reset all fitness for surviving adults. In case of
        # overproduction or generational mixing
        for item in self.children:
            item.fitness_value = 0


        # We need to define a fighting algorithm between all 
        # different strategies.

        battles = self.children[0].battles
        pop = self.children[:]

        def do_battle(child1, child2):

            # if child1.fitness_value is None:
            #     child1.fitness_value = 0
            
            # if child2.fitness_value is None:
            #     child2.fitness_value = 0

            # Fight between children – Loop through battles
            child1_victories = child2_victories = 0
            reployment1 = reployment2 = 0.0
            strength1 = strength2 = 1.0
            
            for battle in range(battles):
                # List up resources for this battle. 
                # Take reployment in account. 
                resource1 = child1.phenotype[battle] + reployment1
                resource2 = child2.phenotype[battle] + reployment2

                if (resource1 * strength1) > (resource2 * strength2):
                    # Child 1 won, Child 2 lost
                    child1_victories += 1

                    # Reduce Child2's strength
                    strength2 -= self.strength_reduction

                    # Reploy remaining forces
                    if i < battles-1:
                        # Distribute remaining forces evenly to other battles.
                        # Reployment concatonates/accumelates.
                        reployment1 += (self.reployment_fraction * (resource2-resource1))/(battles-i+1)

                elif (resource1 * strength1) < (resource2 * strength2):
                    # Child 2 won, CHild 1 lost
                    child2_victories += 1

                    # Reduce Child1's strength
                    strength1 -= self.strength_reduction

                    # Reploy remaining forces
                    if i < battles-1:
                        # Distribute remaining forces evenly to other battles.
                        reployment2 += (self.reployment_fraction * (resource1-resource2))/(battles-i+1)

            if child1_victories == child2_victories:
                # Draw - both get 1 point
                child1.fitness_value += BATTLE_POINT['TIE']
                child2.fitness_value += BATTLE_POINT['TIE']
            elif child1_victories > child2_victories:
                # Child 1 won, Child 2 lost
                child1.fitness_value += BATTLE_POINT['WIN']
            else:
                # Child 2 won, CHild 1 lost
                child2.fitness_value += BATTLE_POINT['WIN']


        for i in range(len(pop)-1):
            for j in range(i+1, len(pop)):
                # Will have the cross product of all 
                # children
                do_battle(pop[i], pop[j])



class BlottoPlotter(Plotter):

    def __init__(self, path = ".", name = "onemax"):
        super(BlottoPlotter, self).__init__(path, name)
        self.avg_entropy = []

    def update (self, generation, population):
        super(BlottoPlotter, self).update(generation, population)
        if self.avg_entropy is None:
            self.avg_entropy = []

        entropy = [i.entropy() for i in self.children]

        self.avg_entropy.append(sum(entropy) / float(len(self.children)))

    def plot_entropy(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(self.generations, self.avg_entropy, 'r', label="Avg Entropy") 
        plt.xlabel('Generation')
        plt.ylabel('Avg Entropy')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3)

        #plt.show()
        fig.savefig(self.find_filename(self.name + "-entropy"))

# Fitness function
def fitness_func(phenotype_value):
    """
        Since we're overriding the normal fitness test,
        we don't need a fitness func.
        Just return 0
    """
    return 0



def create_binary_vector(battles):
    """
        A closure solution to pass arguments in the inner function.
        This way we can initiate the Indevidual subclass with 
        arbitrary number of arguments
    """

    def inner_closure ():
        return BattleStrategy(battles, None)

    return inner_closure

if __name__ == "__main__":

    import argparse
    # from IPython.config import loader


    parser = argparse.ArgumentParser(version='0.1', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-o', 
        action="store", 
        dest="output_file",
        type=str, 
        default="blotto",
        help='The threshold, if not optimized')

    parser.add_argument(
        '-ps', 
        dest="pop_size",
        type=float,
        action="store", 
        default=20,
        help='Population size')

    parser.add_argument(
        '-m', 
        action="store", 
        dest="mutation_probability",
        type=float, 
        default=0.4,
        help='Mutation probability')

    parser.add_argument(
        '-b', 
        action="store", 
        dest="birth_probability",
        type=float, 
        default=0.6,
        help='Birth probability')

    parser.add_argument(
        '-s', 
        action="store", 
        dest="geno_size",
        type=int,
        default=10,
        help='Number of battles.')

    parser.add_argument(
        '-g', 
        action="store", 
        dest="generations",
        type=int,
        default=400,
        help='Number of generations')


    args = parser.parse_args()

    pop_size = args.pop_size
    generations = args.generations
    output_size = 15 # int(pop_size - (pop_size/5))
    birth_probability = args.birth_probability
    mutation_probability = args.mutation_probability
    battles = args.geno_size
    output_file = args.output_file

    print args

    population = Colonels(pop_size, create_binary_vector(battles))

    adult_selection = SelectionStrategy(output_size, GenerationalMixing)
    parent_selection = SelectionStrategy(pop_size, None, SigmaScaling)

    reproduction = BinaryTwoPointCrossover(birth_probability) # Birth probability
    mutation = RandomChangeValue(mutation_probability) # Mutation probability

    if output_file is not None:
        plotter = BlottoPlotter("./plots", output_file)
    else:
        plotter = None

    ea = EA(population, adult_selection, parent_selection, reproduction, mutation, generations, plotter)
    ea.loop()



    if plotter is not None:
        plotter.plot()
        plotter.plot_entropy()

