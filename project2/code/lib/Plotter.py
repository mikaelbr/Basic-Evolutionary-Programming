import math
import fnmatch
import os
import re 
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Plotter(object):

    def __init__(self, path = ".", name = "onemax"):
        self.generations = []
        self.max_fitness = []
        self.avg_fitness = []
        self.std_dev_fitness = []
        self.name = name
        self.path = path

    def update (self, generation, population):
        self.children = copy.deepcopy(population.children[:])
        fitness = [i.fitness_value for i in self.children]
        fitness_size = len(fitness)

        self.generations.append(generation)

        self.max_fitness.append(max(fitness))
        avg = sum(fitness)/fitness_size
        self.avg_fitness.append(avg)

        e_square = sum(math.pow(i, 2.0) for i in fitness)/fitness_size
        print e_square
        print math.pow(avg, 2.0)
        print e_square - math.pow(avg, 2.0)
        print math.sqrt(e_square - math.pow(avg, 2.0))
        self.std_dev_fitness.append(math.sqrt(e_square - math.pow(avg, 2.0)))

    def plot (self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(self.generations, self.max_fitness, 'r', label="Max") 
        ax.plot(self.generations, self.avg_fitness, 'b', label="Avg")
        ax.plot(self.generations, self.std_dev_fitness, 'g', label="Std_Dev")
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3)

        #plt.show()
        fig.savefig(self.find_filename(self.name))

    def find_filename(self, filename):
        max_int = 0
        for file in os.listdir(self.path):
            if fnmatch.fnmatch(file, filename + '-*.png'):
                tmp_int = int(re.sub(r'[^\d+]', '', file))
                if tmp_int > max_int:
                    max_int = int(tmp_int)

        return self.path + "/" + filename + "-" + str(max_int+1) + ".png"


    def print_data(self):
        print 'Generations: \n'
        print self.generations

        print 'Max Fitness: \n'
        print self.max_fitness

        print 'Avg Fitness: \n'
        print self.avg_fitness

        print 'Std Deviation: \n'
        print self.std_dev_fitness