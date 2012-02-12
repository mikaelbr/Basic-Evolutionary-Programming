import math
import matplotlib.pyplot as plt

class Plotter(object):

    def __init__(self):
        self.generations = []
        self.max_fitness = []
        self.avg_fitness = []
        self.std_dev_fitness = []

    def update (self, generation, population):
        children = population.children[:]
        fitness = [i.fitness_value for i in children]
        fitness_size = len(fitness)

        self.generations.append(generation)

        self.max_fitness.append(max(fitness))

        avg = sum(fitness)/fitness_size
        self.avg_fitness.append(avg)

        e_square = sum(map(lambda x: math.pow(float(x), 2.0), fitness)) / fitness_size
        self.std_dev_fitness.append(2.0 * math.sqrt(e_square - math.pow(avg, 2.0)))

    def plot (self):
        plt.plot(self.generations, self.max_fitness, 'r', self.generations, self.avg_fitness, 'b', self.generations, self.std_dev_fitness, 'g')
        plt.show()
       

    def print_data(self):
        print 'Generations: \n'
        print self.generations

        print 'Max Fitness: \n'
        print self.max_fitness

        print 'Avg Fitness: \n'
        print self.avg_fitness

        print 'Std Deviation: \n'
        print self.std_dev_fitness