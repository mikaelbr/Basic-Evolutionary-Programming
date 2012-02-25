#! /usr/bin/env python
# -*- coding: utf-8 -*-

from lib.EA import *
from lib.Indevidual import *
from lib.Selection import *
from lib.Reproduction import *
from lib.Mutation import *
from lib.Population import *
from lib.Plotter import *
import random
from math import pow


param_range = {
    'a': [0.001, 0.2], 
    'b': [0.01, 0.3],
    'c': [-80, -30],
    'd': [0.1, 10],
    'k': [0.01, 1.0]
}

def get_target_data():

    def read_file(path):
        with open(path, 'r') as f:
            return [float(i) for i in f.readline().strip().split(' ')]


    return [
        read_file('data/izzy-train1.dat'), 
        read_file('data/izzy-train2.dat'), 
        read_file('data/izzy-train3.dat'), 
        read_file('data/izzy-train4.dat')
        ]


class SpikingNeuron(Indevidual):

    def __init__(self, gene_size, fitness_func, value = None):

        self.max_value = int(''.join('1' for i in range(gene_size)), 2)

        # Generate new genotype if not defined
        if value is None:
            binary_helper = '{:0'+str(gene_size)+'b}'

            # Create the genotype. A concatonated string of all genes
            value = ''.join([binary_helper.format(random.randint(0, self.max_value)) 
                for i in range(std_values['num_params'])])

        self.gene_size = gene_size
        self.fitness_value = None
        super(SpikingNeuron, self).__init__(value, fitness_func)
    
    
    def fit_range (self, params):
        # mapping x from (n1,n2) to (s1,s2) 
        # x'= s1 + (x*(s2-s1)/n2)

        a, b, c, d, k = params # extract

        pr = param_range
        mx = self.max_value

        a = pr['a'][0] + (a*(pr['a'][1] - pr['a'][0])/mx)
        b = pr['b'][0] + (b*(pr['b'][1] - pr['b'][0])/mx)
        c = pr['c'][0] + (c*(pr['c'][1] - pr['c'][0])/mx)
        d = pr['d'][0] + (d*(pr['d'][1] - pr['d'][0])/mx)
        k = pr['k'][0] + (k*(pr['k'][1] - pr['k'][0])/mx)

        return a, b, c, d, k

    def toPhenotype(self):
        """
            Here we calculate the spike train using a loop of
            "timesteps" iterations, and the derivations of u and v to 
            update the u and v values. 

            Formulas:
            v' = 1/tau * (kv^2 + 5v + 140 - u + I)
            u' = a/tau * (bv - u)

            Update:
            v += v'
            u += u'

            If v > spike threshold:
            v = c
            u += d

            a,b,c,d,k is all genes from a= gene 1, b = gene 2 etc..

        """
        # Collect all denary values from genotype, categorized by 5 bits interval
        params = [int(self.value[i:i+self.gene_size], 2) for i in range(0, len(self.value), self.gene_size)]

        # Need to encode values to fit to given ranges/intervals
        a, b, c, d, k = self.fit_range (params)

        spike_train = [] # What we'll produce

        # Initializations:
        v = -60
        u = 0

        spike_train.append(v)

        for t in range(std_values['timesteps']):
            tmp = v
            v += ( (k * pow(v, 2.0)) + (5*v) + 140 - u + std_values['I'])/std_values['tau']
            u += (a/float(std_values['tau'])) * ((b*tmp) - u)

            spike_train.append(v)

            if v > std_values['spike_threshold']:
                v = c
                u += d

        self.phenotype = spike_train


    def create_child(self, value):
        return SpikingNeuron(self.gene_size, self.fitness_func, ''+value)

class SDM(object):

    @staticmethod
    def compute_spike_times(train):
        """
            1. Define an activation threshold, T, above which the neuron 
            is considered to be spiking. For the Izhikevich model described 
            above (and others that closely mimic real neuron behavior) , a 
            typical value of T is 0 mV (millivolts), since most neurons spend 
            a majority of their time with negative activation levels.
            
            2. Move a k-timestep window (e.g. k = 5) along the spike train, 
            and any activation value that is a) in the exact middle of the 
            time window, b) above T, and c) the maximum of all activations 
            in the time window, is considered a spike.
            
            3. By using this window, you avoid the double-counting of spikes, 
            which will often consist of a few time points above the threshold.

        """
        T = 0 # Activation Threshold
        k = 5 # Window timestep

        spike_times = []
        num_spikes = len(train)

        
        i = j = (k - 1) // 2
        while (i < len(train) - j):
            if train[i] > T and train[i] == max(train[i-j:i+j+1]):
                spike_times.append(i)
                i += j # Jump to next window or?
            i += 1

        return spike_times
        """

        for i in range(0, num_spikes-k):
            # Move through windows of size k. 
            # Check if a spike fits constraints a, b and c from 2. in comments
            middle = (k-1)//2 + i # middle index
            if middle > num_spikes:
                continue

            if train[middle] > T and train[middle] == max(train[i:i+k]):
                spike_times.append(i)
                i += k

        return spike_times
        """


    @staticmethod
    def difference_penalty(spikes1, spikes2, train1):
        M_spikes = len(spikes1)
        N_spikes = len(spikes2)
        diff = abs(N_spikes - M_spikes)
        return diff*len(train1)/2*M_spikes 

    @staticmethod
    def spike_time_distance_metric(train1, train2, p = 2, penalty = True):
        """
            This metric simply compares the times at which corresponding 
            spikes occur, giving reduced similarity (in- crease distance) 
            with increasing gaps. Two spikes correspond when they have 
            identical indices in their respective spike-time lists. The 
            spike-time distance between two spike trains, Ta and Tb is 
            defined as:


            d_st(T^a,T^b) = 1/N * (sum(|t_i^a − t_i^b|^p, 0, N-1))^1/p
        """
        spikes1, spikes2 = SDM.compute_spike_times(train1), SDM.compute_spike_times(train2)

        if not spikes1:
            return 0

        N = min(len(spikes1), len(spikes2))
        tmp = sum(pow(abs(spikes2[i] - spikes1[i]), p) for i in range(N))

        # Implement A Spike-Count Difference Penalty
        if penalty:
            tmp += SDM.difference_penalty(spikes1, spikes2, train1)

        distance = math.sqrt(tmp) / N
        #print distance
        #print "Target: %s \n Spikes1: %s " % (spikes2, spikes1)
        # print "Target Train (%s): %s \n Train1 (%s): %s " % (len(train2), train2, len(train1), train1)
        return 1.0/distance


    @staticmethod
    def spike_interval_distance_metric(train1, train2, p = 2, penalty = True):
        """
            Spike intervals are simply the gaps between successive spike 
            times. This metric compares the lengths of corresponding 
            intervals, giving greater similarity for smaller differences. 
            The spike-interval distance between two trains is defined as:

            d_si(T^a,T^b) = 1/(N - 1) * (sum(|(t_i^a − t_(i-1)^a)−(t_i^b − t_(i-1)^b)|^p, 1, N-1))^1/p
        """
        spikes1, spikes2 = SDM.compute_spike_times(train1), SDM.compute_spike_times(train2)

        N = min(len(spikes1), len(spikes2))


        if N < 2:
            return 0

        tmp = sum(pow(abs((spikes2[i] - spikes2[i-1]) - (spikes1[i] - spikes1[i-1])), p) for i in range(1, N))

        # Implement A Spike-Count Difference Penalty
        if penalty:
            tmp += SDM.difference_penalty(spikes1, spikes2, train1)

        return 1.0/(pow(tmp, (1.0/p))/float((1.0/(N-1))))


    @staticmethod
    def waveform_distance_metric(train1, train2, p = 2, penalty = False):
        """
            The waveform metric is the simplest of all. It compares 
            cotemporaneous activation levels across the two spike trains, 
            with no special treatment for spikes. Hence, it does not use 
            the spike-time list at all. The waveform distance metric is 
            defined as:

            d_wf(T^a,T^b) = 1/M * (sum(|(v_i^a - v_i^b)|^p, 0, M-1))^1/p
        """
        time_points = len(train1)
        tmp = sum(pow(abs(train2[i] - train1[i]), p) for i in range(time_points))
        distance = float(pow(tmp, 1.0/p)/float(time_points))
        return 1.0/distance
        

# Fitness function
def fitness_test(metric_fn, target, penalty = True, p = 2.0):


    
    def closure_fn (phenotype_value):
        return metric_fn(phenotype_value, target, p, penalty)

    return closure_fn

class SpikeMutation(Mutation):

     def do(self, g1): # REWRITE!!!
        """Randomly mutate a gene in the genotype."""
        s = random.randint(0, std_values['num_params']-1) * g1.gene_size
        #print "Mutation s %s" % s

        binary_helper = '{:0'+str(g1.gene_size)+'b}'
        new_gene = binary_helper.format(random.randint(0, g1.max_value))

        #print "New Gene %s " % new_gene
        g1.value = (g1.value[:s] + new_gene + g1.value[s+g1.gene_size:])


def create_data(gene_size, metric_fn, target, penalty = True):
    """
        A closure solution to pass arguments in the inner function.
        This way we can initiate the Indevidual subclass with 
        arbitrary number of arguments
    """

    fitness_func_target = fitness_test(metric_fn, target, penalty)
    def inner_closure ():
        return SpikingNeuron(gene_size, fitness_func_target)

    return inner_closure


class SpikingNeuronPlotter(Plotter):

    def update (self, generation, population):

        self.population = population.children[:] + population.adults[:]
        self.population.sort(key=attrgetter('fitness'), reverse=True)


        super(SpikingNeuronPlotter, self).update(generation, population)

    def plot_act(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        best_result = self.population[0]
        spike_train = best_result.phenotype


        # Collect all denary values from genotype, categorized by 5 bits interval
        params = [int(best_result.value[i:i+best_result.gene_size], 2) for i in range(0, len(best_result.value), best_result.gene_size)]
        # Need to encode values to fit to given ranges/intervals
        a, b, c, d, k = best_result.fit_range (params)

        print "Best result: a: %s, d: %s, c: %s, d: %s, k: %s" % (a, b, c, d, k)

        x_axis = list(range(len(spike_train)));
        ax.plot(list(range(len(spike_train))), spike_train, 'r', label="Act") 
        plt.xlabel('Time(ms)')
        plt.ylabel('Activation-Level(mV)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3)

        #plt.show()
        fig.savefig(self.find_filename(self.name + "-act"))


# Default values for all params. 
std_values = {
    'output_file': 'spiking',
    'do_plot': True,
    'pop_size':  200,
    'mutation_probability': 0.3,
    'birth_probability': 1.0,
    'gene_size': 7, # The bit size for each gene (parameter)
    'generations': 200,
    'protocol': 'FullReplacement',
    'mechanism': 'Tournament',
    'reproduction': 'BinaryOnePointCrossover',
    'elitism': 0.04,
    'truncation': 0.05,
    'tau': 10.0,
    'I': 10.0,
    'timesteps': 1000,
    'spike_threshold': 35, # mV (milli Volts)
    'num_params': 5, # The number of parameters: a, b, c, d, k
    'metric_fn': SDM.spike_time_distance_metric # The number of parameters: a, b, c, d, k
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
        dest="gene_size",
        type=int,
        default=std_values['gene_size'],
        help='Gene size. Number of bits in each gene')

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

    parser.add_argument(
        '-metric', 
        action="store", 
        dest="metric_fn",
        type=str, 
        default=std_values['metric_fn'],
        help='The metric distance function')

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
    output_size = args.pop_size # - int(args.pop_size * 0.05)

    print args


    target_spikes = get_target_data()
    target = target_spikes[1];

    create_objects = create_data(args.gene_size, args.metric_fn, target)
    population = Population(args.pop_size, create_objects)

    adult_selection = SelectionStrategy(output_size, str_to_class(args.protocol))
    parent_selection = SelectionStrategy(args.pop_size, None, str_to_class(args.mechanism), args.elitism, args.truncation)

    reproduction = str_to_class(args.reproduction)(args.birth_probability) # , 0.3) # Birth probability
    mutation = SpikeMutation(args.mutation_probability) # Mutation probability

    if args.output_file is not None and args.do_plot is True:
        plotter = SpikingNeuronPlotter("./plots", args.output_file)
    else:
        plotter = None

    ea = EA(population, adult_selection, parent_selection, reproduction, mutation, args.generations, plotter)
    ea.loop()

    if plotter is not None:
        plotter.plot()
        plotter.plot_act()
