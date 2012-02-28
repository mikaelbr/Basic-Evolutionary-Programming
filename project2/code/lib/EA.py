from Indevidual import *
import random
import copy

class EA:

    def __init__(self, population, adult_selection = None, parent_selection = None, reproduction = None, mutation = None, generations = 200, plotter = None):
        self.generations = generations;
        self.population = population
        self.adult_selection = adult_selection
        self.parent_selection = parent_selection
        self.reproduction = reproduction
        self.mutation = mutation
        self.plotter = plotter

    def development(self):
        """
            Converting the genotypes into phonotypes. 
        """
        new_population = []
        for item in self.population.children:
            if item.phenotype is None:
                item.toPhenotype()
            
            new_population.append(item)

        self.population.children = new_population;
        return self.population.children

    def test_fitness(self):
        """
            Run through entire population and calculate 
            fitness/strength.
        """
        self.population.test_fitness()
        

    def adult_select(self):
        if self.adult_selection is not None:
            self.adult_selection.select(self.population);
        else:
            this.population.adults = self.population.children[:]
        self.population.children = []

    def parent_select(self):
        if self.parent_selection is not None:
            self.parent_selection.select(self.population);
        

    def birth(self):
        if self.reproduction is None:
            return

        parents = self.population.parents[:]
        for i in range(0, len(parents)):
            if i % 2 == 1:
                child1, child2 = copy.deepcopy(parents[i-1]), copy.deepcopy(parents[i])

                if self.reproduction.birth_probability > random.random():
                    # print "Parents: \n %s \n %s" % (child1.value, child2.value)
                    child1, child2 = self.reproduction.do(child1, child2)
                    # print "Kids: \n %s \n %s" % (child1.value, child2.value)


                self.population.children.append(child1)
                self.population.children.append(child2)


    def mutate(self):
        if self.mutation is None:
            return

        for item in self.population.children:
            if self.mutation.mutation_probability > random.random():
                self.mutation.do(item)
        

    def loop(self):

        # initialize child genotype population
        self.population.fill() # Lots of childeren

        for generation in range(self.generations):
            
            # development: Genotypes -> Phenotypes
            self.development()

            # Test fitness of Phenotypes
            self.test_fitness()


            # Do plotting
            if self.plotter is not None:
                self.plotter.update(generation, self.population)

            # print "PRE ADULTS"
            print "\n\n\n------- GENERATION %s -------- \n Length: %s \n\n\n" % (generation, len(self.population.children))
            # for item in self.population.children[:]:
            #     fitness = item.fitness_value;

            #     if fitness is None:
            #         fitness = 0

            #     # Collect all denary values from genotype, categorized by 5 bits interval
            #     params = [int(item.value[i:i+item.gene_size], 2) for i in range(0, len(item.value), item.gene_size)]
            #     # Need to encode values to fit to given ranges/intervals
            #     a, b, c, d, k = item.fit_range (params)
            #     print "Value %s, %s, %s, %s, %s, fitness: %s" % (a, b, c, d, k, fitness)

            # Adult selection
            self.adult_select()

            # print "PRE ADULTS"
            # Parent selection
            self.parent_select()

            # Now the parents-attribute in the Population should
            # be filled and ready for reproduction.

            # Reproduction
            self.birth()

            self.mutate()

            # Intitiate Jean-Luc Picard; The Next Generation
        
        