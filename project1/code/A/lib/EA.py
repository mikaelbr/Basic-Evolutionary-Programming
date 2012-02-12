from Indevidual import *
import random

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

        # Rewrite to take account for the adults and childeren. 
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
        for i in self.population.get_total_population():
            i.fitness()
        

    def adult_select(self):
        if self.adult_selection is not None:
            self.adult_selection.select(self.population);
        else:
            this.population.adults = this.population.children[:]
            this.population.children = []

    def parent_select(self):
        if self.parent_selection is not None:
            self.parent_selection.select(self.population);
        

    def birth(self):
        if self.reproduction is None:
            return
        
        parents = self.population.parents
        for i in range(0, len(parents)-1, 2):
            child1, child2 = parents[i], parents[i+1]

            if self.reproduction.birth_probability > random.random():
                child1, child2 = self.reproduction.do(child1, child2)

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
            self.plotter.update(generation, self.population)

            # Adult selection
            self.adult_select()

            # Parent selection
            self.parent_select()

            # Now the parents-attribute in the Population should
            # be filled and ready for reproduction.

            # Reproduction
            self.birth()
            self.mutate()

            # Intitiate Jean-Luc Picard; The Next Generation
        
        