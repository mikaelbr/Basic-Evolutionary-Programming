
class Indevidual(object):

    def __init__(self, value, fitness_func):
        self.value = value
        self.fitness_func = fitness_func
        self.fitness_value = None
        self.phenotype = None

    @property
    def phenotype(self):
        return self._phenotype


    @phenotype.setter
    def phenotype(self, new_phenotype):
        self._phenotype = new_phenotype

    @property
    def fitness_value(self):
        return self._fitness_value

    @fitness_value.setter
    def fitness_value(self, value):
        self._fitness_value = value

    def fitness(self):
        if self.fitness_value is None:
            self.fitness_value = self.fitness_func(self.phenotype)

        return self.fitness_value  