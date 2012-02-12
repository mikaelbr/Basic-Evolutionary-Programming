from Indevidual import *


class Phenotype(Indevidual):

	def __init__(self, value, fitness_func):
		super(Phenotype, self).__init__(value, fitness_func)

	def fitness(self):
		if self.fitness_value is None:
			self.fitness_value = self.fitness_func(self.value)

		return self.fitness_value