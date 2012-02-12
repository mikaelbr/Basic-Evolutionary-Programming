from Indevidual import *
from Phenotype import *
import random

class Genotype(Indevidual):

	def __init__(self, value, fitness_func, pheno_func):
		super(Genotype, self).__init__(value, fitness_func)
		self.pheno_func = pheno_func



	
