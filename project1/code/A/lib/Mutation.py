import random

class Mutation(object):
    """
        Genetic: Performed on the genotype
    """
    def __init__(self, mutation_probability):
        self.mutation_probability = mutation_probability

    def do(self):
        pass

class BinaryStringInversion(Mutation):

    def __init__(self, mutation_probability, bits = 1):
        super(BinaryStringInversion, self).__init__(mutation_probability)
        self.bits = bits

    def do(self, g1):

        bit_string = random.sample(range(len(g1.value)), self.bits)

        new_string = list(g1.value)
        for i in bit_string:
            new_string[i] = '1' if new_string[i] == '0' else '0'
        g1.value = ''.join(new_string)

