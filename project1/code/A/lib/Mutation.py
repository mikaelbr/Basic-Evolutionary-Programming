

class Mutation(object):
    """
        Genetic: Performed on the genotype
    """
    def __init__(self, mutation_probability):
        self.mutation_probability = mutation_probability

    def do(self):
        pass

    



class BinaryStringInversion(Mutation):

    def __init__(self, mutation_probability):
        super(BinaryStringInversion, self).__init__(mutation_probability)

    def do(self, g1):
        g1.value = (''.join('1' if x == '0' else '0' for x in g1.value))

