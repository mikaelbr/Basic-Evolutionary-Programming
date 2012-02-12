import random

class Reproduction(object):
    """
        Genetic: Performed on the genotype
    """
    def __init__(self, birth_probability):
        self.birth_probability = birth_probability

    def do(self):
        pass


class BinaryUniformCrossover(Reproduction):

    def __init__(self, birth_probability, probability = 0.2):
        super(BinaryUniformCrossover, self).__init__(birth_probability)
        self.probability = probability

    def do(self, g1, g2):
        """
            Randomly decides which parent to copy bits from.
        """
        child1 = []
        child2 = []
        for i in range(len(g1.value)):
            if random.random() > self.probability:
                child1.append(g1.value[i])
                child2.append(g2.value[i])
            else:
                child1.append(g2.value[i])
                child2.append(g1.value[i])

        
        return g1.create_child(''.join(child1)), g2.create_child(''.join(child2))


class BinaryOnePointCrossover(Reproduction):

    def __init__(self, birth_probability, split = None):
        super(BinaryOnePointCrossover, self).__init__(birth_probability)
        self.split = split

    def do(self, g1, g2):
        """
            Creates two children. Split at point 'split'. Randomly generated
            if no split point given. 
        """
        split_in = self.split
        if split_in is None:
            split_in = random.randint(1, len(g1.value)-1)           
        
        child1 = g1.create_child(g1.value[:split_in] + g2.value[split_in:])
        child2 = g2.create_child(g2.value[:split_in] + g1.value[split_in:])
        return child1, child2

class BinaryTwoPointCrossover(Reproduction):

    def __init__(self, birth_probability, split1 = None, split2 = None):
        super(BinaryTwoPointCrossover, self).__init__(birth_probability)
        self.split = split

    def do(self, g1, g2):
        """
            Creates two children. Split twice. Randomly generated
            if no split point given. 
        """
        split_in1, split_in2 = split1, split2
        if split_in1 is None:
            split_in1 = random.randint(1, len(g1.value)-1)
        
        if split_in2 is None:
            split_in2 = random.randint(1, len(g1.value)-1)
        
        if split_in1 > split_in2:
            # Switch up
            split_in1, split_in2 = split_in2, split_in1
        
        child1 = g1.create_child(g1.value[:split_in1] + g2.value[split_in1:split_in2] + g1.value[split_in2:])
        child2 = g1.create_child(g2.value[:split_in1] + g1.value[split_in1:split_in2] + g2.value[split_in2:])
        return child1, child2
