
class Population(object):

    def __init__(self, size, creation_closure):
        self.pop_size = size
        self.children = []
        self.adults = []
        self.parents = []
        self.elitsm = []
        self.creation_closure = creation_closure

    def fill(self):
        for i in range(self.pop_size):
            self.children.append(self.creation_closure())

        return self.children;

    def get_total_population(self):
        return self.children + self.adults

    def kill_adults(self):
        self.adults = []

    def remove_children(self):
        self.children = []

    def test_fitness(self):
        for i in self.get_total_population():
            i.fitness()

