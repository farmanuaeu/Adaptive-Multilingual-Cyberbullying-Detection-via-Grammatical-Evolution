import random
from config import grammar

class GEIndividual:
    def __init__(self, genotype=None):
        self.genotype = genotype or self._random_genotype()
        self.fitness = -float('inf')
        self.config = self.decode()
        
    def _random_genotype(self):
        return [random.randint(0, 100) for _ in range(5)]
    
    def decode(self):
        return {
            'model_type': grammar.GRAMMAR['model_type'][self.genotype[0] % len(grammar.GRAMMAR['model_type'])],
            'layers': grammar.GRAMMAR['layers'][self.genotype[1] % len(grammar.GRAMMAR['layers'])],
            'hidden_size': grammar.GRAMMAR['hidden_size'][self.genotype[2] % len(grammar.GRAMMAR['hidden_size'])],
            'activation': grammar.GRAMMAR['activation'][self.genotype[3] % len(grammar.GRAMMAR['activation'])],
            'dropout': grammar.GRAMMAR['dropout'][self.genotype[4] % len(grammar.GRAMMAR['dropout'])]
        }