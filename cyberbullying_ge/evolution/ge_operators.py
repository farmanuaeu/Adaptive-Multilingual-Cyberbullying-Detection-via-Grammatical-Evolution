import random

def crossover(parent1, parent2):
    child = parent1.copy()
    pt = random.randint(1, len(parent1)-2)
    child[pt:] = parent2[pt:]
    return child

def mutate(individual, mutation_rate):
    return [gene ^ (1 << random.randint(0, 7)) if random.random() < mutation_rate else gene 
            for gene in individual]