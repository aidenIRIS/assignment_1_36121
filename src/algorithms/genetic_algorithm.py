# Genetic Algorithm for job search (template)
import random

def genetic_algorithm(population, fitness_fn, mutate_fn, crossover_fn, select_fn, generations=100):
    """
    population: initial population (list of candidate solutions)
    fitness_fn: function to evaluate fitness of a candidate
    mutate_fn: function to mutate a candidate
    crossover_fn: function to crossover two candidates
    select_fn: function to select candidates for next generation
    generations: number of generations to run
    """
    for _ in range(generations):
        fitness_scores = [fitness_fn(ind) for ind in population]
        next_gen = select_fn(population, fitness_scores)
        offspring = []
        while len(offspring) < len(population):
            parents = random.sample(next_gen, 2)
            child = crossover_fn(parents[0], parents[1])
            child = mutate_fn(child)
            offspring.append(child)
        population = offspring
    best = max(population, key=fitness_fn)
    return best
