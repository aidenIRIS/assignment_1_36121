# Genetic Algorithm for job search (enhanced for metrics)
import random

# --- Encoding Description ---
# For this template, each individual is a job dictionary (can be adapted for other encodings)
ENCODING = 'Job dictionary (can be adapted to binary, feature vector, etc.)'

def genetic_algorithm(
    population,
    fitness_fn,
    mutate_fn,
    crossover_fn,
    select_fn,
    generations=100,
    log_metrics=False,
    population_cap=None,
    elitism_rate=0.05,
    crossover_rate=1.0,
    mutation_rate=0.1,
):
    """
    population: initial population (list of candidate solutions)
    fitness_fn: function to evaluate fitness of a candidate
    mutate_fn: function to mutate a candidate
    crossover_fn: function to crossover two candidates
    select_fn: function to select candidates for next generation
    generations: number of generations to run
    log_metrics: if True, returns (best, metrics_dict)
    ---
    Metrics captured:
      - encoding: ENCODING
      - selection_strategy: select_fn.__name__ or description
      - crossover_operator: crossover_fn.__name__ or description
      - mutation_operator: mutate_fn.__name__ or description
      - fitness_history: list of (best, avg, worst) fitness per generation
      - convergence_generation: generation where best fitness first achieved
    """
    metrics = {
        'encoding': ENCODING,
        'selection_strategy': getattr(select_fn, '__name__', str(select_fn)),
        'crossover_operator': getattr(crossover_fn, '__name__', str(crossover_fn)),
        'mutation_operator': getattr(mutate_fn, '__name__', str(mutate_fn)),
        'fitness_history': [],
        'convergence_generation': None,
        'best_fitness': None,
        'generations': generations,
        'crossovers_total': 0,
        'mutations_total': 0,
    }
    # Optional population cap: keep fittest individuals to cap size
    if population_cap is not None and len(population) > population_cap:
        population = sorted(population, key=fitness_fn, reverse=True)[:population_cap]
    best_fitness = None
    best_individual = None
    convergence_gen = None
    for gen in range(generations):
        fitness_scores = [fitness_fn(ind) for ind in population]
        best = max(fitness_scores)
        avg = sum(fitness_scores) / len(fitness_scores)
        worst = min(fitness_scores)
        metrics['fitness_history'].append({'generation': gen, 'best': best, 'avg': avg, 'worst': worst})
        if best_fitness is None or best > best_fitness:
            best_fitness = best
            best_individual = population[fitness_scores.index(best)]
            convergence_gen = gen
        next_gen = select_fn(population, fitness_scores)
        # Elitism: keep top-k
        elite_count = max(1, int(elitism_rate * len(population))) if elitism_rate > 0 else 0
        elite_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
        elites = [population[i] for i in elite_indices]
        offspring = []
        crossovers = 0
        mutations = 0
        while len(offspring) < len(population):
            parents = random.sample(next_gen, 2)
            if random.random() < crossover_rate:
                child = crossover_fn(parents[0], parents[1])
                crossovers += 1
            else:
                child = random.choice(parents)
            if random.random() < mutation_rate:
                child = mutate_fn(child)
                mutations += 1
            offspring.append(child)
        population = elites + offspring[: max(len(population) - len(elites), 0)]
        metrics['crossovers_total'] += crossovers
        metrics['mutations_total'] += mutations
    metrics['convergence_generation'] = convergence_gen
    metrics['best_fitness'] = best_fitness
    # Elitism: ensure best-so-far is returned even if final generation regresses
    best_individual = best_individual if best_individual is not None else max(population, key=fitness_fn)
    if log_metrics:
        return best_individual, metrics
    return best_individual
