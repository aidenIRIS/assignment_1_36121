# Hill Climbing for job search (template)
def hill_climbing(start, fitness_fn, get_neighbors, max_iterations=100, nodes_expanded=None):
    """
    start: initial candidate solution
    fitness_fn: function to evaluate fitness
    get_neighbors: function to get neighboring solutions for the *current* state
    max_iterations: maximum number of iterations
    nodes_expanded: optional dict counter {'count': int} for instrumentation
    """
    current = start
    for _ in range(max_iterations):
        neighbors = get_neighbors(current)
        if not neighbors:
            break
        # Count the evaluation of the current node
        if nodes_expanded is not None:
            nodes_expanded['count'] += 1
        # Evaluate neighbors
        neighbor = max(neighbors, key=fitness_fn)
        if nodes_expanded is not None:
            nodes_expanded['count'] += len(neighbors)
        if fitness_fn(neighbor) <= fitness_fn(current):
            break
        current = neighbor
    return current


def probabilistic_hill_climb(
    start,
    fitness_fn,
    get_neighbors,
    max_iterations=100,
    temperature=1.0,
    cooling=0.95,
    nodes_expanded=None,
):
    """
    Stochastic hill climbing / simulated annealing style.
    - Accepts worse moves with probability exp(-(delta)/T) to escape local maxima.
    - Temperature decays by `cooling` each iteration.
    """
    import math
    import random

    current = start
    current_f = fitness_fn(current)
    T = temperature
    for _ in range(max_iterations):
        neighbors = get_neighbors(current)
        if not neighbors:
            break
        if nodes_expanded is not None:
            nodes_expanded['count'] += 1 + len(neighbors)
        candidate = random.choice(neighbors)
        cand_f = fitness_fn(candidate)
        delta = cand_f - current_f
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-9)):
            current, current_f = candidate, cand_f
        T *= cooling
    return current
