# Hill Climbing for job search (template)
def hill_climbing(start, fitness_fn, get_neighbors, max_iterations=100):
    """
    start: initial candidate solution
    fitness_fn: function to evaluate fitness
    get_neighbors: function to get neighboring solutions
    max_iterations: maximum number of iterations
    """
    current = start
    for _ in range(max_iterations):
        neighbors = get_neighbors(current)
        if not neighbors:
            break
        neighbor = max(neighbors, key=fitness_fn)
        if fitness_fn(neighbor) <= fitness_fn(current):
            break
        current = neighbor
    return current
