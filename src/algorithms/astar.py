# A* Search implementation for job search
import heapq

def astar_search(graph, start, is_goal, get_neighbors, heuristic_fn, get_cost=None, nodes_expanded=None):
    """
    graph: data structure representing the job dataset (e.g., adjacency list or dict)
    start: starting node (job or skill)
    is_goal: function to check if a node is a goal (AI-related job)
    get_neighbors: function to get neighbors of a node
    heuristic_fn: function to estimate cost from node to goal
    get_cost: optional cost function (defaults to 1 per step)
    nodes_expanded: optional dict counter {'count': int} for instrumentation
    """
    if get_cost is None:
        get_cost = lambda a, b: 1
    visited = set()
    queue = [(heuristic_fn(start), 0, start)]  # (f = g + h, g, node)
    while queue:
        f, g, node = heapq.heappop(queue)
        if nodes_expanded is not None:
            nodes_expanded['count'] += 1
        if node in visited:
            continue
        visited.add(node)
        if is_goal(node):
            return node, g
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                g_new = g + get_cost(node, neighbor)
                f_new = g_new + heuristic_fn(neighbor)
                heapq.heappush(queue, (f_new, g_new, neighbor))
    return None, float('inf')
