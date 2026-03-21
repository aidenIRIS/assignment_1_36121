# Uniform Cost Search (UCS) implementation for job search
import heapq

def ucs_search(graph, start, is_goal, get_neighbors, get_cost):
    """
    graph: data structure representing the job dataset (e.g., adjacency list or dict)
    start: starting node (job or skill)
    is_goal: function to check if a node is a goal (AI-related job)
    get_neighbors: function to get neighbors of a node
    get_cost: function to get cost between nodes
    """
    visited = set()
    queue = [(0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if is_goal(node):
            return node, cost
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                heapq.heappush(queue, (cost + get_cost(node, neighbor), neighbor))
    return None, float('inf')
