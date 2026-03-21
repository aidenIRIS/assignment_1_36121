# Breadth-First Search (BFS) implementation for job search
from collections import deque

def bfs_search(graph, start, is_goal, get_neighbors):
    """
    graph: data structure representing the job dataset (e.g., adjacency list or dict)
    start: starting node (job or skill)
    is_goal: function to check if a node is a goal (AI-related job)
    get_neighbors: function to get neighbors of a node
    """
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        if is_goal(node):
            return node
        queue.extend(get_neighbors(node))
    return None
