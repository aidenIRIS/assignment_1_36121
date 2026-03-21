# Depth-First Search (DFS) implementation for job search
def dfs_search(graph, start, is_goal, get_neighbors):
    """
    graph: data structure representing the job dataset (e.g., adjacency list or dict)
    start: starting node (job or skill)
    is_goal: function to check if a node is a goal (AI-related job)
    get_neighbors: function to get neighbors of a node
    """
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if is_goal(node):
            return node
        stack.extend(get_neighbors(node))
    return None
