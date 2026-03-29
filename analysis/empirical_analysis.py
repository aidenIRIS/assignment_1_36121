# Empirical Analysis Script for Search Algorithms
import os
import time
import tracemalloc

import matplotlib

# Use non-interactive backend to avoid blocking in batch/headless runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data_prep import load_dataset, preprocess_data
from src.algorithms import bfs, dfs, ucs, genetic_algorithm, hill_climbing, utils

# Placeholder for A* implementation
# You must implement astar_search in src/algorithms/astar.py for this to work
try:
    from src.algorithms import astar
    has_astar = True
except ImportError:
    has_astar = False

def measure_performance(algorithm_fn, *args, include_nodes=True, **kwargs):
    tracemalloc.start()
    start_time = time.perf_counter()
    nodes_expanded = {'count': 0} if include_nodes else None
    if include_nodes:
        kwargs['nodes_expanded'] = nodes_expanded
    result = algorithm_fn(*args, **kwargs)
    runtime = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    nodes = nodes_expanded['count'] if nodes_expanded is not None else None
    return result, runtime, peak / 1024 / 1024, nodes

# Load and preprocess data
DATA_PATH = os.environ.get(
    "ASSIGNMENT_DATA_PATH", "data/Sample Datasets-2 data job posts TEST DATA.csv"
)
OUTPUT_DIR = os.environ.get("ASSIGNMENT_OUTPUT_DIR", "analysis")
SKIP_PLOTS = os.environ.get("ASSIGNMENT_SKIP_PLOTS", "0") == "1"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Expected dataset at {DATA_PATH}. Add the file or update DATA_PATH.")
df = load_dataset(DATA_PATH)
df = preprocess_data(df)
title_col = 'title' if 'title' in df.columns else df.columns[0]
desc_col = 'jobdescription' if 'jobdescription' in df.columns else df.columns[-1]
if 'it' in df.columns:
    ai_jobs = df[df['it'] == True]
else:
    ai_jobs = df[df.apply(lambda row: utils.is_ai_related(str(row.get(title_col, '')), str(row.get(desc_col, ''))), axis=1)]
jobs = ai_jobs.to_dict(orient='records')

skill_keywords = ['python', 'tensorflow', 'pytorch', 'nlp', 'data science', 'machine learning', 'ai', 'artificial intelligence']

def job_text(job):
    if job.get("clean_text"):
        return str(job.get("clean_text")).lower()
    requirement = (
        job.get("jobrequirement")
        or job.get("jobrequirment")
        or job.get("jobrequiment")
        or ""
    )
    parts = [
        job.get('title', ''),
        job.get('jobdescription', ''),
        requirement,
        job.get('requiredqual', ''),
    ]
    parts = [p if isinstance(p, str) else '' for p in parts]
    return " ".join(parts).lower()

def is_goal(job):
    return utils.is_ai_related(str(job.get('title', '')), str(job.get('jobdescription', '')))

def build_similarity_neighbors(jobs, top_k=5):
    if not jobs:
        return []
    corpus = [job_text(j) for j in jobs]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    X = vectorizer.fit_transform(corpus)
    sim = cosine_similarity(X)
    neighbors = []
    for i in range(len(jobs)):
        scores = list(enumerate(sim[i]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        neigh = [j for j, _ in scores if j != i][:top_k]
        neighbors.append(neigh)
    return neighbors

similarity_neighbors = build_similarity_neighbors(jobs, top_k=5)

def get_neighbors(idx):
    return similarity_neighbors[idx] if idx < len(similarity_neighbors) else []

def fitness_fn(job):
    text = job_text(job)
    return sum(kw in text for kw in skill_keywords)

def heuristic_fn(idx):
    # Simple heuristic: number of skill keywords in job title/desc (for demo)
    return fitness_fn(jobs[idx])

results = []

if jobs:
    # BFS
    _, runtime, mem, nodes = measure_performance(
        bfs.bfs_search, list(range(len(jobs))), 0, lambda i: is_goal(jobs[i]), get_neighbors
    )
    results.append({'algorithm': 'BFS', 'runtime_sec': runtime, 'memory_mb': mem, 'nodes_expanded': nodes})

    # DFS
    _, runtime, mem, nodes = measure_performance(
        dfs.dfs_search, list(range(len(jobs))), 0, lambda i: is_goal(jobs[i]), get_neighbors
    )
    results.append({'algorithm': 'DFS', 'runtime_sec': runtime, 'memory_mb': mem, 'nodes_expanded': nodes})

    # UCS
    _, runtime, mem, nodes = measure_performance(
        ucs.ucs_search, list(range(len(jobs))), 0, lambda i: is_goal(jobs[i]), get_neighbors, lambda a, b: 1
    )
    results.append({'algorithm': 'UCS', 'runtime_sec': runtime, 'memory_mb': mem, 'nodes_expanded': nodes})

    # A*
    if has_astar:
        _, runtime, mem, nodes = measure_performance(
            astar.astar_search, list(range(len(jobs))), 0, lambda i: is_goal(jobs[i]), get_neighbors, heuristic_fn,
            get_cost=lambda a, b: 1
        )
        results.append({'algorithm': 'A*', 'runtime_sec': runtime, 'memory_mb': mem, 'nodes_expanded': nodes})
    else:
        results.append({'algorithm': 'A*', 'runtime_sec': None, 'memory_mb': None, 'nodes_expanded': None})

    # Genetic Algorithm
    def mutate_fn(x):
        return x
    def crossover_fn(a, b):
        return a
    def select_fn(pop, scores):
        return pop
    def run_ga():
        best, metrics = genetic_algorithm.genetic_algorithm(
            jobs, fitness_fn, mutate_fn, crossover_fn, select_fn, generations=10, log_metrics=True
        )
        return best, metrics
    _, runtime, mem, _ = measure_performance(run_ga, include_nodes=False)
    results.append({'algorithm': 'Genetic Algorithm', 'runtime_sec': runtime, 'memory_mb': mem, 'nodes_expanded': None})

    import random
    def hc_neighbors(job):
        idx = jobs.index(job)
        return [jobs[i] for i in get_neighbors(idx)]

    best_score = -1
    best_run = None
    for start_idx in random.sample(range(len(jobs)), min(3, len(jobs))):
        result, runtime, mem, nodes = measure_performance(
            hill_climbing.hill_climbing, jobs[start_idx], fitness_fn, hc_neighbors, max_iterations=10
        )
        score = fitness_fn(result)
        if score > best_score:
            best_score = score
            best_run = (runtime, mem, nodes)
    runtime, mem, nodes = best_run
    results.append({'algorithm': 'Hill Climbing', 'runtime_sec': runtime, 'memory_mb': mem, 'nodes_expanded': nodes})

    # Probabilistic Hill Climbing (simulated annealing style) with restarts
    best_score = -1
    best_run = None
    for start_idx in random.sample(range(len(jobs)), min(3, len(jobs))):
        result, runtime, mem, nodes = measure_performance(
            hill_climbing.probabilistic_hill_climb,
            jobs[start_idx],
            fitness_fn,
            hc_neighbors,
            max_iterations=15,
            temperature=1.0,
            cooling=0.90
        )
        score = fitness_fn(result)
        if score > best_score:
            best_score = score
            best_run = (runtime, mem, nodes)
    runtime, mem, nodes = best_run
    results.append({'algorithm': 'Prob Hill Climb', 'runtime_sec': runtime, 'memory_mb': mem, 'nodes_expanded': nodes})

# Results DataFrame
empirical_df = pd.DataFrame(results)
print(empirical_df)

# Visualization (save to disk)
plots_dir = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(plots_dir, exist_ok=True)

if not SKIP_PLOTS and not empirical_df.empty:
    sns.barplot(data=empirical_df, x='algorithm', y='runtime_sec')
    plt.title('Algorithm Runtime Comparison (Empirical)')
    plt.ylabel('Runtime (seconds)')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'empirical_runtime_comparison.png'), dpi=200)
    plt.clf()

    sns.barplot(data=empirical_df, x='algorithm', y='memory_mb')
    plt.title('Algorithm Memory Usage Comparison (Empirical)')
    plt.ylabel('Peak Memory (MB)')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'empirical_memory_comparison.png'), dpi=200)
    plt.clf()

    nodes_df = empirical_df.dropna(subset=['nodes_expanded'])
    if not nodes_df.empty:
        sns.barplot(data=nodes_df, x='algorithm', y='nodes_expanded')
        plt.title('Algorithm Nodes Expanded (Empirical)')
        plt.ylabel('Nodes Expanded')
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'empirical_nodes_expanded.png'), dpi=200)
        plt.clf()

# Save table to CSV
empirical_df.to_csv(os.path.join(OUTPUT_DIR, 'empirical_algorithm_metrics.csv'), index=False)
