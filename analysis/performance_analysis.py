# Performance Analysis Template for Search Algorithms
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
from src.algorithms import astar, bfs, dfs, ucs, genetic_algorithm, hill_climbing, utils

# --- Helper Functions ---
def measure_performance(algorithm_fn, *args, include_nodes=True, **kwargs):
    """Measure runtime, memory, and optionally nodes expanded for a search algorithm."""
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

# --- Load Data ---
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

# --- Example Skill Keywords ---
skill_keywords = ['python', 'tensorflow', 'pytorch', 'nlp', 'data science', 'machine learning', 'ai', 'artificial intelligence']

def job_text(job):
    """Concatenate key textual fields for scoring/similarity."""
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

def fitness_fn(job):
    text = job_text(job)
    return sum(kw in text for kw in skill_keywords)

def build_similarity_neighbors(jobs, top_k=5):
    """Precompute top_k similar job indices using TF-IDF cosine over title+description."""
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

# --- Performance Results ---
results = []

# BFS
if jobs:
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
    _, runtime, mem, nodes = measure_performance(
        astar.astar_search, list(range(len(jobs))), 0, lambda i: is_goal(jobs[i]), get_neighbors,
        heuristic_fn=lambda i: fitness_fn(jobs[i]), get_cost=lambda a, b: 1
    )
    results.append({'algorithm': 'A*', 'runtime_sec': runtime, 'memory_mb': mem, 'nodes_expanded': nodes})

    # Genetic Algorithm (with metrics)
    def mutate_fn(x):
        # Example: no mutation (identity)
        return x
    def crossover_fn(a, b):
        # Example: return a (no crossover)
        return a
    def select_fn(pop, scores):
        # Example: return population as is
        return pop

    def run_ga():
        best, metrics = genetic_algorithm.genetic_algorithm(
            jobs, fitness_fn, mutate_fn, crossover_fn, select_fn, generations=10, log_metrics=True
        )
        return best, metrics

    ga_result, runtime, mem, _ = measure_performance(run_ga, include_nodes=False)
    best, ga_metrics = ga_result
    results.append({
        'algorithm': 'Genetic Algorithm',
        'runtime_sec': runtime,
        'memory_mb': mem,
        'nodes_expanded': None,
        'encoding': ga_metrics['encoding'],
        'selection_strategy': ga_metrics['selection_strategy'],
        'crossover_operator': ga_metrics['crossover_operator'],
        'mutation_operator': ga_metrics['mutation_operator'],
        'convergence_generation': ga_metrics['convergence_generation'],
        'best_fitness': ga_metrics['best_fitness'],
        'crossovers_total': ga_metrics['crossovers_total'],
        'mutations_total': ga_metrics['mutations_total'],
    })

    # Plot convergence behaviour (fitness over generations)
    if not SKIP_PLOTS:
        fitness_hist = ga_metrics['fitness_history']
        generations = [f['generation'] for f in fitness_hist]
        bests = [f['best'] for f in fitness_hist]
        avgs = [f['avg'] for f in fitness_hist]
        worsts = [f['worst'] for f in fitness_hist]
        plt.figure()
        plt.plot(generations, bests, label='Best Fitness')
        plt.plot(generations, avgs, label='Average Fitness')
        plt.plot(generations, worsts, label='Worst Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Genetic Algorithm Convergence')
        plt.legend()
        plt.tight_layout()

    # Hill Climbing with similarity-based neighbors and random restarts (deterministic)
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
            cooling=0.90,
        )
        score = fitness_fn(result)
        if score > best_score:
            best_score = score
            best_run = (runtime, mem, nodes)
    runtime, mem, nodes = best_run
    results.append({'algorithm': 'Prob Hill Climb', 'runtime_sec': runtime, 'memory_mb': mem, 'nodes_expanded': nodes})

# --- Results DataFrame ---
df_results = pd.DataFrame(results)
print(df_results)

# --- Visualization (saved to disk) ---
plots_dir = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(plots_dir, exist_ok=True)

if not SKIP_PLOTS and not df_results.empty:
    sns.barplot(data=df_results, x='algorithm', y='runtime_sec')
    plt.title('Algorithm Runtime Comparison')
    plt.ylabel('Runtime (seconds)')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'runtime_comparison.png'), dpi=200)
    plt.clf()

    sns.barplot(data=df_results, x='algorithm', y='memory_mb')
    plt.title('Algorithm Memory Usage Comparison')
    plt.ylabel('Peak Memory (MB)')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'memory_comparison.png'), dpi=200)
    plt.clf()

    nodes_df = df_results.dropna(subset=['nodes_expanded'])
    if not nodes_df.empty:
        sns.barplot(data=nodes_df, x='algorithm', y='nodes_expanded')
        plt.title('Algorithm Nodes Expanded')
        plt.ylabel('Nodes Expanded')
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'nodes_expanded.png'), dpi=200)
        plt.clf()

df_results.to_csv(os.path.join(OUTPUT_DIR, 'performance_metrics.csv'), index=False)
