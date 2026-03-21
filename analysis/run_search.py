# Example script to run all search algorithms on the job dataset
import pandas as pd
from src.data_prep import load_dataset, preprocess_data
from src.algorithms import bfs, dfs, ucs, genetic_algorithm, hill_climbing, utils


# Load and preprocess the test dataset
DATA_PATH = 'data/Sample Datasets-2 data job posts TEST DATA.csv'
df = load_dataset(DATA_PATH)
df = preprocess_data(df)

# Normalize column names for easier access
title_col = 'title' if 'title' in df.columns else df.columns[0]
desc_col = 'jobdescription' if 'jobdescription' in df.columns else df.columns[-1]

# Filter for AI/IT-related jobs using utils.is_ai_related or IT column if present
if 'it' in df.columns:
    ai_jobs = df[df['it'] == True]
else:
    ai_jobs = df[df.apply(lambda row: utils.is_ai_related(str(row.get(title_col, '')), str(row.get(desc_col, ''))), axis=1)]

jobs = ai_jobs.to_dict(orient='records')

# Example skill keywords for ranking
skill_keywords = ['python', 'tensorflow', 'pytorch', 'nlp', 'data science', 'machine learning', 'ai', 'artificial intelligence']

# Rank jobs by skill keywords
ranked_jobs = utils.rank_jobs(jobs, skill_keywords)

print(f"Total jobs: {len(df)}")
print(f"AI/IT-related jobs: {len(jobs)}")
print("Top 5 ranked AI/IT jobs:")
import math
for job in ranked_jobs[:5]:
    desc = job.get('jobdescription', '')
    if desc is None or (isinstance(desc, float) and math.isnan(desc)):
        desc = ''
    print(f"- {job.get('title', '')} | {job.get('company', '')}\n  {desc[:120]}...")

# Example: Run BFS, DFS, UCS on the jobs (treating jobs as nodes, neighbors as next job in list for demo)
def is_goal(job):
    return utils.is_ai_related(str(job.get('title', '')), str(job.get('jobdescription', '')))

def get_neighbors(idx):
    # For demo, neighbors are next 1-2 jobs
    return [i for i in range(idx+1, min(idx+3, len(jobs)))]

print("\nBFS/DFS/UCS demo (first AI/IT job found):")
if jobs:
    # BFS
    bfs_result = bfs.bfs_search(list(range(len(jobs))), 0, lambda i: is_goal(jobs[i]), get_neighbors)
    if bfs_result is not None:
        print("BFS found:", jobs[bfs_result].get('title', ''))
    # DFS
    dfs_result = dfs.dfs_search(list(range(len(jobs))), 0, lambda i: is_goal(jobs[i]), get_neighbors)
    if dfs_result is not None:
        print("DFS found:", jobs[dfs_result].get('title', ''))
    # UCS (cost = 1 per step)
    ucs_result, ucs_cost = ucs.ucs_search(list(range(len(jobs))), 0, lambda i: is_goal(jobs[i]), get_neighbors, lambda a, b: 1)
    if ucs_result is not None:
        print("UCS found:", jobs[ucs_result].get('title', ''), f"(cost: {ucs_cost})")
else:
    print("No AI/IT jobs found for search algorithms.")

# Genetic Algorithm and Hill Climbing demo (randomly select job with most skills)
def fitness_fn(job):
    text = f"{job.get('title', '')} {job.get('jobdescription', '')}".lower()
    return sum(kw in text for kw in skill_keywords)

if jobs:
    # Genetic Algorithm demo
    best_ga = genetic_algorithm.genetic_algorithm(jobs, fitness_fn, lambda x: x, lambda a, b: a, lambda pop, scores: pop, generations=1)
    print("\nGenetic Algorithm best match:", best_ga.get('title', ''))
    # Hill Climbing demo
    best_hc = hill_climbing.hill_climbing(jobs[0], fitness_fn, lambda job: jobs, max_iterations=1)
    print("Hill Climbing best match:", best_hc.get('title', ''))

print("\nAnalysis complete. Extend this script for more detailed metrics and visualizations.")
