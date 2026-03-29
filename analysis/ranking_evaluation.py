# Ranking evaluation for search algorithms using simple relevance scores.
# Relevance = count of skill_keywords in title+jobdescription.
# Metrics: Precision@K, nDCG@K against ideal ranking (sorted by relevance).

import os
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_prep import load_dataset, preprocess_data
from src.algorithms import utils, genetic_algorithm, hill_climbing
from src.algorithms import astar, bfs, dfs, ucs

# --- Config ---
DATA_PATH = os.environ.get(
    "ASSIGNMENT_DATA_PATH", "data/Sample Datasets-2 data job posts TEST DATA.csv"
)
K_VALUES = (5, 10)
skill_keywords = [
    "python",
    "tensorflow",
    "pytorch",
    "nlp",
    "data science",
    "machine learning",
    "ai",
    "artificial intelligence",
]


# --- Helpers ---
def precision_at_k(ranked_indices, relevance, k):
    if not ranked_indices:
        return 0.0
    top = ranked_indices[:k]
    rel_hits = sum(1 for idx in top if relevance[idx] > 0)
    return rel_hits / k


def dcg(scores):
    return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(ranked_indices, relevance, k):
    if not ranked_indices:
        return 0.0
    ideal = sorted(relevance, reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    ranked = [relevance[idx] for idx in ranked_indices[:k]]
    return dcg(ranked) / ideal_dcg


def load_jobs():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = preprocess_data(load_dataset(DATA_PATH))
    title_col = "title" if "title" in df.columns else df.columns[0]
    desc_col = "jobdescription" if "jobdescription" in df.columns else df.columns[-1]
    if "it" in df.columns:
        ai_jobs = df[df["it"] == True]
    else:
        ai_jobs = df[
            df.apply(
                lambda row: utils.is_ai_related(
                    str(row.get(title_col, "")), str(row.get(desc_col, ""))
                ),
                axis=1,
            )
        ]
    jobs = ai_jobs.to_dict(orient="records")
    return jobs


def relevance_scores(jobs):
    scores = []
    for job in jobs:
        parts = [
            job.get('title',''),
            job.get('jobdescription',''),
            job.get('jobrequiment',''),
            job.get('requiredqual',''),
        ]
        parts = [p if isinstance(p, str) else '' for p in parts]
        text = " ".join(parts).lower()
        scores.append(sum(kw in text for kw in skill_keywords))
    return scores


def build_similarity_neighbors(jobs, top_k=5):
    if not jobs:
        return []
    corpus = []
    for j in jobs:
        parts = [
            j.get('title',''),
            j.get('jobdescription',''),
            j.get('jobrequiment',''),
            j.get('requiredqual',''),
        ]
        parts = [p if isinstance(p, str) else '' for p in parts]
        corpus.append(" ".join(parts).lower())
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


def run_bfs(jobs, neighbors):
    visited = set()
    order = []
    queue = [0] if jobs else []
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        queue.extend(neighbors[node])
    return order


def run_dfs(jobs, neighbors):
    visited = set()
    order = []
    stack = [0] if jobs else []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        stack.extend(reversed(neighbors[node]))
    return order


def run_ucs(jobs, neighbors):
    import heapq

    visited = set()
    order = []
    pq = [(0, 0)] if jobs else []
    while pq:
        cost, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for nbr in neighbors[node]:
            if nbr not in visited:
                heapq.heappush(pq, (cost + 1, nbr))
    return order


def taxonomy_heuristic(jobs):
    terms = (
        utils.TIER1_TITLE_TERMS
        + utils.TIER2_SKILLS_TECH
        + utils.TIER3_METHODS_CONCEPTS
        + utils.TIER4_ADJACENT
    )
    max_terms = len(terms)
    weights = {
        "title": 0.30,
        "jobdescription": 0.30,
        "jobrequirment": 0.25,
        "requiredqual": 0.15,
    }
    requirement_keys = ("jobrequirment", "jobrequiment", "jobrequirement")

    def h(idx):
        job = jobs[idx]
        title_text = str(job.get("title", "") or "").lower()
        desc_text = str(job.get("jobdescription", "") or "").lower()
        req_text = str(
            next((job.get(k) for k in requirement_keys if job.get(k)), "") or ""
        ).lower()
        qual_text = str(job.get("requiredqual", "") or "").lower()

        # Weighted field relevance:
        # f(j) = 0.30*Title + 0.30*JobDescription + 0.25*JobRequirement + 0.15*RequiredQual
        title_score = sum(term in title_text for term in terms) / max_terms
        desc_score = sum(term in desc_text for term in terms) / max_terms
        req_score = sum(term in req_text for term in terms) / max_terms
        qual_score = sum(term in qual_text for term in terms) / max_terms

        f_j = (
            weights["title"] * title_score
            + weights["jobdescription"] * desc_score
            + weights["jobrequirment"] * req_score
            + weights["requiredqual"] * qual_score
        )
        return 1.0 - f_j  # lower h is better for min-heap A*

    return h


def run_astar(jobs, neighbors):
    h = taxonomy_heuristic(jobs)
    visited = set()
    order = []
    import heapq

    openq = [(h(0), 0, 0)] if jobs else []
    while openq:
        f, g, node = heapq.heappop(openq)
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for nbr in neighbors[node]:
            if nbr not in visited:
                g_new = g + 1
                f_new = g_new + h(nbr)
                heapq.heappush(openq, (f_new, g_new, nbr))
    return order


def run_hill_climb(jobs, neighbors):
    rel = relevance_scores(jobs)

    # track path (indices)
    path = []
    def fitness(job):
        idx = jobs.index(job)
        return rel[idx]

    current_idx = 0 if jobs else None
    for _ in range(5):
        if current_idx is None:
            break
        path.append(current_idx)
        neigh_indices = neighbors[current_idx]
        if not neigh_indices:
            break
        best_idx = max(neigh_indices, key=lambda i: fitness(jobs[i]))
        if fitness(jobs[best_idx]) <= fitness(jobs[current_idx]):
            break
        current_idx = best_idx
    return path


def run_ga(jobs):
    if not jobs:
        return []
    def fitness(job):
        parts = [
            job.get('title',''),
            job.get('jobdescription',''),
            job.get('jobrequiment',''),
            job.get('requiredqual',''),
        ]
        parts = [p if isinstance(p, str) else '' for p in parts]
        text = " ".join(parts).lower()
        return sum(kw in text for kw in skill_keywords)
    def mutate_fn(x): return x
    def crossover_fn(a,b): return a
    def select_fn(pop,scores): return pop
    pop = jobs[:50] if len(jobs) > 50 else jobs
    best, metrics = genetic_algorithm.genetic_algorithm(pop, fitness, mutate_fn, crossover_fn, select_fn, generations=5, log_metrics=True)
    # Use final population sorted by fitness as ranking
    scored = [(fitness(j), j) for j in pop]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [jobs.index(j) for _, j in scored]


def evaluate(print_table=True):
    jobs = load_jobs()
    relevance = relevance_scores(jobs)
    neighbors = build_similarity_neighbors(jobs, top_k=5)

    # Probabilistic hill climb wrapper for ranking (single run)
    def run_prob_hc():
        if not jobs:
            return []
        current_idx = 0
        path = []
        import math, random
        T = 1.0
        cooling = 0.9
        rel = relevance_scores(jobs)
        for _ in range(10):
            path.append(current_idx)
            neigh = neighbors[current_idx]
            if not neigh:
                break
            cand = random.choice(neigh)
            delta = rel[cand] - rel[current_idx]
            if delta > 0 or random.random() < math.exp(delta / max(T, 1e-9)):
                current_idx = cand
            T *= cooling
        return path

    systems = {
        "BFS": run_bfs(jobs, neighbors),
        "DFS": run_dfs(jobs, neighbors),
        "UCS": run_ucs(jobs, neighbors),
        "A*": run_astar(jobs, neighbors),
        "Hill Climb": run_hill_climb(jobs, neighbors),
        "Prob Hill Climb": run_prob_hc(),
        "Genetic Algorithm": run_ga(jobs),
    }

    rows = []
    for name, ranking in systems.items():
        for k in K_VALUES:
            p = precision_at_k(ranking, relevance, k)
            n = ndcg_at_k(ranking, relevance, k)
            rows.append({"algorithm": name, "K": k, "precision_at_k": p, "ndcg_at_k": n})
    df = pd.DataFrame(rows)
    if print_table:
        print(df.pivot(index="algorithm", columns="K", values=["precision_at_k", "ndcg_at_k"]))
    return df


if __name__ == "__main__":
    evaluate(print_table=True)
