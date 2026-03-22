"""
Colab-friendly runner for performance, empirical, and ranking analyses.
Usage (local):
    python colab_analysis.py --data-path data/Sample\ Datasets-2\ data\ job\ posts\ TEST\ DATA.csv --out-dir outputs

In Colab:
    %pip install -q -r requirements.txt
    !python colab_analysis.py --data-path /content/drive/MyDrive/your_dataset.csv --out-dir /content/outputs --skip-plots
"""

import argparse
import os
import sys
import pathlib

def in_colab():
    return "google.colab" in sys.modules

def ensure_matplotlib_noninteractive(out_dir: str):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(out_dir, ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")

def run_performance(data_path: str, out_dir: str, skip_plots: bool):
    from analysis import performance_analysis as perf
    perf.DATA_PATH = data_path
    perf.os.makedirs(out_dir, exist_ok=True)
    perf.os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    df = None
    # Re-run with plotting toggled by skip_plots
    orig_show = perf.plt.show
    if skip_plots:
        perf.plt.show = lambda *args, **kwargs: None
    df = perf.df_results if hasattr(perf, "df_results") else None
    if df is None:
        # If not already run, call module (it executes on import)
        pass
    # Save outputs if present
    try:
        perf.df_results.to_csv(os.path.join(out_dir, "performance_metrics.csv"), index=False)
    except Exception:
        pass
    perf.plt.show = orig_show
    return df

def run_empirical(data_path: str, out_dir: str, skip_plots: bool):
    from analysis import empirical_analysis as emp
    emp.DATA_PATH = data_path
    emp.os.makedirs(out_dir, exist_ok=True)
    emp.os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    orig_show = emp.plt.show
    if skip_plots:
        emp.plt.show = lambda *args, **kwargs: None
    df = None
    try:
        emp.empirical_df.to_csv(os.path.join(out_dir, "empirical_metrics.csv"), index=False)
    except Exception:
        pass
    emp.plt.show = orig_show
    return df

def run_ranking(data_path: str, out_dir: str):
    import pandas as pd
    from analysis import ranking_evaluation as rank
    rank.DATA_PATH = data_path
    # Re-run evaluate to rebuild df
    rank.evaluate()
    # The evaluate prints, we rebuild for saving
    jobs = rank.load_jobs()
    relevance = rank.relevance_scores(jobs)
    neighbors = rank.build_similarity_neighbors(jobs, top_k=5)
    systems = {
        "BFS": rank.run_bfs(jobs, neighbors),
        "DFS": rank.run_dfs(jobs, neighbors),
        "UCS": rank.run_ucs(jobs, neighbors),
        "A*": rank.run_astar(jobs, neighbors),
        "Hill Climb": rank.run_hill_climb(jobs, neighbors),
        "Prob Hill Climb": rank.run_prob_hc() if hasattr(rank, "run_prob_hc") else [],
        "Genetic Algorithm": rank.run_ga(jobs),
    }
    rows = []
    for name, ranking in systems.items():
        for k in rank.K_VALUES:
            p = rank.precision_at_k(ranking, relevance, k)
            n = rank.ndcg_at_k(ranking, relevance, k)
            rows.append({"algorithm": name, "K": k, "precision_at_k": p, "ndcg_at_k": n})
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "ranking_metrics.csv"), index=False)
    return df

def main():
    parser = argparse.ArgumentParser(description="Colab-friendly analysis runner")
    parser.add_argument("--data-path", default="data/Sample Datasets-2 data job posts TEST DATA.csv")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--skip-plots", action="store_true", help="Skip saving plots (useful in headless)")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    ensure_matplotlib_noninteractive(out_dir)

    # Add repo root to PYTHONPATH
    repo_root = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))

    print(f"Running analyses with data: {args.data_path}")
    perf_df = run_performance(args.data_path, out_dir, args.skip_plots)
    emp_df = run_empirical(args.data_path, out_dir, args.skip_plots)
    rank_df = run_ranking(args.data_path, out_dir)

    print("Done. Outputs saved to", out_dir)


if __name__ == "__main__":
    main()
