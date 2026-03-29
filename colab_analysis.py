"""
Colab-friendly runner for performance, empirical, and ranking analyses.
Usage (local):
    python colab_analysis.py --data-path "data/Sample Datasets-2 data job posts TEST DATA.csv" --out-dir outputs

In Colab:
    %pip install -q -r requirements.txt
    !python colab_analysis.py --data-path /content/drive/MyDrive/your_dataset.csv --out-dir /content/outputs --skip-plots
"""

import argparse
import importlib
import os
import sys
import pathlib

def in_colab():
    return "google.colab" in sys.modules

def ensure_matplotlib_noninteractive(out_dir: str):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(out_dir, ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")


def load_module_fresh(module_name: str):
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)

def run_performance(data_path: str, out_dir: str, skip_plots: bool):
    os.environ["ASSIGNMENT_DATA_PATH"] = data_path
    os.environ["ASSIGNMENT_OUTPUT_DIR"] = out_dir
    os.environ["ASSIGNMENT_SKIP_PLOTS"] = "1" if skip_plots else "0"
    perf = load_module_fresh("analysis.performance_analysis")
    return getattr(perf, "df_results", None)

def run_empirical(data_path: str, out_dir: str, skip_plots: bool):
    os.environ["ASSIGNMENT_DATA_PATH"] = data_path
    os.environ["ASSIGNMENT_OUTPUT_DIR"] = out_dir
    os.environ["ASSIGNMENT_SKIP_PLOTS"] = "1" if skip_plots else "0"
    emp = load_module_fresh("analysis.empirical_analysis")
    return getattr(emp, "empirical_df", None)

def run_ranking(data_path: str, out_dir: str):
    rank = load_module_fresh("analysis.ranking_evaluation")
    rank.DATA_PATH = data_path
    df = rank.evaluate(print_table=False)
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
