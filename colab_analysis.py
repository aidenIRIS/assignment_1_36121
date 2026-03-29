"""
Colab-friendly runner for performance, empirical, and ranking analyses.
Usage (local):
    python colab_analysis.py --data-paths "data/file1.csv" "data/file2.csv" --out-dir outputs

In Colab:
    %pip install -q -r requirements.txt
    !python colab_analysis.py --data-paths /content/drive/MyDrive/data1.csv /content/drive/MyDrive/data2.csv --out-dir /content/outputs --skip-plots
"""

import argparse
import importlib
import os
import sys
import pathlib
import re
import pandas as pd

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


def slugify_filename(path: str) -> str:
    stem = pathlib.Path(path).stem
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return slug or "dataset"


def run_all_for_dataset(data_path: str, base_out_dir: str, skip_plots: bool):
    dataset_tag = slugify_filename(data_path)
    dataset_out_dir = os.path.join(base_out_dir, dataset_tag)
    os.makedirs(dataset_out_dir, exist_ok=True)

    print(f"\nRunning analyses with data: {data_path}")
    perf_df = run_performance(data_path, dataset_out_dir, skip_plots)
    emp_df = run_empirical(data_path, dataset_out_dir, skip_plots)
    rank_df = run_ranking(data_path, dataset_out_dir)
    print(f"Finished: {data_path}")
    print(f"Saved outputs to: {dataset_out_dir}")
    return dataset_tag, dataset_out_dir, perf_df, emp_df, rank_df


def discover_default_datasets(sample_data_dir: str):
    candidates = [
        "Sample Dataset-2 data job posts.csv",
        "Sample Datasets-3 Job posted LinkedIn (2023 - 2024).csv",
        "Sample Datasets-2 data job posts TEST DATA.csv",
    ]
    search_dirs = [
        pathlib.Path.cwd() / "data",
        pathlib.Path(sample_data_dir),
        pathlib.Path("/content/sample_data"),
    ]
    found = []
    seen = set()
    for base in search_dirs:
        if not base.exists():
            continue
        for name in candidates:
            p = base / name
            if p.exists():
                s = str(p)
                if s not in seen:
                    found.append(s)
                    seen.add(s)
    return found

def main():
    parser = argparse.ArgumentParser(description="Colab-friendly analysis runner")
    parser.add_argument(
        "--data-paths",
        nargs="+",
        default=[],
        help="One or more CSV dataset paths.",
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=[],
        help="Backwards-compatible single dataset argument (can be repeated).",
    )
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument(
        "--sample-data-dir",
        default="/content/sample_data",
        help="Directory used for automatic dataset discovery when --data-paths is not provided.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip saving plots (useful in headless)")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    ensure_matplotlib_noninteractive(out_dir)

    # Add repo root to PYTHONPATH
    repo_root = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))

    all_data_paths = list(args.data_paths) + list(args.data_path)
    if not all_data_paths:
        all_data_paths = discover_default_datasets(args.sample_data_dir)
        if all_data_paths:
            print("Auto-discovered datasets:")
            for path in all_data_paths:
                print(" -", path)

    summary_rows = []
    for data_path in all_data_paths:
        if not os.path.exists(data_path):
            print(f"WARNING: dataset not found, skipping: {data_path}")
            continue
        dataset_tag, dataset_out_dir, perf_df, emp_df, rank_df = run_all_for_dataset(
            data_path, out_dir, args.skip_plots
        )
        summary_rows.append(
            {
                "dataset": data_path,
                "dataset_tag": dataset_tag,
                "output_dir": dataset_out_dir,
                "performance_rows": 0 if perf_df is None else len(perf_df),
                "empirical_rows": 0 if emp_df is None else len(emp_df),
                "ranking_rows": 0 if rank_df is None else len(rank_df),
            }
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(out_dir, "dataset_run_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print("\nDone. Summary saved to", summary_path)
    else:
        print("\nNo datasets were processed. Check --data-paths values.")


if __name__ == "__main__":
    main()
