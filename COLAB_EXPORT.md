# Google Colab Export Guide

## 1) Upload project to Colab
- Upload `colab_export.zip` to Colab and extract it, or clone your repo.
- Ensure your dataset CSV is available in the runtime (or Google Drive).

## 2) Install dependencies
```bash
%cd /content/assignment_1_36121
%pip install -q -r requirements.txt
```

## 3) Run all analyses
```bash
!python colab_analysis.py \
  --data-paths \
  "/content/assignment_1_36121/data/Sample Dataset-2 data job posts.csv" \
  "/content/assignment_1_36121/data/Sample Datasets-3 Job posted LinkedIn (2023 - 2024).csv" \
  --out-dir "/content/outputs" \
  --skip-plots
```

Use `--skip-plots` only if you do not need PNG charts.

## 4) Output files
- `/content/outputs/<dataset_tag>/performance_metrics.csv`
- `/content/outputs/<dataset_tag>/empirical_algorithm_metrics.csv`
- `/content/outputs/<dataset_tag>/ranking_metrics.csv`
- `/content/outputs/<dataset_tag>/plots/*.png` (if plots are enabled)
- `/content/outputs/dataset_run_summary.csv`

## 5) Optional: run with Google Drive dataset
```bash
from google.colab import drive
drive.mount('/content/drive')

!python colab_analysis.py \
  --data-paths "/content/drive/MyDrive/dataset1.csv" "/content/drive/MyDrive/dataset2.csv" \
  --out-dir "/content/drive/MyDrive/assignment_outputs"
```
