# Google Colab (GitHub-first) Guide

## 1) Clone from GitHub in Colab
```bash
%cd /content
!git clone https://github.com/aidenIRIS/assignment_1_36121.git
%cd /content/assignment_1_36121
```

No zip upload is required.

## 2) Install dependencies
```bash
%pip install -q -r requirements.txt
```

## 3) Put datasets into `/content/sample_data`
- Colab often already uses `/content/sample_data`.
- Place these two files there:
  - `Sample Dataset-2 data job posts.csv`
  - `Sample Datasets-3 Job posted LinkedIn (2023 - 2024).csv`

## 4) Run all analyses (auto-discovery)
```bash
!python colab_analysis.py \
  --out-dir "/content/outputs" \
  --skip-plots
```

The runner now auto-discovers known dataset names from `/content/sample_data`.

## 5) Output files
- `/content/outputs/<dataset_tag>/performance_metrics.csv`
- `/content/outputs/<dataset_tag>/empirical_algorithm_metrics.csv`
- `/content/outputs/<dataset_tag>/ranking_metrics.csv`
- `/content/outputs/<dataset_tag>/plots/*.png` (if plots are enabled)
- `/content/outputs/dataset_run_summary.csv`

## 6) Optional: explicit paths or custom sample_data dir
```bash
!python colab_analysis.py \
  --data-paths "/content/drive/MyDrive/dataset1.csv" "/content/drive/MyDrive/dataset2.csv" \
  --out-dir "/content/drive/MyDrive/assignment_outputs"
```

Or:
```bash
!python colab_analysis.py --sample-data-dir "/content/sample_data" --out-dir "/content/outputs"
```
