# Analysis Plan

## 1. Data Preparation
- Place your datasets in the `data/` directory.
- Use `src/data_prep.py` to load and preprocess datasets.

## 2. Algorithm Implementation
- Implement search algorithms in `src/algorithms/` (to be created):
  - bfs.py
  - dfs.py
  - ucs.py
  - genetic_algorithm.py
  - hill_climbing.py

## 3. Search & Ranking
- Define AI-related job criteria in a config or utility module.
- Implement ranking logic in each algorithm module or a shared utility.

## 4. Evaluation & Metrics
- Create scripts in `analysis/` to:
  - Run each algorithm
  - Collect and log metrics (runtime, memory, etc.)
  - Compare results

## 5. Visualization & Reporting
- Use Jupyter notebooks or Python scripts in `analysis/` for:
  - Performance graphs
  - Tables of results
  - Skill frequency and cluster analysis

## 6. Portability & Documentation
- Ensure all scripts are compatible with Google Colab.
- Document usage in README.md and code comments.

---

### Next Steps
1. Add your datasets to the `data/` folder.
2. Confirm dataset formats and columns.
3. Begin implementing search algorithms in `src/algorithms/`.
