# Business Requirements Document (BRD)

## Project Title
AI Job Search Algorithm Analysis

## Objective
Develop, implement, and analyze multiple search algorithms (uninformed, informed, and evolutionary) to retrieve and rank AI-related job postings from two datasets. Evaluate algorithm effectiveness and visualize findings to identify top AI job clusters and skill frequency distributions.

## Scope
- Implement uninformed search algorithms: Breadth-First Search (BFS), Depth-First Search (DFS), Uniform Cost Search (UCS)
- Implement informed/evolutionary search algorithms: Genetic Algorithm, Hill Climbing
- Analyze and compare algorithms on:
  - Time complexity
  - Space complexity
  - Runtime
  - Memory usage
  - Node expansion
  - Convergence
  - Ranking quality
- Visualize results (tables, graphs)
- Identify top AI job clusters and skill frequency distributions
- Ensure code is portable to Google Colab

## Functional Requirements
1. Data ingestion and preprocessing for two job advertisement datasets
2. Implementation of each search algorithm as modular, reusable Python code
3. Ranking and retrieval of AI-related job postings
4. Metrics collection for each algorithm
5. Visualization of results (matplotlib, seaborn, pandas, etc.)
6. Exportable reports (tables, graphs)
7. Documentation for code portability (especially for Google Colab)

## Non-Functional Requirements
- Code must be well-documented and modular
- Visualizations must be clear and publication-ready
- All dependencies must be open-source and Colab-compatible

## Project Plan

### 1. Data Preparation
- Gather and inspect datasets
- Clean and preprocess data (normalize, handle missing values, extract relevant features)

### 2. Algorithm Implementation
- Uninformed: BFS, DFS, UCS
- Informed/Evolutionary: Genetic Algorithm, Hill Climbing
- Modularize code for easy reuse and portability

### 3. Search & Ranking
- Define criteria for “AI-related” jobs
- Implement ranking mechanism for search results

### 4. Evaluation & Metrics
- Measure time/space complexity, runtime, memory usage, node expansion, convergence, ranking quality
- Log and store metrics for each run

### 5. Visualization & Reporting
- Create tables and graphs for:
  - Algorithm performance comparison
  - Top AI job clusters
  - Skill frequency distributions
- Prepare summary report

### 6. Portability & Documentation
- Ensure all code runs in Google Colab
- Write clear documentation and usage instructions

---

## Next Steps

1. Confirm datasets and data format
2. Set up project folder structure and requirements.txt
3. Begin with data ingestion and preprocessing scripts
