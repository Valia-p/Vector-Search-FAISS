# FAISS Vector Search Evaluation: IVFPQ, HNSWSQ, and LSH

## Project Overview
This project evaluates three FAISS approximate nearest-neighbor methods (`IVFPQ`, `HNSWSQ`, `LSH`) on three benchmark datasets (`SIFT`, `GIST`, `GloVe`). The experimental pipeline measures the recall-speed trade-off using `Recall@10` and `QPS`, and extends the analysis with Relative Contrast (RC), easy-vs-hard query splits, and noisy-query robustness experiments.

## Repository Structure
- `notebooks/experiments.ipynb`: Main end-to-end experimental notebook (SIFT, GIST, GloVe, cross-dataset summary).
- `src/data_loading.py`: Dataset readers/loaders for SIFT and GIST.
- `src/preprocessing.py`: Vector preprocessing utilities (L2 normalization).
- `src/metrics.py`: Evaluation metrics (Recall@k).
- `src/benchmarks.py`: Ground-truth construction and search benchmarking helpers.
- `src/indexes.py`: FAISS index builders and search-parameter helpers.
- `src/rc_analysis.py`: RC computation and noisy-query generation.
- `src/utils.py`: Reserved utility module.
- `data/raw/`: Raw dataset files (local, not committed).
- `data/processed/`: Optional processed artifacts.
- `data/noisy_queries/`: Optional stored noisy query sets.
- `results/tables/`: CSV outputs (per-dataset and cross-dataset summaries).
- `results/plots/`: Generated figures.
- `results/exports/`: Optional exported notebook/report artifacts.

## Environment Setup (Windows PowerShell)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) Register a dedicated notebook kernel:

```powershell
python -m ipykernel install --user --name vector-search-faiss --display-name "Vector-Search-FAISS"
```

4. Start Jupyter:

```powershell
jupyter notebook notebooks/experiments.ipynb
```

If needed, switch the notebook kernel to `Vector-Search-FAISS`.

## Dataset Placement
Place dataset files exactly in the following locations:

- `data/raw/sift/`
  - `sift_base.fvecs`
  - `sift_query.fvecs`
  - `sift_learn.fvecs`
  - `sift_groundtruth.ivecs`
- `data/raw/gist/`
  - `gist_base.fvecs`
  - `gist_query.fvecs`
  - `gist_learn.fvecs`
  - `gist_groundtruth.ivecs`
- `data/raw/glove/`
  - `glove.twitter.27B.100d.txt`

Metric assumptions:
- `SIFT` and `GIST`: Euclidean (`L2`) search.
- `GloVe`: Angular-style setup via L2 normalization + inner product search.

## How to Run the Experiments
Run `notebooks/experiments.ipynb` in this order:

1. Setup cells (imports, paths, module imports).
2. `SIFT` sections (starter + sweeps + RC + easy/hard + noisy robustness).
3. `GIST` sections (same pipeline).
4. `GloVe` sections (same pipeline, with normalized inner-product setup).
5. `Cross-Dataset Summary` section (final aggregation tables and comparison plots).

If a kernel restart occurs, rerun required upstream cells (or reload CSV outputs before cross-dataset summary).

## Outputs
Generated artifacts are saved under:

- `results/tables/`
  - Per-dataset sweeps: `sift_*`, `gist_*`, `glove_*`
  - RC/easy-hard/noise tables per dataset
  - Cross-dataset summaries (for example `all_datasets_results.csv`, best-config summaries, robustness/easy-hard combined tables)
- `results/plots/`
  - Recall-QPS plots per dataset
  - RC histograms
  - easy-vs-hard recall comparison plots
  - noisy-query robustness plots
  - cross-dataset comparison plots
- `results/exports/`
  - Optional exported notebook/report artifacts

## Methodological Notes
- Exact ground truth is built with FAISS Flat indices:
  - `IndexFlatL2` for SIFT/GIST
  - `IndexFlatIP` for normalized GloVe
- Main metrics are `Recall@10` and `QPS`.
- RC is used as a query-hardness indicator.
- Noisy queries are generated with Gaussian perturbation.
- For GloVe (`d=100`), IVFPQ subquantizers were adapted to valid divisors of 100 (instead of invalid `M` choices).

## Reproducibility Notes
- Timing values vary slightly across runs/machines.
- `QPS` is measured on search time only (not training/build/add time).
- Minor timing jitter is normal, especially for very fast runs.
- GloVe uses a subset (`max_vectors`) for feasibility and faster iteration.

## Deliverables
- Completed notebook: `notebooks/experiments.ipynb`
- Exported notebook artifacts (HTML/PDF)
- Result tables (`results/tables/`)
- Result plots (`results/plots/`)
- Project README
- Poster for the final presentation
