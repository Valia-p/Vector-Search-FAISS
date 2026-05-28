# FAISS Vector Search Evaluation

An experimental study comparing three approximate nearest-neighbor (ANN) index methods — **IVFPQ**, **HNSWSQ**, and **LSH** — across the SIFT, GIST, and GloVe benchmark datasets. The evaluation covers accuracy (Recall@10), throughput (QPS), index size, robustness to query noise, and performance on hard vs. easy queries via Relative Contrast analysis.

---

## Table of Contents

- [Methods](#methods)
- [Datasets](#datasets)
- [Running the Notebook](#running-the-notebook)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [Configuration](#configuration)

---

## Methods

| Method | Description |
|--------|-------------|
| **IVFPQ** | Inverted File index with Product Quantization. Trades accuracy for speed via coarse clustering (`nprobe`) and compressed vector codes. |
| **HNSWSQ** | Hierarchical Navigable Small World graph with Scalar Quantization. High accuracy at the cost of memory; controlled by `efSearch`. |
| **LSH** | Locality-Sensitive Hashing. Fastest build time; accuracy improves with the number of hash bits (`nbits`). |

SIFT and GIST use **L2 (Euclidean) distance**. GloVe vectors are L2-normalized and searched with **inner product** (equivalent to cosine similarity).

---

## Datasets

Download and extract the datasets into the following structure before running:

```
data/raw/
├── sift/
│   ├── sift_base.fvecs
│   ├── sift_query.fvecs
│   ├── sift_learn.fvecs
│   └── sift_groundtruth.ivecs
├── gist/
│   ├── gist_base.fvecs
│   ├── gist_query.fvecs
│   ├── gist_learn.fvecs
│   └── gist_groundtruth.ivecs
└── glove/
    └── glove.twitter.27B.100d.txt
```

| Dataset | Vectors | Dimensions | Source |
|---------|--------:|-----------:|--------|
| SIFT    | 1M base + 10K queries | 128 | ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz |
| GIST    | 1M base + 1K queries  | 960 | ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz |
| GloVe   | 200K (capped)         | 100 | http://nlp.stanford.edu/data/glove.twitter.27B.zip |

---

## Running the Notebook

There are two ways to run the experiments — choose whichever fits your setup.

### Option 1 — Local (Jupyter / VS Code)

The notebook at `notebooks/experiments.ipynb` is configured for **local execution**. It auto-detects the project root and saves all outputs to `results/`.

**Steps:**

1. Clone the repository and enter the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the datasets (see [Datasets](#datasets)) and place them under `data/raw/`.
4. Open `notebooks/experiments.ipynb` in Jupyter or VS Code and click **Run All**.

The first cell runs `%pip install -r ../requirements.txt` automatically, so dependencies are always in sync with the active kernel.

---

### Option 2 — Google Colab

A ready-to-run Colab version of the notebook is available here:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-XruUVMSTFoDz0Ck2Q4ue0OuvbDUgggu?usp=sharing)

The Colab notebook reads datasets from Google Drive. Mount your Drive and place the datasets under `MyDrive/faiss_project/data/raw/` before running.

---

## Project Structure

```
.
├── data/
│   ├── raw/                  # Downloaded datasets (not tracked by git)
│   ├── processed/            # Intermediate processed data
│   └── noisy_queries/        # Perturbed query vectors (.npy)
├── notebooks/
│   └── experiments.ipynb     # Main experiment notebook (local)
├── results/
│   ├── plots/                # Generated figures (.png)
│   └── tables/               # Generated result tables (.csv)
├── src/                      # Source modules
├── requirements.txt
└── README.md
```

---

## Outputs

Each notebook run produces the following artifacts:

| Path | Contents |
|------|----------|
| `results/tables/` | CSV files with Recall@10, QPS, and index size for every configuration |
| `results/plots/`  | Recall–QPS trade-off curves, RC histograms, noise robustness plots, easy/hard comparisons |
| `data/noisy_queries/` | Perturbed query vectors at σ = 0.01, 0.05, 0.10 |

Re-running the notebook overwrites all existing outputs.

---

## Configuration

All experiment parameters are defined in the **Experiment Configuration** cell of the notebook.

| Parameter | Values |
|-----------|--------|
| k (nearest neighbors) | 10 |
| IVFPQ `nprobe` sweep | 1, 4, 8, 16, 32, 64 |
| IVFPQ PQ subspaces (SIFT/GIST) | PQ8x8, PQ16x8, PQ32x8 |
| IVFPQ PQ subspaces (GloVe) | PQ10x8, PQ20x8, PQ25x8 |
| HNSWSQ `efSearch` sweep | 16, 32, 64, 128, 256 |
| HNSWSQ SQ bit-width | SQ4, SQ6, SQ8 |
| LSH `nbits` | 128, 256, 512 |
| Noise σ (standard) | 0.00, 0.01, 0.05, 0.10 |
| Noise σ (extended SIFT) | 0.00, 0.10, 0.50, 1.00, 2.00 |
| GloVe max vectors | 200,000 |
