# FAISS Project Poster Outline (A0 Drafting Guide)

## 1) Title and Context
- **FAISS Vector Search Evaluation: IVFPQ vs HNSWSQ vs LSH**
- Benchmark study on **SIFT**, **GIST**, **GloVe**
- Focus: accuracy-speed trade-offs and robustness
- Add authors, affiliation, course, date

**Suggested visual**
- Small pipeline icon or experiment flow sketch

## 2) Motivation and Questions
- ANN systems require balancing recall and throughput
- Compare three FAISS method families under a common protocol
- Main questions:
- Which method gives best Recall@10?
- Which method gives best QPS?
- How do query hardness and noise affect performance?

**Suggested visual**
- Compact “research questions” box

## 3) Methods
- **IVFPQ**: inverted file + product quantization
- **HNSWSQ**: graph search + scalar quantization
- **LSH**: hashing baseline
- Swept parameters:
- IVFPQ: `nprobe`
- HNSWSQ: `efSearch` and SQ precision
- LSH: `nbits`

**Suggested visual**
- One comparison table (method, key params, expected effect)

## 4) Datasets and Setup
- SIFT and GIST evaluated with **L2** metric
- GloVe evaluated as angular-style search:
- L2-normalize vectors
- use inner product search
- Exact ground truth built with FAISS Flat indices
- Metrics: **Recall@10** and **QPS** (search-time only)

**Suggested visual**
- Setup block diagram + metric definitions

## 5) Main Recall-QPS Results
- HNSWSQ reaches the strongest recall in all datasets
- IVFPQ provides very high throughput in speed-oriented settings
- LSH generally trails in recall

**Suggested figures**
- `results/plots/sift_recall_vs_qps_final.png`
- `results/plots/gist_recall_vs_qps_final.png`
- `results/plots/glove_recall_vs_qps_final.png`
- `results/plots/best_recall_by_method_dataset.png`

**Suggested tables**
- `results/tables/best_overall_per_dataset.csv`
- `results/tables/best_config_per_dataset_method.csv`

## 6) Throughput-Oriented View
- High-QPS settings sacrifice recall, especially for IVFPQ/LSH
- HNSWSQ remains competitive with strong recall across settings
- Use this section for deployment-oriented interpretation

**Suggested table**
- `results/tables/high_qps_summary.csv`

## 7) Query Hardness (RC) and Easy/Hard Split
- Relative Contrast (RC) used as query hardness indicator
- Easy/hard split from RC quantiles (q75 / q25)
- Consistent pattern: `recall_easy > recall_all > recall_hard`

**Suggested figures**
- `results/plots/easy_hard_comparison.png`
- `results/plots/sift_rc_histogram.png`
- `results/plots/gist_rc_histogram.png`
- `results/plots/glove_rc_histogram.png`

**Suggested table**
- `results/tables/all_easy_hard_results.csv`

## 8) Noisy Query Robustness
- Queries perturbed with Gaussian noise (`sigma = 0.00, 0.01, 0.05, 0.10`)
- Compare recall degradation by method and dataset
- Robustness is dataset-dependent

**Suggested figures**
- `results/plots/noise_robustness_comparison.png`
- Optional per-dataset noise plots:
- `sift_noise_vs_recall.png`, `gist_noise_vs_recall.png`, `glove_noise_vs_recall.png`

**Suggested table**
- `results/tables/all_noisy_query_results.csv`

## 9) Discussion and Practical Takeaways
- If priority is maximum recall, HNSWSQ is the strongest choice here
- If priority is very high throughput, IVFPQ is often preferable
- LSH remains a lightweight baseline with lower recall
- RC and noise analyses add useful stress-testing beyond static Recall-QPS points

**Suggested visual**
- “When to choose each method” decision box

## 10) Conclusions and Reproducibility
- Three-dataset FAISS scaffold completed end-to-end
- Cross-dataset summaries and comparison plots generated
- Next steps:
- finalize README/report wording
- export notebook (HTML/PDF)
- assemble final A0 poster layout

**Suggested footer block**
- Repo path / QR code
- Key artifact folders: `results/tables/`, `results/plots/`, `results/exports/`

---

## Recommended Presentation Flow (60-90 seconds core)
1. Problem and evaluation goal.
2. Methods and datasets.
3. Recall-QPS results (main comparison).
4. RC easy/hard findings.
5. Noise robustness findings.
6. Final takeaways and practical method choice.

## Final Poster Build Checklist
- Clear title, authors, affiliation
- Research questions
- Setup + metric definitions
- 3 Recall-QPS final plots
- Best-method comparison plot
- RC/easy-hard figure(s)
- Noise robustness figure
- Concise conclusions
- Reproducibility/QR footer
