# FAISS Project Poster Outline (A0 Drafting Guide)

## 1) Title & Scope
- **FAISS Vector Search Evaluation: IVFPQ vs HNSWSQ vs LSH**
- Datasets: **SIFT**, **GIST**, **GloVe**
- Goal: evaluate recall-speed trade-offs and robustness
- Add authors, affiliation, course, date

**Suggested visual**
- Small pipeline sketch (data ? index build ? search ? metrics)

## 2) Motivation & Research Questions
- ANN systems need both high accuracy and high throughput
- Compare three FAISS method families under a common protocol
- Questions:
- Which method maximizes Recall@10?
- Which method maximizes QPS?
- How do hardness and noise affect behavior?

**Suggested visual**
- Compact “questions” callout box

## 3) Methods
- **IVFPQ**: inverted file + product quantization
- **HNSWSQ**: graph search + scalar quantization
- **LSH**: hashing baseline
- Sweeps:
- IVFPQ: `nprobe`
- HNSWSQ: `efSearch` and SQ bit precision
- LSH: `nbits`

**Suggested visual**
- One method/parameter comparison table

## 4) Datasets & Experimental Setup
- SIFT and GIST: **L2** metric
- GloVe: angular-style setup (L2 normalize + inner product)
- Exact ground truth from FAISS Flat indexes
- Primary metrics: **Recall@10** and **QPS**
- Extra analyses: RC, easy/hard splits, noisy queries

**Suggested visual**
- Setup block + metric definitions

## 5) Main Cross-Dataset Results (Recall-QPS)
- HNSWSQ is the top-recall method on all three datasets
- IVFPQ reaches the highest throughput in speed-oriented settings
- LSH is generally weaker on recall

**Suggested figures**
- `results/plots/sift_recall_vs_qps_final.png`
- `results/plots/gist_recall_vs_qps_final.png`
- `results/plots/glove_recall_vs_qps_final.png`
- `results/plots/best_recall_by_method_dataset.png`

**Suggested tables**
- `results/tables/best_overall_per_dataset.csv`
- `results/tables/best_config_per_dataset_method.csv`
- `results/tables/high_qps_summary.csv`

## 6) Query Hardness (RC) and Easy/Hard Performance
- RC used as a query hardness signal
- Easy/hard split via q75/q25 quantiles
- Observed pattern: `recall_easy > recall_all > recall_hard`

**Suggested figures**
- `results/plots/easy_hard_comparison.png`
- `results/plots/sift_rc_histogram.png`
- `results/plots/gist_rc_histogram.png`
- `results/plots/glove_rc_histogram.png`

**Suggested table**
- `results/tables/all_easy_hard_results.csv`

## 7) Noisy-Query Robustness
- Gaussian perturbation applied at multiple sigma levels
- Compare recall degradation by method and dataset
- Robustness is strongly dataset-dependent

**Suggested figures**
- `results/plots/noise_robustness_comparison.png`
- Optional per-dataset curves:
- `results/plots/sift_noise_vs_recall.png`
- `results/plots/gist_noise_vs_recall.png`
- `results/plots/glove_noise_vs_recall.png`

**Suggested table**
- `results/tables/all_noisy_query_results.csv`

## 8) Extra Insight A: Adaptive Search Effort on Hard Queries
- Evaluate low vs high search effort on hard-query subsets
- IVFPQ: low/high `nprobe` (same factory)
- HNSWSQ: low/high `efSearch` (same factory)
- Use this to show whether hard-query recall can be recovered at speed cost

**Suggested figure if available**
- `results/plots/adaptive_hard_query_comparison.png`

**Suggested table if available**
- `results/tables/adaptive_hard_query_results.csv`

**Poster note if not generated yet**
- Mark as “extra analysis in progress / optional extension”

## 9) Extra Insight B: Extended SIFT Noise Sensitivity
- Motivation: original SIFT noise curve was near-flat
- Test larger sigma range to check scale-mismatch hypothesis
- Compare IVFPQ/HNSWSQ/LSH under stronger perturbations
- Clarify whether SIFT robustness is intrinsic or sigma-scale dependent

**Suggested figure if available**
- `results/plots/sift_extended_noise_vs_recall.png`

**Suggested table if available**
- `results/tables/sift_extended_noise_results.csv`

**Poster note if not generated yet**
- Mark as “supplementary stress test”

## 10) Discussion, Conclusions & Reproducibility
- Recall-first choice: HNSWSQ consistently strongest
- Throughput-first choice: IVFPQ often fastest
- Best recall method does not change across datasets in current results
- RC and noise analyses provide defensible robustness insights
- Next steps: finalize README, export notebook (HTML/PDF), assemble poster

**Suggested footer block**
- Repo/QR + artifact folders: `results/tables/`, `results/plots/`, `results/exports/`

---

## Results-Based Conclusions (Poster-Ready)
- **Recall dominance:** HNSWSQ achieved the highest Recall@10 on SIFT, GIST, and GloVe.
- **Throughput dominance:** IVFPQ produced the strongest high-QPS configurations in most dataset/method comparisons.
- **Method stability across datasets:** The top-recall method (HNSWSQ) did not change across datasets.
- **RC behavior:** RC split behaved as expected, with easy queries consistently yielding higher recall than hard queries.
- **Noise behavior:** Robustness trends are dataset-specific; degradation is stronger in some datasets than others.
- **Adaptive effort insight:** Increasing `nprobe`/`efSearch` for hard queries is a plausible strategy to improve hard-query recall, with explicit speed cost.

## Recommended Talk Track (60-90 seconds)
1. Start with the ANN trade-off problem and research questions.
2. Show Recall-QPS plots and immediately state the main winner by recall and speed.
3. Use RC/easy-hard to show that query difficulty materially changes outcomes.
4. Show noise robustness to stress-test conclusions.
5. Add adaptive-effort and extended-SIFT-noise as practical extensions.
6. Close with actionable method-selection guidance.

## Final Poster Checklist
- Title/authors/affiliation/date
- Research questions and setup
- Method/parameter summary
- Final Recall-QPS plots (3 datasets)
- Best-method comparison figure
- RC/easy-hard figure(s) + summary table
- Noise robustness figure + summary table
- Extra-insight placeholders or final figures:
- `adaptive_hard_query_comparison.png`
- `sift_extended_noise_vs_recall.png`
- Clear conclusions + reproducibility/QR footer
