"""Benchmark helpers for FAISS indices."""

from __future__ import annotations

import time

import faiss
import numpy as np


def build_ground_truth(
    xb: np.ndarray, xq: np.ndarray, k: int, metric: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact k-NN ground truth with a Flat index."""
    if k <= 0:
        raise ValueError("`k` must be positive.")
    if xb.ndim != 2 or xq.ndim != 2:
        raise ValueError("`xb` and `xq` must be 2D arrays.")
    if xb.shape[1] != xq.shape[1]:
        raise ValueError(
            "`xb` and `xq` must have the same dimensionality. "
            f"Got {xb.shape[1]} and {xq.shape[1]}."
        )

    xb_f32 = xb.astype(np.float32, copy=False)
    xq_f32 = xq.astype(np.float32, copy=False)
    d = xb_f32.shape[1]

    if metric == faiss.METRIC_INNER_PRODUCT:
        gt = faiss.IndexFlatIP(d)
    elif metric == faiss.METRIC_L2:
        gt = faiss.IndexFlatL2(d)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    gt.add(xb_f32)
    D_gt, I_gt = gt.search(xq_f32, k)
    return D_gt, I_gt


def benchmark_index(
    index: faiss.Index, xq: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Search an index and return distances, ids, total time, and QPS."""
    if k <= 0:
        raise ValueError("`k` must be positive.")
    if xq.ndim != 2:
        raise ValueError("`xq` must be a 2D array.")

    xq_f32 = xq.astype(np.float32, copy=False)

    t0 = time.perf_counter()
    D, I = index.search(xq_f32, k)
    t1 = time.perf_counter()

    total_time = t1 - t0
    qps = xq_f32.shape[0] / total_time if total_time > 0 else float("inf")
    return D, I, total_time, qps
