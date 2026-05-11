"""Relative-contrast utilities for query-hardness analysis."""

from __future__ import annotations

import faiss
import numpy as np


def compute_rc(xb: np.ndarray, xq: np.ndarray, k: int, metric: int) -> np.ndarray:
    """Compute Relative Contrast (RC) for each query vector."""
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
    Dk, _ = gt.search(xq_f32, k)

    if metric == faiss.METRIC_L2:
        rk = Dk[:, k - 1]
        xb_sq_mean = np.mean(np.sum(xb_f32 * xb_f32, axis=1))
        xb_mean = np.mean(xb_f32, axis=0)
        q_sq = np.sum(xq_f32 * xq_f32, axis=1)
        dmean = xb_sq_mean + q_sq - 2.0 * (xq_f32 @ xb_mean)
    else:
        rk = 1.0 - Dk[:, k - 1]
        xb_mean = np.mean(xb_f32, axis=0)
        dmean = 1.0 - (xq_f32 @ xb_mean)

    eps = np.finfo(np.float32).eps
    rc = dmean / np.maximum(rk, eps)
    return rc.astype(np.float32, copy=False)


def add_noise(xq: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to query vectors and return float32 output."""
    if sigma < 0:
        raise ValueError("`sigma` must be non-negative.")
    xq_f32 = xq.astype(np.float32, copy=False)
    noisy = xq_f32 + sigma * np.random.randn(*xq_f32.shape).astype(np.float32)
    return noisy.astype(np.float32, copy=False)
