"""Evaluation metrics for ANN search results."""

from __future__ import annotations

import numpy as np


def recall_at_k(I_approx: np.ndarray, I_gt: np.ndarray, k: int) -> float:
    """Compute average Recall@k between approximate and exact neighbors."""
    if k <= 0:
        raise ValueError("`k` must be positive.")
    if I_approx.ndim != 2 or I_gt.ndim != 2:
        raise ValueError("`I_approx` and `I_gt` must be 2D arrays.")
    if I_approx.shape[0] != I_gt.shape[0]:
        raise ValueError(
            "`I_approx` and `I_gt` must have the same number of queries. "
            f"Got {I_approx.shape[0]} and {I_gt.shape[0]}."
        )
    if I_approx.shape[1] < k or I_gt.shape[1] < k:
        raise ValueError(
            "`k` cannot exceed the second dimension of inputs. "
            f"Got k={k}, I_approx.shape[1]={I_approx.shape[1]}, I_gt.shape[1]={I_gt.shape[1]}."
        )

    hits = 0
    for i in range(I_gt.shape[0]):
        hits += len(set(I_approx[i, :k]).intersection(set(I_gt[i, :k])))
    return hits / (I_gt.shape[0] * k)
