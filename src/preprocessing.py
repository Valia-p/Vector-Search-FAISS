"""Preprocessing helpers for FAISS vector-search experiments."""

from __future__ import annotations

import faiss
import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D float32 array of vectors and return it."""
    if x.ndim != 2:
        raise ValueError(f"`x` must be 2D, got shape {x.shape}.")

    x_f32 = x.astype(np.float32, copy=False)
    faiss.normalize_L2(x_f32)
    return x_f32
