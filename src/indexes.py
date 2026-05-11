"""FAISS index builders and search-parameter helpers."""

from __future__ import annotations

import faiss
import numpy as np


def _build_factory_index(
    xb: np.ndarray, d: int, metric: int, factory: str
) -> faiss.Index:
    """Build, train (if needed), and populate a FAISS factory index."""
    if d <= 0:
        raise ValueError("`d` must be positive.")
    if xb.ndim != 2:
        raise ValueError("`xb` must be a 2D array.")
    if xb.shape[1] != d:
        raise ValueError(f"`d` must match xb.shape[1]. Got d={d}, xb.shape[1]={xb.shape[1]}.")
    if not factory:
        raise ValueError("`factory` cannot be empty.")

    xb_f32 = xb.astype(np.float32, copy=False)
    index = faiss.index_factory(d, factory, metric)
    if not index.is_trained:
        index.train(xb_f32)
    index.add(xb_f32)
    return index


def build_ivfpq_index(xb: np.ndarray, d: int, metric: int, factory: str) -> faiss.Index:
    """Build an IVFPQ index from a FAISS factory string."""
    return _build_factory_index(xb=xb, d=d, metric=metric, factory=factory)


def build_hnswsq_index(
    xb: np.ndarray, d: int, metric: int, factory: str
) -> faiss.Index:
    """Build an HNSWSQ index from a FAISS factory string."""
    return _build_factory_index(xb=xb, d=d, metric=metric, factory=factory)


def build_lsh_index(d: int, nbits: int) -> faiss.Index:
    """Build an LSH index with explicit hash-bit count."""
    if d <= 0:
        raise ValueError("`d` must be positive.")
    if nbits <= 0:
        raise ValueError("`nbits` must be positive.")
    return faiss.IndexLSH(d, nbits)


def set_ivfpq_nprobe(index: faiss.Index, nprobe: int) -> None:
    """Set the IVFPQ search parameter `nprobe`."""
    if nprobe <= 0:
        raise ValueError("`nprobe` must be positive.")
    ps = faiss.ParameterSpace()
    ps.set_index_parameter(index, "nprobe", int(nprobe))


def set_hnswsq_efsearch(index: faiss.Index, ef_search: int) -> None:
    """Set the HNSWSQ search parameter `efSearch`."""
    if ef_search <= 0:
        raise ValueError("`ef_search` must be positive.")
    ps = faiss.ParameterSpace()
    ps.set_index_parameter(index, "efSearch", int(ef_search))
