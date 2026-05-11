"""Dataset loading utilities for SIFT/GIST FAISS experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"


def _resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to project root when needed."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def read_fvecs(path: str | Path) -> np.ndarray:
    """Read a `.fvecs` file into a float32 array of shape (n, d)."""
    file_path = _resolve_path(path)
    raw = np.fromfile(file_path, dtype=np.int32)

    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension in {file_path}: {dim}")
    if raw.size % (dim + 1) != 0:
        raise ValueError(f"Corrupted fvecs file: {file_path}")

    vectors_i32 = raw.reshape(-1, dim + 1)
    if not np.all(vectors_i32[:, 0] == dim):
        raise ValueError(f"Inconsistent vector dimensions in {file_path}")

    return vectors_i32[:, 1:].view(np.float32).copy()


def read_ivecs(path: str | Path) -> np.ndarray:
    """Read an `.ivecs` file into an int32 array of shape (n, d)."""
    file_path = _resolve_path(path)
    raw = np.fromfile(file_path, dtype=np.int32)

    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)

    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension in {file_path}: {dim}")
    if raw.size % (dim + 1) != 0:
        raise ValueError(f"Corrupted ivecs file: {file_path}")

    vectors_i32 = raw.reshape(-1, dim + 1)
    if not np.all(vectors_i32[:, 0] == dim):
        raise ValueError(f"Inconsistent vector dimensions in {file_path}")

    return vectors_i32[:, 1:].copy()


def load_sift() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the SIFT dataset as (xb, xq, xlearn, gt)."""
    root = RAW_DATA_ROOT / "sift"
    xb = read_fvecs(root / "sift_base.fvecs")
    xq = read_fvecs(root / "sift_query.fvecs")
    xlearn = read_fvecs(root / "sift_learn.fvecs")
    gt = read_ivecs(root / "sift_groundtruth.ivecs")
    return xb, xq, xlearn, gt


def load_gist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the GIST dataset as (xb, xq, xlearn, gt)."""
    root = RAW_DATA_ROOT / "gist"
    xb = read_fvecs(root / "gist_base.fvecs")
    xq = read_fvecs(root / "gist_query.fvecs")
    xlearn = read_fvecs(root / "gist_learn.fvecs")
    gt = read_ivecs(root / "gist_groundtruth.ivecs")
    return xb, xq, xlearn, gt
