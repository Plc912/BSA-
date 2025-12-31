from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


def generate_piecewise_gaussian(
    n_samples: int = 1000,
    change_points: Optional[List[int]] = None,
    means: Optional[List[float]] = None,
    stds: Optional[List[float]] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate a 1D piecewise-Gaussian time series with structural changes.
    If change_points is None, randomly choose 2-3 change points.
    Returns (series, cps) where cps includes the final endpoint n.
    """
    rng = np.random.default_rng(seed)

    if change_points is None:
        num_cps = int(max(1, min(4, n_samples // 200)))
        # choose unique sorted points avoiding edges
        candidates = rng.choice(np.arange(100, n_samples - 100), size=num_cps, replace=False)
        cps = sorted(int(c) for c in candidates)
    else:
        cps = sorted(int(c) for c in change_points if 0 < c < n_samples)

    cps_full = cps + [n_samples]

    num_segs = len(cps_full)
    if means is None:
        # random walk means
        steps = rng.normal(0.0, 1.5, size=num_segs)
        m = np.cumsum(steps)
    else:
        m = np.asarray(means, dtype=float)
        if m.size != num_segs:
            raise ValueError("means length must equal number of segments")
    if stds is None:
        s = rng.uniform(0.5, 2.0, size=num_segs)
    else:
        s = np.asarray(stds, dtype=float)
        if s.size != num_segs:
            raise ValueError("stds length must equal number of segments")

    y = np.empty(n_samples, dtype=float)
    start = 0
    for i, cp in enumerate(cps_full):
        seg_len = cp - start
        y[start:cp] = rng.normal(m[i], s[i], size=seg_len)
        start = cp

    return y, cps_full


def save_series_csv(path: str, series: np.ndarray) -> None:
    df = pd.DataFrame({"value": series})
    df.to_csv(path, index=False)


def load_series_from_file(
    file_path: str,
    delimiter: str = ",",
    column: int = 0,
    skip_header: int = 0,
    column_name: str | None = None,
) -> np.ndarray:
    """
    Load a numeric series from CSV/TXT by column index or name.

    Rules:
    - If CSV and column_name provided: use that column.
    - If CSV and no column_name:
      * header present (skip_header>0 or file has header): use header=0 and column index
      * else header=None, use positional index
    - For TXT: load with numpy and pick positional column
    """
    if file_path.lower().endswith(".csv"):
        try:
            # Try reading with header row
            df = pd.read_csv(file_path, delimiter=delimiter, header=0)
            if column_name is not None:
                if column_name not in df.columns:
                    raise ValueError(f"Column '{column_name}' not found in CSV")
                return df[column_name].to_numpy(dtype=float)
            # fallback to index
            if isinstance(column, int):
                col = df.columns[column]
                return df[col].to_numpy(dtype=float)
            else:
                # column passed as name
                return df[column].to_numpy(dtype=float)
        except Exception:
            # No header present, read without header
            df = pd.read_csv(file_path, delimiter=delimiter, header=None)
            return df.iloc[:, column].to_numpy(dtype=float)
    else:
        # TXT or others: numpy loadtxt
        arr = np.loadtxt(file_path, delimiter=delimiter, skiprows=skip_header, dtype=float, ndmin=1)
        if arr.ndim == 2:
            arr = arr[:, column]
        return arr
