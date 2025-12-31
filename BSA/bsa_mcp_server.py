from __future__ import annotations

from typing import Optional, Dict, Any, List
import os

from fastmcp import FastMCP
import numpy as np

from bsa.bocpd import detect_change_points
from bsa.utils import (
    generate_piecewise_gaussian,
    save_series_csv,
    load_series_from_file,
)


mcp = FastMCP("bsa")


@mcp.tool()
def bsa_detect_from_series(
    series: List[float],
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    min_seg_len: int = 20,
    bf_threshold: float = 5.0,
    max_changes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    对原始数值序列执行贝叶斯分割，返回结构变化的分割点。

    返回：
    - change_points: 分割点索引列表（包含末尾 n）
    - bayes_factors: 接受的分割对应的对数贝叶斯因子
    - segments: 段落区间列表 (start, end)
    """
    res = detect_change_points(
        series=series,
        mu0=mu0,
        kappa0=kappa0,
        alpha0=alpha0,
        beta0=beta0,
        min_seg_len=min_seg_len,
        bf_threshold=bf_threshold,
        max_changes=max_changes,
    )
    return res


@mcp.tool()
def bsa_detect_from_file(
    file_path: str,
    delimiter: str = ",",
    column: int = 0,
    column_name: Optional[str] = None,
    skip_header: int = 0,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    min_seg_len: int = 20,
    bf_threshold: float = 5.0,
    max_changes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    从文件加载序列并执行贝叶斯分割。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    series = load_series_from_file(
        file_path,
        delimiter=delimiter,
        column=column,
        skip_header=skip_header,
        column_name=column_name,
    )
    res = detect_change_points(
        series=series,
        mu0=mu0,
        kappa0=kappa0,
        alpha0=alpha0,
        beta0=beta0,
        min_seg_len=min_seg_len,
        bf_threshold=bf_threshold,
        max_changes=max_changes,
    )
    res["n"] = int(len(series))
    return res


@mcp.tool()
def bsa_generate_synthetic(
    save_path: Optional[str] = None,
    n_samples: int = 1200,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    生成分段高斯测试数据，并写入CSV（列名 value）。
    返回数据路径与真实分割点，便于验证。
    """
    series, cps = generate_piecewise_gaussian(n_samples=n_samples, seed=seed)
    if save_path is None:
        save_path = os.path.abspath("synthetic_bsa.csv")
    save_series_csv(save_path, series)
    # Build segments from cps
    segs = []
    prev = 0
    for cp in cps:
        segs.append((prev, int(cp)))
        prev = int(cp)
    return {
        "path": save_path,
        "n": int(n_samples),
        "true_change_points": [int(c) for c in cps],
        "segments": segs,
    }


if __name__ == "__main__":
    # 与 deeplog_mcp_server 相同的传输方式
    mcp.run(transport="sse", port=2264)
