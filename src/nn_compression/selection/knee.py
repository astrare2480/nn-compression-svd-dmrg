"""Pareto frontier 上の knee 点を求める既存関数。

参照元:
    notebooks/20_fashion_mnist/mlp/02_mlp_svd_rank_selection.ipynb
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _require_xy_frame(df, x, y) -> pd.DataFrame:
    """空・NaN・inf を弾いた DataFrame を返す。"""
    if df is None or len(df) == 0:
        raise ValueError("knee 入力が空です。")
    if x not in df.columns or y not in df.columns:
        raise KeyError(f"列 {x!r} / {y!r} が DataFrame にありません。")

    xs = pd.to_numeric(df[x], errors="coerce")
    ys = pd.to_numeric(df[y], errors="coerce")
    if xs.isna().any() or ys.isna().any():
        raise ValueError("knee 入力に NaN があります。")

    x_values = xs.to_numpy(dtype=float)
    y_values = ys.to_numpy(dtype=float)
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("knee 入力に inf があります。")

    out = df.copy()
    out[x] = x_values
    out[y] = y_values
    return out


def get_first_point(df, column, x, y):
    """column で並べた先頭点の座標ペアを返す。

    元Notebookの ``GetPoint`` を、Pythonの命名規則に合わせて改名した。
    """
    frame = _require_xy_frame(df, x, y)
    if column not in frame.columns:
        raise KeyError(f"列 {column!r} が DataFrame にありません。")
    df_sorted = frame.sort_values(column).reset_index(drop=True)
    return (float(df_sorted[x].iloc[0]), float(df_sorted[y].iloc[0]))


def get_endpoints(df, column, x, y):
    """column で並べた先頭点と末尾点の座標ペアを返す。"""
    frame = _require_xy_frame(df, x, y)
    if column not in frame.columns:
        raise KeyError(f"列 {column!r} が DataFrame にありません。")
    df_sorted = frame.sort_values(column).reset_index(drop=True)

    left = df_sorted.iloc[0]
    right = df_sorted.iloc[-1]
    return (
        (float(left[x]), float(left[y])),
        (float(right[x]), float(right[y])),
    )


def line_equation(p0, p1):
    """2点を通る直線の傾き・切片を返す。

    垂直線（x が同じ）は ``(inf, x0)`` を返す。切片スロットに x 定数を入れる。
    同一点なら傾き 0・切片 y0（水平な退化線）とする。
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    for value in (x0, y0, x1, y1):
        if not math.isfinite(value):
            raise ValueError("直線の端点に NaN / inf があります。")

    if x0 == x1 and y0 == y1:
        return 0.0, y0
    if x0 == x1:
        return math.inf, x0

    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - x0 * slope
    return slope, intercept


def _point_line_distance(x_value, y_value, slope, intercept) -> float:
    if math.isinf(slope):
        return abs(x_value - intercept)
    denom = math.sqrt(slope**2 + 1.0)
    if denom == 0.0:
        raise ValueError("直線の正規化定数が 0 です。")
    return abs(slope * x_value - y_value + intercept) / denom


def find_knee_point(df, slope, intercept, x, y):
    """端点間の直線から最も遠い点をループで求める。

    1 点しか無い場合はその点を knee、距離 0 として返す。
    全点が直線上（距離 0）なら先頭点を返す。
    ``(None, None)`` は返さない。
    """
    frame = _require_xy_frame(df, x, y)
    if not math.isfinite(intercept) or (
        not math.isfinite(slope) and not math.isinf(slope)
    ):
        raise ValueError("直線パラメータに NaN / inf があります。")

    if len(frame) == 1:
        row = frame.iloc[0]
        return (float(row[x]), float(row[y])), 0.0

    best_dist = -1.0
    best_x = float(frame.iloc[0][x])
    best_y = float(frame.iloc[0][y])

    for _, row in frame.iterrows():
        dist = _point_line_distance(float(row[x]), float(row[y]), slope, intercept)
        if dist > best_dist:
            best_dist = dist
            best_x, best_y = float(row[x]), float(row[y])

    if best_dist < 0.0:
        raise ValueError("knee 点を決められませんでした。")
    return (best_x, best_y), float(best_dist)


def find_knee_point_numpy(df, slope, intercept, x, y):
    """端点間の直線から最も遠い点を NumPy で求める高速版。"""
    frame = _require_xy_frame(df, x, y)
    if not math.isfinite(intercept) or (
        not math.isfinite(slope) and not math.isinf(slope)
    ):
        raise ValueError("直線パラメータに NaN / inf があります。")

    xs = frame[x].to_numpy(dtype=float)
    ys = frame[y].to_numpy(dtype=float)
    if len(frame) == 1:
        return (float(xs[0]), float(ys[0])), 0.0

    if math.isinf(slope):
        dist = np.abs(xs - intercept)
    else:
        dist = np.abs(slope * xs - ys + intercept) / math.sqrt(slope**2 + 1.0)

    if not np.isfinite(dist).all():
        raise ValueError("knee 距離に NaN / inf があります。")

    knee_idx = int(np.argmax(dist))
    return (float(xs[knee_idx]), float(ys[knee_idx])), float(dist[knee_idx])
