"""Pareto frontier 上の knee 点を求める既存関数。

参照元:
    notebooks/20_fashion_mnist/mlp/02_mlp_svd_rank_selection.ipynb
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
"""

import numpy as np


def get_first_point(df, column, x, y):
    """column で並べた先頭点の座標ペアを返す。

    元Notebookの ``GetPoint`` を、Pythonの命名規則に合わせて改名した。
    """
    df_sorted = df.sort_values(column).reset_index(drop=True)
    point = (df_sorted[x].iloc[0], df_sorted[y].iloc[0])

    return point


def get_endpoints(df, column, x, y):
    """column で並べた先頭点と末尾点の座標ペアを返す。"""
    df_sorted = df.sort_values(column).reset_index(drop=True)

    left = df_sorted.iloc[0]
    right = df_sorted.iloc[-1]

    point_start = (left[x], left[y])
    point_end = (right[x], right[y])

    return point_start, point_end


def line_equation(p0, p1):
    """2点を通る直線の傾き・切片を返す。

    Pareto曲線の端点を結ぶ基準線を作り、その線から最も遠い候補を
    knee とする元Notebookの手順に用いる。
    """
    x0, y0 = p0
    x1, y1 = p1
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - x0 * slope
    return slope, intercept


def find_knee_point(df, slope, intercept, x, y):
    """端点間の直線から最も遠い点をループで求める。

    各点と ``y = slope * x + intercept`` の垂直距離を計算し、最大の点を
    knee とみなす。
    """
    best_dist = -1.0
    best_x, best_y = None, None
    denom = (slope**2 + 1) ** 0.5

    for _, row in df.iterrows():
        dist = abs(slope * row[x] - row[y] + intercept) / denom
        if dist > best_dist:
            best_dist = dist
            best_x, best_y = row[x], row[y]

    return (best_x, best_y), best_dist


def find_knee_point_numpy(df, slope, intercept, x, y):
    """端点間の直線から最も遠い点を NumPy で求める高速版。"""
    xs = df[x].to_numpy(dtype=float)
    ys = df[y].to_numpy(dtype=float)

    dist = np.abs(
        slope * xs - ys + intercept
    ) / np.sqrt(slope**2 + 1)

    knee_idx = int(np.argmax(dist))
    knee_row = df.iloc[knee_idx]
    return (knee_row[x], knee_row[y]), float(dist[knee_idx])
