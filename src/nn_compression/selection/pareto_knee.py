r"""Pareto frontier から knee 点を安全に選ぶ。

参照元:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/09_*.ipynb
    notebooks/30_tt_mps/20_fashion_mnist_cnn/03_pretrained_cnn_conv2_tt_rank_sweep.ipynb

既存の ``find_knee_point`` 系は「端点間の直線から最も遠い点」を必ず返す。
そのため、点が2つしかない・全点が直線上にある・最大距離が同率・距離が最大の点が
性能の悪い側にある、といった場合にも「knee」を返してしまう。ここでは

1. 2目的とも小さいほど良いものとして Pareto frontier を取り、
2. frontier 上の x / y をそれぞれ ``[0, 1]`` へ正規化し、
3. 端点（正規化後は ``(0, 1)`` と ``(1, 0)``）を結ぶ直線から最も遠い内部点を候補にし、
4. 候補が信頼できないときは knee を返さず **未定義理由** を返す。

既存の ``extract_pareto_frontier`` / ``get_endpoints`` / ``line_equation`` /
``find_knee_point_numpy`` を再利用する。どの点を採用するか（前後何点を使うか、
どう命名するか）、結果の保存・図示は Notebook の責任で、ここでは扱わない。
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Hashable

import numpy as np
import pandas as pd

from .knee import _require_xy_frame, find_knee_point_numpy, get_endpoints, line_equation
from .pareto import extract_pareto_frontier

# 未定義理由の文字列（``ParetoKneeSelection.undefined_reason`` の取りうる値）。
_REASON_TOO_FEW_POINTS = "too_few_points"
_REASON_ZERO_X_WIDTH = "zero_x_width"
_REASON_ZERO_Y_WIDTH = "zero_y_width"
_REASON_NEARLY_COLLINEAR = "nearly_collinear"
_REASON_UNDESIRABLE_SIDE = "undesirable_side"
_REASON_AMBIGUOUS_TIE = "ambiguous_tie"


@dataclass(frozen=True, eq=False)
class ParetoKneeSelection:
    """``select_pareto_knee`` の結果。

    - ``frontier``: Pareto frontier（元の列・dtype のまま、x 昇順、index は 0 からの連番）。
      ``extract_pareto_frontier`` と同じ行集合。
    - ``frontier_source_index``: ``frontier`` の各行に対応する、入力 DataFrame 上の
      index ラベルの tuple（``frontier`` と同じ順序・長さ）。
    - ``is_defined``: knee を定義できたか。
    - ``knee_index``: knee の入力 DataFrame 上の index ラベル（未定義なら ``None``）。
    - ``knee_row``: knee の行（入力 DataFrame の行のコピー。未定義なら ``None``）。
    - ``knee_distance``: 正規化座標（``[0, 1]^2``）での、端点間の直線から knee までの距離
      （未定義なら ``None``）。
    - ``undefined_reason``: 未定義の理由（定義できたなら ``None``）。値は次のどれか。

      * ``"too_few_points"``: 重複を除いた Pareto frontier の点が2つ以下。
      * ``"zero_x_width"`` / ``"zero_y_width"``: その frontier 座標の x（または y）の幅が 0。
        入力全体の幅では判定しない。
      * ``"nearly_collinear"``: 内部点がすべて端点間の直線から ``tolerance`` 以内。
      * ``"undesirable_side"``: 最も遠い内部点が、2目的の最小化にとって望ましくない側
        （直線より原点から遠い側）にある。
      * ``"ambiguous_tie"``: 最も遠い内部点が ``tolerance`` 以内の差で複数ある。
    """

    frontier: pd.DataFrame
    frontier_source_index: tuple
    is_defined: bool
    knee_index: Hashable | None
    knee_row: pd.Series | None
    knee_distance: float | None
    undefined_reason: str | None


def _validate_tolerance(tolerance) -> float:
    """``tolerance`` が有限で 0 以上の実数（bool 以外）であることを確かめて ``float`` にする。"""
    if isinstance(tolerance, bool) or not isinstance(tolerance, numbers.Real):
        raise TypeError(f"tolerance は実数である必要があります: {tolerance!r} ({type(tolerance)!r})")
    value = float(tolerance)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"tolerance は有限で 0 以上である必要があります: {tolerance!r}")
    return value


def _unit_interval(values: np.ndarray, *, lower: float, upper: float) -> np.ndarray:
    """``[lower, upper]`` を ``[0, 1]`` へ写す。2 のべき乗で先にスケールして overflow を避ける。"""
    largest = max(abs(lower), abs(upper))
    scale = math.ldexp(1.0, math.frexp(largest)[1] - 1) if largest > 0.0 else 1.0
    scaled_lower, scaled_upper = lower / scale, upper / scale
    return (values / scale - scaled_lower) / (scaled_upper - scaled_lower)


def _undefined(
    frontier: pd.DataFrame,
    source_index: tuple,
    reason: str,
) -> ParetoKneeSelection:
    """knee を持たない結果を作る。"""
    return ParetoKneeSelection(
        frontier=frontier,
        frontier_source_index=source_index,
        is_defined=False,
        knee_index=None,
        knee_row=None,
        knee_distance=None,
        undefined_reason=reason,
    )


def select_pareto_knee(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    tolerance: float = 1e-9,
) -> ParetoKneeSelection:
    """2目的（どちらも小さいほど良い）の候補表から、Pareto frontier の knee を選ぶ。

    frame:
        候補の表（例: ``parameters`` と ``validation_loss`` の列を持つ DataFrame）。
        変更しない。
    x_column / y_column:
        最小化する2つの数値列。同じ列は指定できない。
    tolerance:
        正規化座標での距離の許容誤差（有限・0 以上、既定 ``1e-9``）。
        「ほぼ直線」と「最大距離の同率」の判定に使う。

    戻り値:
        ``ParetoKneeSelection``。knee が定義できたときだけ ``is_defined=True`` で、
        ``knee_index`` / ``knee_row`` / ``knee_distance`` が入る。定義できないときは
        誤った knee を返さず、``is_defined=False`` と ``undefined_reason`` を返す
        （frontier は常に返す）。

    判定の手順と未定義理由:
        1. Pareto frontier を作る。
        2. frontier 上の完全に同じ座標を重複除去する。
        3. 異なる点が2つ以下なら ``too_few_points``。入力全体で x や y が同じでも、
           frontier が1点に潰れるならこちら（幅の理由にはしない）。
        4. 手順2の座標列で x の幅が 0 なら ``zero_x_width``、y の幅が 0 なら
           ``zero_y_width``。幅は入力全体ではなく、正規化する frontier 座標で見る。
           2目的の最小化では、同じ x（または y）の点は支配されるので、異なる点が
           3つ以上ある frontier で幅が 0 になることは通常ない。
        5. その同じ座標列の x / y をそれぞれ ``[0, 1]`` へ正規化する。frontier は x が
           単調増加・y が単調減少するので、端点は ``(0, 1)`` と ``(1, 0)`` になる。
        6. 端点間の直線から最も遠い点を候補にする。端点は直線上で距離 0 なので、
           候補が端点になるのは全点が直線上のときだけで、その場合は手順7で
           ``nearly_collinear`` になる（端点を knee として返すことはない）。
        7. 候補の距離が ``tolerance`` 以下なら ``nearly_collinear``、候補が直線より
           原点から遠い側にあれば ``undesirable_side``、候補を除いた点の最大距離との
           差が ``tolerance`` 以下なら ``ambiguous_tie``。

    例外（入力そのものの誤り。未定義理由にはしない）:
        frame が DataFrame でない・tolerance が数でない場合は ``TypeError``。
        空の frame、x / y に NaN・inf、同じ列の指定、不正な tolerance は ``ValueError``。
        列が存在しない場合は ``KeyError``。

    Note:
        - 座標が完全に同じ行は同じ点として扱い、knee は入力で最初に現れる行を返す。
        - 行の並びを入れ替えても、frontier の行集合と knee の座標は変わらない
          （完全に同じ座標の行が複数ある場合の ``knee_index`` を除く）。
        - 「knee 前後の点を自動採用する」「rank や予算を決め打ちする」といった
          実験方針や、CSV 保存・図示は行わない。
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame は pandas.DataFrame である必要があります: {type(frame)!r}")
    if x_column == y_column:
        raise ValueError(f"x_column と y_column は異なる列である必要があります: {x_column!r}")
    tolerance = _validate_tolerance(tolerance)

    # 空・NaN・inf を弾き、x / y を float にした作業用コピーを得る（frame は変えない）。
    # index が重複していても ``extract_pareto_frontier`` の ``df.loc`` が行を重複取得しないよう、
    # 作業用コピーは 0 からの連番 index にする（元の index は位置から ``frame.index`` で戻す）。
    work = _require_xy_frame(frame, x_column, y_column).reset_index(drop=True)

    # 元の行へ戻すため、位置（0 始まり）を列として持たせる。既存の列名と衝突しない名前にする。
    position_column = "__pareto_source_position__"
    while position_column in work.columns:
        position_column += "_"
    work[position_column] = np.arange(len(work))

    # --- Pareto frontier（既存関数を再利用）。同点の並びが不定にならないよう安定に並べ直す。
    pareto = extract_pareto_frontier(work, x_column, y_column)
    order = np.lexsort(
        (
            pareto[position_column].to_numpy(),
            pareto[y_column].to_numpy(dtype=float),
            pareto[x_column].to_numpy(dtype=float),
        )
    )
    frontier_positions = pareto[position_column].to_numpy()[order]
    frontier = frame.iloc[frontier_positions].reset_index(drop=True)
    source_index = tuple(frame.index[frontier_positions])

    # --- 座標が完全に同じ点は1つにまとめる（先頭 = 入力で最初の行を代表にする）。
    points = pd.DataFrame(
        {
            x_column: work[x_column].to_numpy(dtype=float)[frontier_positions],
            y_column: work[y_column].to_numpy(dtype=float)[frontier_positions],
            position_column: frontier_positions,
        }
    ).drop_duplicates(subset=[x_column, y_column], keep="first")
    if len(points) <= 2:
        return _undefined(frontier, source_index, _REASON_TOO_FEW_POINTS)

    # 幅は、このあと正規化する frontier 座標で判定する（入力全体ではない）。
    xs = points[x_column].to_numpy(dtype=float)
    ys = points[y_column].to_numpy(dtype=float)
    if float(xs.max()) == float(xs.min()):
        return _undefined(frontier, source_index, _REASON_ZERO_X_WIDTH)
    if float(ys.max()) == float(ys.min()):
        return _undefined(frontier, source_index, _REASON_ZERO_Y_WIDTH)
    normalized = points.copy()
    normalized[x_column] = _unit_interval(xs, lower=float(xs[0]), upper=float(xs[-1]))
    normalized[y_column] = _unit_interval(ys, lower=float(ys[-1]), upper=float(ys[0]))

    # --- 端点間の直線と、直線から最も遠い点（既存関数を再利用）。
    # 端点は直線上（距離 0）なので、距離が tolerance を超える候補は必ず内部点になる。
    # find_knee_point_numpy は1行だけの frame を「距離 0 の knee」として返すため、
    # 内部点だけでなく端点を含めた frame（常に3行以上）を渡す。
    start, end = get_endpoints(normalized, x_column, x_column, y_column)
    slope, intercept = line_equation(start, end)
    (knee_x, knee_y), distance = find_knee_point_numpy(normalized, slope, intercept, x_column, y_column)

    if distance <= tolerance:
        return _undefined(frontier, source_index, _REASON_NEARLY_COLLINEAR)

    # 直線より下（原点側）なら望ましい側。最小化では上側の点は knee として採用しない。
    if (slope * knee_x + intercept) - knee_y < 0.0:
        return _undefined(frontier, source_index, _REASON_UNDESIRABLE_SIDE)

    # 候補を除いた点（端点を含む。2行以上残る）の最大距離と近ければ、knee を1つに決められない。
    is_candidate = (normalized[x_column].to_numpy() == knee_x) & (normalized[y_column].to_numpy() == knee_y)
    _, second_distance = find_knee_point_numpy(
        normalized.loc[~is_candidate], slope, intercept, x_column, y_column
    )
    if distance - second_distance <= tolerance:
        return _undefined(frontier, source_index, _REASON_AMBIGUOUS_TIE)

    knee_position = int(normalized.loc[is_candidate, position_column].iloc[0])
    return ParetoKneeSelection(
        frontier=frontier,
        frontier_source_index=source_index,
        is_defined=True,
        knee_index=frame.index[knee_position],
        knee_row=frame.iloc[knee_position].copy(),
        knee_distance=float(distance),
        undefined_reason=None,
    )
