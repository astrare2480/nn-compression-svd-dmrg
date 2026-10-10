"""TT-matrix の tensorization（mode 分解）候補を列挙する汎用 helper。

参照元:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/00_mlp_ttlinear_logits_equivalence.ipynb

TT-matrix では ``out_features = prod(out_modes)``、``in_features = prod(in_modes)``
を満たす mode 列を選ぶ。mode の並び順が変わると TT の cut（どの軸で切るか）と
各 bond の rank が変わるため、順序違いの候補を同一視しない。

ここには「候補の列挙」と、候補の数値評価に使う小さな pure function だけを置く。

- ``ordered_factorizations``: 順序付き因数分解候補の列挙。
- ``tt_matrix_physical_sizes``: 各 site の physical size ``m_k * n_k``。
- ``tt_matrix_shape_imbalance``: physical size の ``max / min``（候補評価用の指標）。
- ``tt_matrix_num_parameters_from_ranks``: core を作る前に、modes と rank から
  TT-matrix の parameter 数 ``sum_k r_{k-1} m_k n_k r_k`` を計算する。
  生成済み core に対する ``tt_matrix_num_parameters(cores)`` と同じ値になる。

次は意図的に置かない。最終選択は呼び出し側の実験判断として残す。

- 推奨 tensorization の自動選択（``balanced_modes`` 等）:
  site 間の shape balance が accuracy や最適 rank を保証しないため。
  ``tt_matrix_shape_imbalance`` は「評価値を返す」だけで、候補を選ばない。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .svd import coerce_integer_scalar
from .tt_matrix_validation import validate_tt_matrix_modes


def _positive_int(value, *, name: str) -> int:
    """bool / 非整数を ``TypeError``、1 未満を ``ValueError`` で拒否する。"""
    value_int = coerce_integer_scalar(value, name=name)
    if value_int < 1:
        raise ValueError(f"{name} は 1 以上の整数である必要があります: {value_int}")
    return value_int


def _divisors_at_least(value: int, lower_bound: int) -> list[int]:
    """``value`` の約数のうち ``lower_bound`` 以上のものを昇順で返す。"""
    small: list[int] = []
    large: list[int] = []
    for divisor in range(1, math.isqrt(value) + 1):
        if value % divisor == 0:
            small.append(divisor)
            paired = value // divisor
            if paired != divisor:
                large.append(paired)
    return [d for d in small + large[::-1] if d >= lower_bound]


def ordered_factorizations(
    value: int,
    num_factors: int,
    min_factor: int = 2,
) -> list[tuple[int, ...]]:
    """``value`` を ``num_factors`` 個の整数の「順序付き積」へ分解した全候補を返す。

    value:
        分解する正の整数（例: ``out_features`` / ``in_features``）。
    num_factors:
        因子の個数（TT-matrix の site 数 ``d``）。1 以上。
    min_factor:
        各因子の下限。既定は 2（1 を含む退化した mode を作らない）。1 以上。

    戻り値:
        各要素が ``num_factors`` 個の整数からなる ``tuple`` の ``list``。

        - 各候補の積は ``value`` と一致する。
        - 各因子は ``min_factor`` 以上。
        - 順序違いは別候補として残す（``(2, 6)`` と ``(6, 2)`` は両方返す）。
        - 並びは決定的で、先頭の因子から順に辞書式の昇順。
        - 条件を満たす分解が存在しない場合は例外ではなく空 ``list`` を返す
          （例: ``ordered_factorizations(7, 2)``、``ordered_factorizations(1, 1)``）。
        - ``num_factors=1`` では、``value >= min_factor`` のとき ``[(value,)]``、
          そうでなければ ``[]``。

    例外:
        ``value`` / ``num_factors`` / ``min_factor`` が bool・非整数なら
        ``TypeError``、1 未満なら ``ValueError``。

    Note:
        候補数は ``value`` と ``num_factors`` に対して急速に増えうる。
        この関数は列挙のみを行い、評価・並べ替え・選択はしない。

        探索は再帰を使わない反復的な深さ優先探索なので、``num_factors`` が
        大きくても Python の再帰上限には依存しない
        （例: ``ordered_factorizations(1, 1000, min_factor=1)``）。
    """
    value = _positive_int(value, name="value")
    num_factors = _positive_int(num_factors, name="num_factors")
    min_factor = _positive_int(min_factor, name="min_factor")

    if num_factors == 1:
        return [(value,)] if value >= min_factor else []

    # min_factor >= 2 のとき、積が min_factor**num_factors >= 2**num_factors
    # 以上になるため、num_factors が value.bit_length() を超えれば候補は存在しない。
    # 巨大な冪 ``min_factor ** k`` や約数列挙を始める前に打ち切る。
    if min_factor >= 2 and num_factors > value.bit_length():
        return []

    # 残り k 個の因子を min_factor 以上にするには、残りの積が min_factor**k 以上
    # である必要がある（枝刈り用）。
    min_products = [min_factor**k for k in range(num_factors)]

    # 再帰を使わない深さ優先探索。再帰版と同じく、各段で約数を昇順に試すため
    # 候補の生成順（先頭因子から辞書式の昇順）は変わらない。
    #
    # level L のフレーム: ``remainders[L]`` を、prefix の L 個の因子を選んだ
    # 後の残りの積とし、``frames[L]`` はその段で試す約数の iterator。
    # 最後の 1 因子は残りの積そのものなので、探索するのは num_factors - 1 段。
    results: list[tuple[int, ...]] = []
    prefix: list[int] = []
    remainders: list[int] = [value]
    frames = [iter(_divisors_at_least(value, min_factor))]
    last_level = num_factors - 2  # 約数を選ぶ最後の段（0 始まり）

    while frames:
        level = len(frames) - 1
        factor = next(frames[-1], None)
        if factor is None:
            frames.pop()
            remainders.pop()
            if prefix:
                prefix.pop()
            continue

        next_remaining = remainders[level] // factor
        factors_left_after = num_factors - (level + 1)
        if next_remaining < min_products[factors_left_after]:
            continue

        if level == last_level:
            # 残り 1 因子 = next_remaining（上の枝刈りで min_factor 以上が保証済み）
            results.append((*prefix, factor, next_remaining))
        else:
            prefix.append(factor)
            remainders.append(next_remaining)
            frames.append(iter(_divisors_at_least(next_remaining, min_factor)))

    return results


def tt_matrix_physical_sizes(
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> tuple[int, ...]:
    """TT-matrix の各 site の physical size ``q_k = m_k * n_k`` を返す。

    これは dense weight を ``dense_to_tt_matrix_tensor`` で並べ替えたとき、
    TT-SVD が扱う通常の TT テンソルの各軸の長さに一致する。

    out_modes, in_modes:
        出力側・入力側の mode 列。空でなく、長さが等しく、各要素が bool ではない
        正の整数（``validate_tt_matrix_modes`` と同じ contract）。

    戻り値:
        長さ ``d`` の ``tuple[int, ...]``。
    """
    out_modes_t, in_modes_t = validate_tt_matrix_modes(out_modes, in_modes)
    return tuple(m_k * n_k for m_k, n_k in zip(out_modes_t, in_modes_t))


def tt_matrix_shape_imbalance(
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> float:
    """physical size の ``max / min`` を返す（mode 候補の評価用）。

    1.0 に近いほど site 間の physical size が揃っている。これは候補を比較する
    ための数値であり、小さいほど accuracy や圧縮率が良いことを意味しない。
    最も均等な候補を自動で採用する関数はここには置かない。

    引数と contract は ``tt_matrix_physical_sizes`` と同じ。
    """
    sizes = tt_matrix_physical_sizes(out_modes, in_modes)
    return max(sizes) / min(sizes)


def tt_matrix_num_parameters_from_ranks(
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    ranks: Sequence[int],
) -> int:
    """core を作る前に TT-matrix の parameter 数 ``sum_k r_{k-1} m_k n_k r_k`` を返す。

    out_modes, in_modes:
        出力側・入力側の mode 列（``tt_matrix_physical_sizes`` と同じ contract）。
    ranks:
        境界 rank を含む TT-rank ``(r_0, r_1, ..., r_d)``。長さは ``d + 1``、
        各要素は bool ではない正の整数、境界 ``r_0 = r_d = 1``。
        整数の扱いは既存 API と同じ（NumPy 整数を受け付け、bool・float は拒否）。

    戻り値:
        Python の ``int``。生成済み core に対する
        ``tt_matrix_num_parameters(cores)`` と一致する。

    例外:
        ranks が列でない・要素が整数でない（bool 含む）なら ``TypeError``。
        長さの不一致・非正の rank・境界 rank が 1 でない場合は ``ValueError``。
    """
    out_modes_t, in_modes_t = validate_tt_matrix_modes(out_modes, in_modes)
    d = len(out_modes_t)

    if isinstance(ranks, (str, bytes)) or not isinstance(ranks, Sequence):
        raise TypeError(f"ranks は整数の列である必要があります: {ranks!r}")
    if len(ranks) != d + 1:
        raise ValueError(
            f"len(ranks)={len(ranks)} は len(out_modes)+1={d + 1} と一致する必要があります。"
        )

    ranks_t = tuple(
        _positive_int(rank, name=f"ranks[{index}]") for index, rank in enumerate(ranks)
    )
    if ranks_t[0] != 1 or ranks_t[-1] != 1:
        raise ValueError(
            f"境界 rank（ranks[0] と ranks[-1]）は1である必要があります: {ranks_t}"
        )

    return sum(
        ranks_t[k] * out_modes_t[k] * in_modes_t[k] * ranks_t[k + 1] for k in range(d)
    )
