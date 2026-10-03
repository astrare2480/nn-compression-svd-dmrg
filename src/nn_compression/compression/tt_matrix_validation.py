"""TT-matrix（TT表現された行列・線形層）専用の入力 validation。

参照元:
    notebooks/30_tt_mps/00_fundamentals/14_tt_matrix_dense_reconstruction_and_tt_linear_forward.ipynb
    notebooks/30_tt_mps/00_fundamentals/15_ttlinear_dense_forward_equivalence_learning.ipynb

Notebook 14/15 の ``validate_tt_matrix_cores`` / ``dense_to_tt_matrix_tensor``
内の ``assert`` は、学習用に dtype を ``torch.float64`` 固定・device を CPU 固定
としていた。production 向けには、既存の TT API（``tt_validation.py``）と同じ
contract（``torch.float32`` / ``torch.float64`` の両方、CPU/CUDA の両方を許可）
へ一般化し、``assert`` はすべて ``TypeError`` / ``ValueError`` に置き換える。

通常の TT core（3階、``(r_{k-1}, n_k, r_k)``）の構造 contract は
``tt_validation.validate_tt_cores`` が既に持っているため、ここでは
TT-matrix core（4階、``(r_{k-1}, m_k, n_k, r_k)``）専用の contract だけを
独立して持つ。dtype の正式対応判定（``torch.float32`` / ``torch.float64``）は
``tt_validation.validate_tt_dtype`` を再利用し、重複実装しない。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .svd import coerce_integer_scalar
from .tt_validation import validate_tt_dtype


def validate_tt_matrix_modes(
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    name_out: str = "out_modes",
    name_in: str = "in_modes",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``out_modes`` / ``in_modes`` の基本 contract を検証し、``tuple[int, ...]`` へ正規化する。

    - 両方とも空でない
    - site 数（長さ）が一致する
    - 各 mode は bool ではない正の整数

    戻り値:
        ``(out_modes, in_modes)``。各要素を ``int`` へ変換した ``tuple``。
    """
    if (
        isinstance(out_modes, (str, bytes))
        or not isinstance(out_modes, Sequence)
    ):
        raise TypeError(
            f"{name_out} は正の整数の列である必要があります: {out_modes!r}"
        )
    if (
        isinstance(in_modes, (str, bytes))
        or not isinstance(in_modes, Sequence)
    ):
        raise TypeError(
            f"{name_in} は正の整数の列である必要があります: {in_modes!r}"
        )

    if len(out_modes) == 0 or len(in_modes) == 0:
        raise ValueError(
            f"{name_out} と {name_in} は空であってはいけません: "
            f"len({name_out})={len(out_modes)}, len({name_in})={len(in_modes)}"
        )

    if len(out_modes) != len(in_modes):
        raise ValueError(
            f"{name_out} と {name_in} の site 数が一致しません: "
            f"{len(out_modes)} != {len(in_modes)}"
        )

    out_modes_int = tuple(
        coerce_integer_scalar(mode, name=f"{name_out}[{index}]")
        for index, mode in enumerate(out_modes)
    )
    in_modes_int = tuple(
        coerce_integer_scalar(mode, name=f"{name_in}[{index}]")
        for index, mode in enumerate(in_modes)
    )

    for index, mode in enumerate(out_modes_int):
        if mode <= 0:
            raise ValueError(
                f"{name_out}[{index}] は正の整数である必要があります: {mode}"
            )
    for index, mode in enumerate(in_modes_int):
        if mode <= 0:
            raise ValueError(
                f"{name_in}[{index}] は正の整数である必要があります: {mode}"
            )

    return out_modes_int, in_modes_int


def validate_tt_matrix_cores(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int] | None = None,
    in_modes: Sequence[int] | None = None,
    *,
    name: str = "cores",
) -> None:
    """TT-matrix core 列の構造 contract を検証する。

    前提とする形式:

        G^{(1)}: (1, m_1, n_1, r_1), ..., G^{(d)}: (r_{d-1}, m_d, n_d, 1)

    最低限、以下を確認する。

    - ``cores`` が list または tuple で、空でない
    - 各要素が ``torch.Tensor``
    - 各 core が4階
    - 各 dimension が正の整数
    - 隣接 core 間で bond dimension（``shape[-1]`` と次の ``shape[0]``）が一致する
    - 左端 core の left bond（``shape[0]``）が1、右端 core の right bond
      （``shape[-1]``）が1
    - core 間で dtype / device が一致する
    - dtype は ``torch.float32`` / ``torch.float64`` のみ正式対応
      （``validate_tt_dtype`` を再利用）

    out_modes, in_modes:
        両方 ``None``（既定）の場合、``cores`` のみから構造を検証する
        （``tt_matrix_ranks`` / ``tt_matrix_num_parameters`` など、
        modes を引数に取らない関数向け）。

        両方指定した場合、追加で次を確認する。

        - ``len(out_modes) == len(in_modes) == len(cores)``
        - 各 mode が bool ではない正の整数
        - ``cores[k].shape[1] == out_modes[k]``
        - ``cores[k].shape[2] == in_modes[k]``

        片方だけ指定することは許さない（``ValueError``）。
    """
    if not isinstance(cores, (list, tuple)) or len(cores) == 0:
        raise ValueError(
            f"{name} は空でない list または tuple である必要があります: {cores!r}"
        )

    for index, core in enumerate(cores):
        if not torch.is_tensor(core):
            raise TypeError(
                f"{name}[{index}] は torch.Tensor である必要があります: "
                f"{type(core)!r}"
            )
        if core.ndim != 4:
            raise ValueError(
                f"{name}[{index}] は4階（r_left, m_k, n_k, r_right）である"
                f"必要があります: ndim={core.ndim}, shape={tuple(core.shape)}"
            )
        for dim_index, dim in enumerate(core.shape):
            if dim <= 0:
                raise ValueError(
                    f"{name}[{index}].shape[{dim_index}] は正の整数である"
                    f"必要があります: {dim}"
                )

    if cores[0].shape[0] != 1:
        raise ValueError(
            f"{name}[0] の left bond（shape[0]）は1である必要があります: "
            f"{cores[0].shape[0]}"
        )
    if cores[-1].shape[-1] != 1:
        raise ValueError(
            f"{name}[-1] の right bond（shape[-1]）は1である必要があります: "
            f"{cores[-1].shape[-1]}"
        )

    for index in range(len(cores) - 1):
        right_bond = cores[index].shape[-1]
        left_bond_next = cores[index + 1].shape[0]
        if right_bond != left_bond_next:
            raise ValueError(
                f"{name}[{index}] の right bond ({right_bond}) と "
                f"{name}[{index + 1}] の left bond ({left_bond_next}) が"
                "一致しません。"
            )

    dtype0 = cores[0].dtype
    device0 = cores[0].device
    for index, core in enumerate(cores[1:], start=1):
        if core.dtype != dtype0:
            raise TypeError(
                f"{name}[{index}].dtype={core.dtype} は "
                f"{name}[0].dtype={dtype0} と一致しません。"
            )
        if core.device != device0:
            raise ValueError(
                f"{name}[{index}].device={core.device} は "
                f"{name}[0].device={device0} と一致しません。"
            )

    validate_tt_dtype(cores[0], name=f"{name}[0]")

    if out_modes is None and in_modes is None:
        return
    if out_modes is None or in_modes is None:
        raise ValueError(
            "out_modes と in_modes は両方指定するか、両方省略してください。"
        )

    out_modes_t, in_modes_t = validate_tt_matrix_modes(out_modes, in_modes)
    if len(out_modes_t) != len(cores):
        raise ValueError(
            f"len(out_modes)={len(out_modes_t)} は len({name})={len(cores)} "
            "と一致する必要があります。"
        )

    for index, core in enumerate(cores):
        if core.shape[1] != out_modes_t[index]:
            raise ValueError(
                f"{name}[{index}].shape[1]（out mode）は "
                f"out_modes[{index}]={out_modes_t[index]} と一致する必要が"
                f"あります: {core.shape[1]}"
            )
        if core.shape[2] != in_modes_t[index]:
            raise ValueError(
                f"{name}[{index}].shape[2]（in mode）は "
                f"in_modes[{index}]={in_modes_t[index]} と一致する必要が"
                f"あります: {core.shape[2]}"
            )


def validate_dense_tt_matrix_weight(
    W: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    name: str = "W",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """dense weight ``W`` が TT-matrix 用の tensorization に使える contract を満たすか検証する。

    - ``W`` が ``torch.Tensor``
    - 2階である
    - shape が ``(product(out_modes), product(in_modes))``
    - dtype が ``torch.float32`` / ``torch.float64``（``validate_tt_dtype`` 再利用）
    - device は制限しない（CPU固定にしない）

    戻り値:
        ``validate_tt_matrix_modes`` と同じ、正規化済み ``(out_modes, in_modes)``。
    """
    if not torch.is_tensor(W):
        raise TypeError(f"{name} は torch.Tensor である必要があります: {type(W)!r}")
    if W.ndim != 2:
        raise ValueError(f"{name} は2階である必要があります: ndim={W.ndim}, shape={tuple(W.shape)}")

    out_modes_t, in_modes_t = validate_tt_matrix_modes(out_modes, in_modes)
    expected_shape = (
        math.prod(out_modes_t),
        math.prod(in_modes_t),
    )
    if tuple(W.shape) != expected_shape:
        raise ValueError(
            f"{name}.shape={tuple(W.shape)} は (product(out_modes), "
            f"product(in_modes))={expected_shape} と一致する必要があります。"
        )

    validate_tt_dtype(W, name=name)

    return out_modes_t, in_modes_t


def validate_tt_linear_input(
    x: torch.Tensor,
    cores: Sequence[torch.Tensor],
    in_modes: Sequence[int],
    *,
    name: str = "x",
) -> None:
    """``tt_linear_forward`` の入力 ``x`` が満たすべき contract を検証する。

    - ``x`` が ``torch.Tensor``
    - 2階（``(batch, in_features)``）のみ正式対応
    - 最終次元が ``product(in_modes)`` と一致する
    - dtype / device が ``cores[0]`` と一致する

    ``cores`` 自体の構造検証（``validate_tt_matrix_cores``）は呼び出し側の
    責務とし、ここでは行わない（``cores[0]`` の dtype/device 参照のみ使う）。
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} は torch.Tensor である必要があります: {type(x)!r}")
    if x.ndim != 2:
        raise ValueError(
            f"{name} は2階（(batch, in_features)）である必要があります"
            f"（今回は2階入力のみ正式対応）: ndim={x.ndim}, shape={tuple(x.shape)}"
        )

    expected_n = math.prod(int(n) for n in in_modes)
    if x.shape[-1] != expected_n:
        raise ValueError(
            f"{name}.shape[-1]={x.shape[-1]} は product(in_modes)={expected_n} "
            "と一致する必要があります。"
        )

    core0 = cores[0]
    if x.dtype != core0.dtype:
        raise TypeError(
            f"{name}.dtype={x.dtype} は cores[0].dtype={core0.dtype} と"
            "一致する必要があります。"
        )
    if x.device != core0.device:
        raise ValueError(
            f"{name}.device={x.device} は cores[0].device={core0.device} と"
            "一致する必要があります。"
        )


def validate_tt_linear_bias(
    bias: torch.Tensor | None,
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    *,
    name: str = "bias",
) -> None:
    """``tt_linear_forward`` の ``bias`` が満たすべき contract を検証する。

    - ``bias`` が ``None`` の場合は常に許可する
      （Notebook 14 の実装は ``bias is None`` でも ``bias.shape`` を参照して
      ``AttributeError`` になる不具合があったため、ここでは最初に ``None``
      チェックを行う）。
    - ``None`` でない場合、``torch.Tensor`` かつ shape が
      ``(product(out_modes),)``
    - dtype / device が ``cores[0]`` と一致する
    """
    if bias is None:
        return

    if not torch.is_tensor(bias):
        raise TypeError(f"{name} は torch.Tensor または None である必要があります: {type(bias)!r}")

    expected_m = math.prod(int(m) for m in out_modes)
    if tuple(bias.shape) != (expected_m,):
        raise ValueError(
            f"{name}.shape={tuple(bias.shape)} は (product(out_modes),)="
            f"({expected_m},) と一致する必要があります。"
        )

    core0 = cores[0]
    if bias.dtype != core0.dtype:
        raise TypeError(
            f"{name}.dtype={bias.dtype} は cores[0].dtype={core0.dtype} と"
            "一致する必要があります。"
        )
    if bias.device != core0.device:
        raise ValueError(
            f"{name}.device={bias.device} は cores[0].device={core0.device} と"
            "一致する必要があります。"
        )
