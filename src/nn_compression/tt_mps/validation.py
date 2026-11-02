"""TT-SVD 専用の入力 validation（アルゴリズム本体ではない）。

TT の内部 cut は ``1 <= k < d`` であり、汎用の mode index
（``nn_compression.tensor.validation.validate_mode_index`` の
``0 <= mode < ndim``）とは意味が異なる。Tucker/HOOI 専用の validation を
``tensor/validation.py`` に混ぜず ``compression/tucker_validation.py`` に
分離しているのと同じ理由で、TT 専用の contract もここに置く。

dtype の contract は Tucker/HOOI とは異なり、TT-SVD 内部の
``torch.linalg.matrix_rank`` / ``torch.linalg.svd`` が CPU 環境で
``float16`` / ``bfloat16`` を扱えない可能性があるため、
``validate_tt_dtype`` で ``float32`` / ``float64`` のみを正式対応とする。
``tucker_validation.validate_real_dtype`` 自体は変更しない。

参照元:
    notebooks/30_tt_mps/00_fundamentals/01_tt_rank_unfolding.ipynb
"""

from __future__ import annotations

import torch

_TT_SUPPORTED_DTYPES = (torch.float32, torch.float64)


def validate_tt_dtype(tensor: torch.Tensor, *, name: str = "X") -> None:
    """TT-SVD が正式対応する dtype であることを確認する。

    正式対応は ``torch.float32`` と ``torch.float64`` のみ。
    ``float16`` / ``bfloat16``、整数、bool、複素 dtype は public API
    入口で ``TypeError`` として拒否する。
    """
    if tensor.is_complex():
        raise TypeError(
            f"{name} は実数 dtype である必要があります（複素数は未対応）: "
            f"dtype={tensor.dtype}"
        )
    if not tensor.is_floating_point():
        raise TypeError(
            f"{name} は実数浮動小数点 dtype である必要があります: "
            f"dtype={tensor.dtype}"
        )
    if tensor.dtype not in _TT_SUPPORTED_DTYPES:
        raise TypeError(
            f"{name} の dtype は torch.float32 または torch.float64 "
            f"である必要があります: dtype={tensor.dtype}"
        )


def validate_tt_cut_index(mode: int, ndim: int, *, name: str = "mode") -> None:
    """TT cut index の contract: ``1 <= mode < ndim``。

    TT の cut ``1:k | k+1:d`` は ``k=0``（左側が空）や ``k=d``（右側が空）
    を許さない。bool は整数のサブクラスだが意図しない誤用を避けるため
    明示的に拒否する。
    """
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise TypeError(
            f"{name} は bool 以外の整数である必要があります: {mode!r}"
        )
    if not 1 <= mode < ndim:
        raise ValueError(
            f"{name}={mode} は 1〜{ndim - 1} の範囲で指定してください。"
        )


def validate_tt_cores(cores: list[torch.Tensor], *, name: str = "cores") -> None:
    """TT core 列の構造 contract を検証する。

    ``tt_reconstruct`` / ``tt_num_parameters`` が前提とする TT 形式:

        G_1: (1, n_1, r_1), G_2: (r_1, n_2, r_2), ..., G_d: (r_{d-1}, n_d, 1)

    最低限、以下を確認する。

    - core 列が空でない
    - 各 core が ``torch.Tensor`` かつ 3 階
    - 各 dimension が正の整数
    - 左端 core の left bond が 1、右端 core の right bond が 1
    - 隣接 core 間で bond dimension が一致する
    - core 間で dtype / device が一致する
    """
    if not isinstance(cores, (list, tuple)) or len(cores) == 0:
        raise ValueError(
            f"{name} は空でない list である必要があります: {cores!r}"
        )

    for index, core in enumerate(cores):
        if not torch.is_tensor(core):
            raise TypeError(
                f"{name}[{index}] は torch.Tensor である必要があります: "
                f"{type(core)!r}"
            )
        if core.ndim != 3:
            raise ValueError(
                f"{name}[{index}] は3階である必要があります: ndim={core.ndim}"
            )
        for dim_index, dim in enumerate(core.shape):
            if dim <= 0:
                raise ValueError(
                    f"{name}[{index}].shape[{dim_index}] は正の整数である"
                    f"必要があります: {dim}"
                )

    if cores[0].shape[0] != 1:
        raise ValueError(
            f"{name}[0] の left bond は1である必要があります: "
            f"{cores[0].shape[0]}"
        )
    if cores[-1].shape[-1] != 1:
        raise ValueError(
            f"{name}[-1] の right bond は1である必要があります: "
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


def validate_tt_pair(
    cores_a: list[torch.Tensor],
    cores_b: list[torch.Tensor],
    *,
    name_a: str = "cores_a",
    name_b: str = "cores_b",
) -> None:
    """2つの TT core列が、内積・距離などの2項演算に使える contract を満たすか検証する。

    ``tt_inner`` / ``tt_distance`` などが前提とする条件:

    - order（core の個数、TT の階数 ``d``）が一致する
    - 各 site の物理次元 ``n_k`` が一致する
    - dtype が一致する
    - device が一致する

    TT-rank（各 core の bond dimension）は A/B で異なっていてよい。
    ``validate_tt_cores`` による各TT単独の structure 検証は、この関数の
    前提として呼び出し側が別途行う。
    """
    if len(cores_a) != len(cores_b):
        raise ValueError(
            f"{name_a} と {name_b} の order が一致しません: "
            f"{len(cores_a)} != {len(cores_b)}"
        )

    for index, (core_a, core_b) in enumerate(zip(cores_a, cores_b)):
        if core_a.shape[1] != core_b.shape[1]:
            raise ValueError(
                f"{name_a}[{index}] と {name_b}[{index}] の物理次元が"
                f"一致しません: {core_a.shape[1]} != {core_b.shape[1]}"
            )

    if cores_a[0].dtype != cores_b[0].dtype:
        raise TypeError(
            f"{name_a}.dtype={cores_a[0].dtype} は "
            f"{name_b}.dtype={cores_b[0].dtype} と一致しません。"
        )
    if cores_a[0].device != cores_b[0].device:
        raise ValueError(
            f"{name_a}.device={cores_a[0].device} は "
            f"{name_b}.device={cores_b[0].device} と一致しません。"
        )
