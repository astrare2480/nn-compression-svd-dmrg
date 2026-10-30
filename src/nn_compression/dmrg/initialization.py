r"""実験支援: DMRG の初期 MPS を作る。

参照元:
    notebooks/40_dmrg/00_fundamentals/02_environment_two_site_dmrg_sweep_exercises.ipynb
    （``SWEEP_INITIAL_CORES`` を作っていたセル）

DMRG 本体（environment / 局所更新 / sweep / 収束判定）はこの module に依存しない。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..compression.tt_validation import validate_tt_dtype
from .validation import _validate_positive_int


def _canonical_torch_device(device: torch.device) -> torch.device:
    """index を補った device。``cuda`` は現在の CUDA device index を付ける。"""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        return torch.device("cuda", index=torch.cuda.current_device())
    return resolved


def make_random_mps(
    physical_dimensions: Sequence[int],
    max_bond: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device | str,
) -> list[torch.Tensor]:
    r"""開境界のランダム MPS core 列を作る。

    physical_dimensions:
        各 site の物理次元 ``(n_1, ..., n_d)``。site ごとに異なってよく、2 に限らない。
        各要素は 1 以上の整数（bool 不可）、``d >= 1``。
    max_bond:
        内部 bond 次元の上限（1 以上の整数、bool 不可）。cut ``k`` の bond 次元は
        ``min(max_bond, prod(n_1..n_k), prod(n_{k+1}..n_d))``（その cut で表現可能な最大 rank
        との小さい方）なので、常に ``max_bond`` 以下。端の bond 次元は 1。
    generator:
        乱数 generator（``torch.Generator``）。module-level の seed や global RNG は使わない。
        同じ状態の generator から同じ引数で呼べば、同じ core 列が再現される。generator の
        device は出力先と、index を含めて一致している必要がある（``cuda:0`` と ``cuda:1``
        は別）。``device="cuda"`` のように index を省略した場合は、現在の CUDA device に
        正規化して比べる。不一致は乱数生成の前に ``ValueError``。
    dtype, device:
        ``torch.float32`` / ``torch.float64`` と、作成先の device。

    戻り値:
        shape ``(r_{k-1}, n_k, r_k)`` の core 列。各要素は標準正規分布から独立に抽出する。
        **正規化も正準化もしない**（ノルムは 1 にならない）。必要なら ``normalize_mps`` や
        ``tt_canonicalize`` を呼び出し側で適用すること。
    """
    if isinstance(physical_dimensions, (str, bytes)) or not isinstance(physical_dimensions, Sequence):
        raise TypeError(f"physical_dimensions は整数の列である必要があります: {physical_dimensions!r}")
    if len(physical_dimensions) == 0:
        raise ValueError("physical_dimensions は空であってはいけません。")
    dimensions = [
        _validate_positive_int(dimension, name=f"physical_dimensions[{index}]")
        for index, dimension in enumerate(physical_dimensions)
    ]
    bond_limit = _validate_positive_int(max_bond, name="max_bond")
    if not isinstance(generator, torch.Generator):
        raise TypeError(f"generator は torch.Generator である必要があります: {type(generator)!r}")
    probe = torch.empty(0, dtype=dtype)
    validate_tt_dtype(probe, name="dtype")
    target_device = torch.device(device)
    generator_device = generator.device
    if _canonical_torch_device(generator_device) != _canonical_torch_device(target_device):
        raise ValueError(
            f"generator.device={generator_device} が要求された device={target_device} と一致しません。"
        )

    num_sites = len(dimensions)
    ranks = [1]
    for cut in range(1, num_sites):
        ranks.append(min(bond_limit, math.prod(dimensions[:cut]), math.prod(dimensions[cut:])))
    ranks.append(1)

    return [
        torch.randn(
            (ranks[site], dimensions[site], ranks[site + 1]),
            generator=generator,
            dtype=dtype,
            device=target_device,
        )
        for site in range(num_sites)
    ]
