r"""MPS-MPO environment、期待値、energy、正規化。

参照元:
    notebooks/40_dmrg/00_fundamentals/01_mpo_environment_one_site_local_update_exercises.ipynb
    notebooks/40_dmrg/00_fundamentals/02_environment_two_site_dmrg_sweep_exercises.ipynb

規約:

- MPS core: ``(r_left, physical, r_right)``。
- MPO core: ``(w_left, physical_out, physical_in, w_right)``。``physical_out`` が bra、
  ``physical_in`` が ket と縮約される。
- environment: ``(bra bond, MPO bond, ket bond)`` の 3 階 Tensor。bra 側は ``.conj()``
  する（実数 dtype では恒等）。
- left environment のリストは長さ ``d + 1`` で、``left[k]`` は site ``0..k-1`` を縮約した
  もの（``left[0]`` は境界）。right environment のリストも長さ ``d + 1`` で、
  ``right[k]`` は site ``k..d-1`` を縮約したもの（``right[d]`` は境界）。

この module は TFIM、dense Hamiltonian、厳密 energy、固定 site 数、物理次元 2 に依存しない。
dtype / device は入力 core から決め、module-level の既定値を持たない。

autograd: environment の縮約・期待値・energy は ``detach()`` / ``no_grad()`` を使わず、
入力 core が ``requires_grad=True`` なら graph を維持する。
"""

from __future__ import annotations

import torch

from ..compression import tt_canonicalize, tt_fro_norm, tt_inner
from .validation import _validate_mps_cores, _validate_mps_mpo_pair, _validate_update_operands


def _boundary_environment(reference: torch.Tensor) -> torch.Tensor:
    """開境界の environment ``ones((1, 1, 1))`` を ``reference`` の dtype / device で作る。"""
    return torch.ones((1, 1, 1), dtype=reference.dtype, device=reference.device)


def _update_left(environment: torch.Tensor, mps_core: torch.Tensor, mpo_core: torch.Tensor) -> torch.Tensor:
    """検査なしの left environment 更新（4 項の einsum を 3 段に分け、大きな中間 tensor を避ける）。"""
    # a: bra left, b: MPO left, c: ket left, d: bra physical, e: ket physical
    # i: bra right, j: MPO right, k: ket right
    ket_applied = torch.einsum("abc,cek->abek", environment, mps_core)
    mpo_applied = torch.einsum("abek,bdej->adjk", ket_applied, mpo_core)
    return torch.einsum("adjk,adi->ijk", mpo_applied, mps_core.conj())


def _update_right(environment: torch.Tensor, mps_core: torch.Tensor, mpo_core: torch.Tensor) -> torch.Tensor:
    """検査なしの right environment 更新。"""
    ket_applied = torch.einsum("ijk,cek->ijce", environment, mps_core)
    mpo_applied = torch.einsum("ijce,bdej->ibcd", ket_applied, mpo_core)
    return torch.einsum("ibcd,adi->abc", mpo_applied, mps_core.conj())


def update_left_environment(
    environment: torch.Tensor,
    mps_core: torch.Tensor,
    mpo_core: torch.Tensor,
) -> torch.Tensor:
    r"""left environment に 1 site 分（MPS core と MPO core）を取り込む。

    environment:
        shape ``(r_left, w_left, r_left)``（bra, MPO, ket の順）。
    mps_core:
        shape ``(r_left, n, r_right)``。bra 側では ``.conj()`` される。
    mpo_core:
        shape ``(w_left, n, n, w_right)``。物理次元 ``physical_out == physical_in == n``
        が必要。

    戻り値:
        shape ``(r_right, w_right, r_right)`` の新しい environment。

    $$
    L'_{ijk}=\sum_{a,b,c,d,e}L_{abc}\,\overline{A}_{adi}\,W_{bdej}\,A_{cek}
    $$

    dtype / device は 3 つの入力で一致している必要がある。入力は変更しない。
    有限性（NaN / inf）は検査しない。autograd graph は維持する。
    """
    _validate_update_operands(environment, mps_core, mpo_core, side="left")
    return _update_left(environment, mps_core, mpo_core)


def update_right_environment(
    environment: torch.Tensor,
    mps_core: torch.Tensor,
    mpo_core: torch.Tensor,
) -> torch.Tensor:
    r"""right environment に 1 site 分（MPS core と MPO core）を取り込む。

    environment:
        shape ``(r_right, w_right, r_right)``（bra, MPO, ket の順）。
    mps_core:
        shape ``(r_left, n, r_right)``。bra 側では ``.conj()`` される。
    mpo_core:
        shape ``(w_left, n, n, w_right)``。

    戻り値:
        shape ``(r_left, w_left, r_left)`` の新しい environment。

    $$
    R'_{abc}=\sum_{i,j,k,d,e}R_{ijk}\,\overline{A}_{adi}\,W_{bdej}\,A_{cek}
    $$

    dtype / device は 3 つの入力で一致している必要がある。入力は変更しない。
    有限性は検査しない。autograd graph は維持する。
    """
    _validate_update_operands(environment, mps_core, mpo_core, side="right")
    return _update_right(environment, mps_core, mpo_core)


def _build_left(mps_cores, mpo_cores) -> list[torch.Tensor]:
    environments = [_boundary_environment(mps_cores[0])]
    for mps_core, mpo_core in zip(mps_cores, mpo_cores):
        environments.append(_update_left(environments[-1], mps_core, mpo_core))
    return environments


def _build_right(mps_cores, mpo_cores) -> list[torch.Tensor]:
    environments = [_boundary_environment(mps_cores[-1])]
    for mps_core, mpo_core in zip(reversed(mps_cores), reversed(mpo_cores)):
        environments.append(_update_right(environments[-1], mps_core, mpo_core))
    environments.reverse()
    return environments


def build_left_environments(
    mps_cores: list[torch.Tensor],
    mpo_cores: list[torch.Tensor],
) -> list[torch.Tensor]:
    """全 site の left environment を作る。

    戻り値は長さ ``d + 1`` のリスト。``result[k]`` は site ``0..k-1`` を縮約した environment
    （``result[0]`` は ``ones((1, 1, 1))``）で、``result[d]`` は ``<psi|H|psi>`` を
    要素にもつ ``(1, 1, 1)`` の Tensor。MPS / MPO の構造・物理次元・dtype / device・
    有限性を検査する。入力は変更しない。autograd graph は維持する。
    """
    _validate_mps_mpo_pair(mps_cores, mpo_cores)
    return _build_left(mps_cores, mpo_cores)


def build_right_environments(
    mps_cores: list[torch.Tensor],
    mpo_cores: list[torch.Tensor],
) -> list[torch.Tensor]:
    """全 site の right environment を作る。

    戻り値は長さ ``d + 1`` のリスト。``result[k]`` は site ``k..d-1`` を縮約した environment
    （``result[d]`` は ``ones((1, 1, 1))``）で、``result[0]`` は ``<psi|H|psi>`` を
    要素にもつ ``(1, 1, 1)`` の Tensor。検査と副作用の規約は
    ``build_left_environments`` に同じ。
    """
    _validate_mps_mpo_pair(mps_cores, mpo_cores)
    return _build_right(mps_cores, mpo_cores)


def _rayleigh_is_defined(numerator: torch.Tensor, norm_squared: torch.Tensor) -> bool:
    """分子と ``<psi|psi>`` が、Rayleigh 商を有限値として定義できるか。"""
    return bool(norm_squared > 0) and bool(torch.isfinite(norm_squared)) and bool(torch.isfinite(numerator))


def _mps_has_exact_zero_core(mps_cores) -> bool:
    """いずれかの core が厳密に零なら、状態のノルムは 0。"""
    return any(bool(core.abs().amax() == 0) for core in mps_cores)


def _expectation(mps_cores, mpo_cores) -> torch.Tensor:
    environment = _boundary_environment(mps_cores[0])
    for mps_core, mpo_core in zip(mps_cores, mpo_cores):
        environment = _update_left(environment, mps_core, mpo_core)
    return environment.reshape(())


def _energy(mps_cores, mpo_cores) -> torch.Tensor:
    """検査なしの Rayleigh 商。零ノルムは ``ValueError``。

    直接の縮約が有限ならその値を返す（通常入力はこの経路で、結果は変わらない）。
    隣接 core を逆数スケールした gauge では中間積が overflow / underflow して NaN に
    なる。そのときは状態を変えない ``tt_canonicalize`` でスケールを吸収してから
    同じ縮約をやり直す。NaN を 0 へは置き換えない。
    """
    numerator = _expectation(mps_cores, mpo_cores)
    norm_squared = tt_inner(list(mps_cores), list(mps_cores))
    if _rayleigh_is_defined(numerator, norm_squared):
        return numerator / norm_squared
    if _mps_has_exact_zero_core(mps_cores):
        raise ValueError(f"MPS のノルムの二乗が正の有限値ではありません: {norm_squared.item()!r}")
    balanced = tt_canonicalize(list(mps_cores), center=0)
    numerator = _expectation(balanced, mpo_cores)
    norm_squared = tt_inner(list(balanced), list(balanced))
    if not _rayleigh_is_defined(numerator, norm_squared):
        raise ValueError(
            "隣接 core のスケールを正準化しても energy を有限値として計算できません: "
            f"norm_squared={norm_squared.item()!r}"
        )
    return numerator / norm_squared


def mps_mpo_expectation(
    mps_cores: list[torch.Tensor],
    mpo_cores: list[torch.Tensor],
) -> torch.Tensor:
    r"""期待値 ``<psi|H|psi>`` を MPS-MPO 縮約で計算する（正規化しない）。

    mps_cores:
        MPS core 列 ``(r_left, n_k, r_right)``。正規化されている必要はない。
    mpo_cores:
        MPO core 列 ``(w_left, n_k, n_k, w_right)``。

    戻り値:
        shape ``()`` のスカラー Tensor。状態を dense 化しない。MPS が正規化されて
        いなければ、返り値は ``<psi|psi>`` 倍される。Rayleigh 商が必要なら
        ``mps_mpo_energy`` を使う。

    入力は変更しない。autograd graph は維持する。
    """
    _validate_mps_mpo_pair(mps_cores, mpo_cores)
    return _expectation(mps_cores, mpo_cores)


def mps_mpo_energy(
    mps_cores: list[torch.Tensor],
    mpo_cores: list[torch.Tensor],
) -> torch.Tensor:
    r"""energy（Rayleigh 商）``<psi|H|psi> / <psi|psi>`` を MPS-MPO 縮約で計算する。

    分子は MPS-MPO 縮約、分母は既存の ``tt_inner`` で求め、状態を dense 化しない。
    MPS が正規化されていなくても正しい Rayleigh 商になる。零ノルムの MPS は
    ``ValueError``。隣接 core が ``s`` と ``1/s`` の gauge（物理状態は不変）で直接縮約が
    非有限になるときは、``tt_canonicalize`` でスケールを吸収してから同じ商を計算する。
    通常の有限な入力は直接縮約の結果のまま返す。

    戻り値:
        shape ``()`` のスカラー Tensor（入力 core と同じ dtype / device）。

    入力は変更しない。autograd graph は維持する。
    """
    _validate_mps_mpo_pair(mps_cores, mpo_cores)
    return _energy(mps_cores, mpo_cores)


def normalize_mps(mps_cores: list[torch.Tensor]) -> list[torch.Tensor]:
    """MPS のノルムを 1 にした新しい core 列を返す。

    ノルムは既存の ``tt_fro_norm`` で求め、状態を dense 化しない。直接ノルムが有限で
    正なら、スケールは ``cores[0]`` の 1 つだけに吸収し、他の core は値を変えずに複製する。
    隣接 core の逆数スケールで直接ノルムが NaN / inf になるときは、状態を変えない
    ``tt_canonicalize`` でスケールを吸収してから、そのノルムを ``cores[0]`` に吸収する。
    入力は変更しない。厳密な零 core を含む MPS、正準化後もノルムが有限にならない
    MPS は ``ValueError``。autograd graph は維持する。
    """
    _validate_mps_cores(mps_cores, check_finite=False)
    direct_norm = tt_fro_norm(list(mps_cores))
    if bool(direct_norm > 0) and bool(torch.isfinite(direct_norm)):
        result = [core.clone() for core in mps_cores]
        result[0] = result[0] / direct_norm
        return result
    if _mps_has_exact_zero_core(mps_cores):
        raise ValueError(f"MPS のノルムが正の有限値ではないため正規化できません: {direct_norm.item()!r}")
    balanced = tt_canonicalize(list(mps_cores), center=0)
    norm = tt_fro_norm(balanced)
    if not bool(norm > 0) or not bool(torch.isfinite(norm)):
        raise ValueError(f"MPS のノルムが正の有限値ではないため正規化できません: {norm.item()!r}")
    balanced[0] = balanced[0] / norm
    if any(not bool(torch.isfinite(core).all()) for core in balanced):
        raise ValueError("正規化後の MPS core を有限値として表現できません。")
    return balanced
