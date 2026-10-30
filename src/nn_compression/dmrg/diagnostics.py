r"""実験支援: dense 化による小規模検証用の診断（MPO の dense 化、全残差、energy 分散）。

参照元:
    notebooks/40_dmrg/00_fundamentals/01_mpo_environment_one_site_local_update_exercises.ipynb
    notebooks/40_dmrg/00_fundamentals/03_dmrg_convergence_and_bond_dimension_exercises.ipynb

この module の関数は **小規模系の検証専用** で、DMRG 本体（environment / 局所更新 /
sweep / 収束判定）からは呼ばれない。dense 化のコスト・メモリは物理次元の積 ``D = prod(n_k)`` に
対して、``mpo_to_dense`` が ``O(D^2)``、``compute_dense_residual_and_variance`` が
``O(D^2)``（行列ベクトル積）になる。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .validation import _require_finite, _validate_mpo_cores, _validate_tensor


def mpo_to_dense(mpo_cores: list[torch.Tensor]) -> torch.Tensor:
    r"""MPO を dense 行列 ``(prod(n_out), prod(n_in))`` へ復元する（小規模検証専用）。

    mpo_cores:
        MPO core 列 ``(w_left, n_out_k, n_in_k, w_right)``。端の bond は 1。site ごとの
        ``n_out`` と ``n_in`` は異なってよく（長方形の行列になる）、物理次元は 2 に限らない。

    戻り値:
        行 index は ``(out_1, ..., out_d)``、列 index は ``(in_1, ..., in_d)`` を、site 1 を最も
        上位の桁とする C 順で平坦化したもの（``torch.kron`` の積の順と同じ）。入力と同じ
        dtype / device。

    コスト:
        時間・メモリとも行列の要素数 ``prod(n_out) * prod(n_in)`` に比例し、site 数に対して
        指数的に増える。DMRG 本体からは呼ばれず、数 site の検証にのみ使う。

    入力は変更しない。autograd graph は維持する。
    """
    _validate_mpo_cores(mpo_cores)
    # dense: (行の累積 index, 列の累積 index, 右 MPO bond)
    dense = mpo_cores[0][0]
    for core in mpo_cores[1:]:
        rows, columns, _ = dense.shape
        merged = torch.einsum("xyw,woiv->xoyiv", dense, core)
        dense = merged.reshape(rows * core.shape[1], columns * core.shape[2], core.shape[3])
    return dense[:, :, 0]


@dataclass(frozen=True)
class ResidualVarianceDiagnostics:
    """全 Hamiltonian に対する energy、残差ノルム、energy 分散（すべて shape ``()`` の Tensor）。"""

    energy: torch.Tensor
    residual_norm: torch.Tensor
    energy_variance: torch.Tensor


def _dense_rayleigh_statistics(vector: torch.Tensor, hamiltonian: torch.Tensor):
    """energy、残差ノルム、補正前分散、二次モーメント。定義できなければ ``None``。

    戻り値の Tensor は ``vector`` の autograd graph を維持する。零ノルムや非有限な
    結果は例外にせず ``None``（呼び出し側がスケールを変えて再試行する）。
    """
    norm_squared = torch.dot(vector, vector)
    if not bool(norm_squared > 0) or not bool(torch.isfinite(norm_squared)):
        return None
    hamiltonian_state = hamiltonian @ vector
    energy = torch.dot(vector, hamiltonian_state) / norm_squared
    residual_norm = torch.linalg.vector_norm(hamiltonian_state - energy * vector) / norm_squared.sqrt()
    second_moment = torch.dot(hamiltonian_state, hamiltonian_state) / norm_squared
    raw_variance = second_moment - energy.square()
    if not bool(torch.isfinite(torch.stack([energy, residual_norm, raw_variance, second_moment])).all()):
        return None
    return energy, residual_norm, raw_variance, second_moment


def compute_dense_residual_and_variance(
    state: torch.Tensor,
    hamiltonian: torch.Tensor,
) -> ResidualVarianceDiagnostics:
    r"""dense 状態と dense Hamiltonian から、energy・全残差ノルム・energy 分散を同じ規約で計算する。

    state:
        状態ベクトル（任意の shape で、平坦化して使う）。正規化されている必要はなく、
        内部で ``<psi|psi>`` により割る。ノルム 0 や非有限値は ``ValueError``。
    hamiltonian:
        正方行列 ``(D, D)``、``D = state.numel()``。**Hermite（実対称）であることを前提** に
        する（検査はしない）。dtype / device は ``state`` と一致する必要がある。

    戻り値（``psi`` を正規化した状態として）:
        - ``energy``: ``E = <psi|H|psi>``
        - ``residual_norm``: ``rho = || (H - E) psi ||``
        - ``energy_variance``: ``Var(H) = <H^2> - E^2``。Hermite な ``H`` では ``rho^2`` に一致する。

    分散は ``<H^2> = ||H psi||^2`` から引き算で求めるため、丸め誤差で理論上非負の値が微小に
    負になり得る。その **数値誤差の範囲**（``100 * eps * max(<H^2>, E^2, 1)`` 以内）の負値だけ
    0 へ補正し、それを超える負値は ``FloatingPointError``。

    energy・残差ノルム・分散は、非零状態のスカラー倍（符号を含む）に不変である。直接の
    二乗ノルムが有限で正のときはその計算をそのまま使う。float32 の ``1e30`` / ``1e-30``
    のように二乗ノルムが overflow / underflow する非零状態は、最大絶対値で割ってから
    同じ式を評価する。厳密な零状態だけを拒否し、結果が NaN のときは返さない。

    この関数は小規模検証専用で、DMRG 本体や停止判定からは呼ばれない。入力は変更しない。
    autograd graph は維持する（補正の ``clamp_min`` を除く）。
    """
    _validate_tensor(state, name="state", ndim=state.ndim if torch.is_tensor(state) else 0)
    _validate_tensor(hamiltonian, name="hamiltonian", ndim=2)
    if hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError(f"hamiltonian は正方行列である必要があります: shape={tuple(hamiltonian.shape)}")
    if hamiltonian.dtype != state.dtype:
        raise TypeError(f"hamiltonian.dtype={hamiltonian.dtype} は state.dtype={state.dtype} と一致する必要があります。")
    if hamiltonian.device != state.device:
        raise ValueError(
            f"hamiltonian.device={hamiltonian.device} は state.device={state.device} と一致する必要があります。"
        )
    vector = state.reshape(-1)
    if hamiltonian.shape[1] != vector.numel():
        raise ValueError(
            f"state の要素数 ({vector.numel()}) と hamiltonian の次元 ({hamiltonian.shape[1]}) が一致しません。"
        )
    _require_finite([vector, hamiltonian], name="state / hamiltonian")

    magnitude = vector.abs().amax()
    if bool(magnitude == 0):
        raise ValueError("state のノルムは正である必要があります。")
    statistics = _dense_rayleigh_statistics(vector, hamiltonian)
    if statistics is None:
        # 二乗ノルムが overflow / underflow する非零状態。最大絶対値は有限なので割れる。
        statistics = _dense_rayleigh_statistics(vector / magnitude, hamiltonian)
    if statistics is None:
        raise ValueError("energy、residual、variance を対象 dtype で有限値として計算できません。")
    energy, residual_norm, raw_variance, second_moment = statistics

    scale = max(abs(float(second_moment)), abs(float(energy.square())), 1.0)
    negative_tolerance = 100.0 * torch.finfo(vector.dtype).eps * scale
    if float(raw_variance) < -negative_tolerance:
        raise FloatingPointError(
            f"energy 分散が数値誤差では説明できない負値です: {float(raw_variance):.3e} "
            f"(許容 {negative_tolerance:.3e})"
        )
    return ResidualVarianceDiagnostics(
        energy=energy,
        residual_norm=residual_norm,
        energy_variance=raw_variance.clamp_min(0.0),
    )
