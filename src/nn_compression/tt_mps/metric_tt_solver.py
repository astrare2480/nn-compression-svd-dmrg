"""device を選べる、計量つき TT-matrix の 1 サイト ALS。

``metric_tt.py`` は CPU・float64 の参照実装である。このモジュールは同じ局所問題
``H x = b`` を、device・精度・前処理・CG の許容誤差を引数で切り替えて解く。
戻り値のコアは、保存と比較のために CPU の float64 に戻す。
"""

from __future__ import annotations

import contextvars
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch

from .matrix_two_site import canonicalize_tt_matrix_cores
from .metric_tt import (
    _contract_left_group,
    _contract_right_group,
    _modes,
    move_tt_matrix_center,
)

PrecisionName = Literal["float64", "float32_refine"]
PreconditionerName = Literal["jacobi", "kronecker"]
ToleranceModeName = Literal["fixed", "adaptive"]
CgStatusName = Literal[
    "converged", "max_iterations", "nonpositive_curvature", "nonfinite", "breakdown"
]

# 掃引側がセットしたときだけ、HVP の呼び出し回数を足す。未設定なら何もしない。
_hvp_calls: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar("metric_tt_hvp_calls", default=None)


@dataclass(frozen=True)
class MetricTTSolverConfig:
    """1 サイト ALS の実行設定。既定は CPU 版と同じで、device だけ CUDA。

    ``timing_sync=False`` のとき、HVP と前処理の区間時間は GPU と同期しない。
    足すのは CPU から見た時間で、GPU の実行時間ではない。掃引全体の ``seconds`` は変えない。
    """

    device: str | torch.device = "cuda"
    precision: PrecisionName = "float64"
    preconditioner: PreconditionerName = "jacobi"
    cg_tolerance_mode: ToleranceModeName = "fixed"
    cg_relative_tolerance: float = 1e-8
    cg_tolerance_max: float = 1e-2
    adaptive_eta: float = 0.1
    max_cg_iterations: int = 500
    max_refinements: int = 5
    sweep_tolerance: float = 1e-6
    max_sweeps: int = 50
    gradient_tolerance: float | None = None
    time_limit_seconds: float | None = None
    cg_check_every: int = 1
    timing_sync: bool = True

    def __post_init__(self) -> None:
        if self.precision not in ("float64", "float32_refine"):
            raise ValueError(f"precision が不明です: {self.precision}")
        if self.preconditioner not in ("jacobi", "kronecker"):
            raise ValueError(f"preconditioner が不明です: {self.preconditioner}")
        if self.cg_tolerance_mode not in ("fixed", "adaptive"):
            raise ValueError(f"cg_tolerance_mode が不明です: {self.cg_tolerance_mode}")
        if self.cg_relative_tolerance <= 0 or self.cg_tolerance_max <= 0:
            raise ValueError("CG の許容誤差は正である必要があります。")
        if self.cg_tolerance_max < self.cg_relative_tolerance:
            raise ValueError("cg_tolerance_max は cg_relative_tolerance 以上である必要があります。")
        if self.adaptive_eta <= 0:
            raise ValueError("adaptive_eta は正である必要があります。")
        if self.max_cg_iterations < 1 or self.max_sweeps < 1 or self.max_refinements < 1:
            raise ValueError("反復回数の上限は 1 以上である必要があります。")
        if self.gradient_tolerance is not None and self.gradient_tolerance <= 0:
            raise ValueError("gradient_tolerance は正である必要があります。")
        if self.time_limit_seconds is not None and self.time_limit_seconds < 0:
            raise ValueError("time_limit_seconds は 0 以上である必要があります。")
        if isinstance(self.cg_check_every, bool) or not isinstance(self.cg_check_every, int):
            raise ValueError(f"cg_check_every は 1 以上の int である必要があります: {self.cg_check_every!r}")
        if self.cg_check_every < 1:
            raise ValueError(f"cg_check_every は 1 以上である必要があります: {self.cg_check_every}")
        if not isinstance(self.timing_sync, bool):
            raise ValueError(f"timing_sync は bool である必要があります: {self.timing_sync!r}")


@dataclass(frozen=True)
class DeviceSweepRecord:
    """1 掃引の履歴。``stop_reason`` は最後の掃引だけが入る。"""

    sweep: int
    q_lambda: float
    relative_decrease: float
    cg_iterations: tuple[int, ...]
    cg_residuals: tuple[float, ...]
    n_rejected: int
    seconds: float
    local_gradient_norms: tuple[float, ...]
    cg_tolerances: tuple[float, ...]
    refinements: tuple[int, ...]
    seconds_hvp: float
    seconds_preconditioner: float
    seconds_preconditioner_apply: float
    seconds_qr: float
    seconds_accept: float
    peak_memory_bytes: int | None
    stop_reason: str | None
    kronecker_relative_errors: tuple[float | None, ...]
    kronecker_clipped: tuple[bool, ...]


@dataclass(frozen=True)
class KroneckerMetricFactors:
    """``G ≈ G_L ⊗ G_j ⊗ G_R``。因子は対称で、固有値を下から切って正定値にしてある。"""

    left: torch.Tensor
    middle: torch.Tensor
    right: torch.Tensor
    relative_error: float
    clipped: bool
    sign_flipped: bool


@contextmanager
def _matmul_without_tf32():
    matmul_flag = torch.backends.cuda.matmul.allow_tf32
    cudnn_flag = torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_flag
        torch.backends.cudnn.allow_tf32 = cudnn_flag
        torch.set_float32_matmul_precision(precision)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _Accumulator:
    """区間の前後の時刻を足す。

    ``sync`` が真のときは、区間の前後で CUDA を同期し、GPU の仕事が終わってから時刻を読む。
    偽のときは同期しない。足されるのは CPU から見た時間で、GPU がその仕事に使った時間ではない。
    """

    def __init__(self, device: torch.device, *, sync: bool = True) -> None:
        self.device = device
        self.sync = sync
        self.total = 0.0

    def __enter__(self) -> "_Accumulator":
        if self.sync:
            _sync(self.device)
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.sync:
            _sync(self.device)
        self.total += time.perf_counter() - self._started


def _resolve_device(config: MetricTTSolverConfig) -> torch.device:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA が使えません。")
    return device


def _same_floating(tensor: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} は torch.Tensor である必要があります。")
    if tensor.dtype not in (torch.float32, torch.float64) or tensor.dtype != reference.dtype:
        raise TypeError(f"{name} の dtype が {reference.dtype} と一致しません: {tensor.dtype}")
    if tensor.device != reference.device:
        raise ValueError(f"{name} の device が {reference.device} と一致しません: {tensor.device}")
    return tensor


def metric_tt_hvp_on_device(
    cores: Sequence[torch.Tensor],
    site: int,
    direction: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    """中心コア方向に局所ヘッセを掛ける。dtype と device は ``direction`` に合わせる。"""
    vector = _same_floating(direction, cores[site], "direction")
    metric = _same_floating(gram, cores[site], "gram")
    if vector.shape != cores[site].shape:
        raise ValueError(
            f"direction の shape {tuple(vector.shape)} が cores[{site}] の {tuple(cores[site].shape)} と一致しません。"
        )
    replaced = list(cores)
    replaced[site] = vector
    from .matrix import tt_matrix_to_dense

    counter = _hvp_calls.get()
    if counter is not None:
        counter[0] += 1
    with torch.profiler.record_function("hvp_dense"):
        dense = tt_matrix_to_dense(replaced, out_modes, in_modes)
    with torch.profiler.record_function("hvp_gemm"):
        product = dense @ metric
    with torch.profiler.record_function("hvp_adjoint"):
        return _adjoint(cores, site, product, out_modes, in_modes)


def metric_tt_rhs_on_device(
    cores: Sequence[torch.Tensor],
    site: int,
    weight_gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    """右辺 ``Pᵀ vec(W G)``。``weight_gram`` は使い回す ``W @ G``。"""
    metric_product = _same_floating(weight_gram, cores[site], "weight_gram")
    return _adjoint(cores, site, metric_product, out_modes, in_modes)


def metric_tt_jacobi_diagonal_on_device(
    cores: Sequence[torch.Tensor],
    site: int,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    """Jacobi 対角。出力 mode 方向には同じ値を繰り返す。"""
    out_tuple, in_tuple = _modes(out_modes, in_modes)
    metric = _same_floating(gram, cores[site], "gram")
    n_sites = len(cores)
    left = _contract_left_group(cores, site)
    right = _contract_right_group(cores, site + 1)
    n_left = math.prod(in_tuple[:site]) if site else 1
    n_site = in_tuple[site]
    n_right = math.prod(in_tuple[site + 1 :]) if site < n_sites - 1 else 1
    if metric.shape != (n_left * n_site * n_right, n_left * n_site * n_right):
        raise ValueError(f"gram の shape {tuple(metric.shape)} が入力次元と一致しません。")
    gram_left = torch.einsum("aip,aiq->apq", left, left)
    gram_right = torch.einsum("bur,bus->brs", right, right)
    reshaped = metric.reshape(n_left, n_site, n_right, n_left, n_site, n_right)
    diagonal_slice = torch.diagonal(reshaped, dim1=1, dim2=4)
    diagonal_slice = diagonal_slice.permute(4, 0, 1, 2, 3)
    reduced = torch.einsum("apq,tprqs,brs->atb", gram_left, diagonal_slice, gram_right)
    return reduced.unsqueeze(1).expand(-1, out_tuple[site], -1, -1).contiguous()


def _adjoint(
    cores: Sequence[torch.Tensor],
    site: int,
    matrix: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    out_tuple, in_tuple = _modes(out_modes, in_modes)
    dense = _same_floating(matrix, cores[site], "matrix")
    n_sites = len(cores)
    if not 0 <= site < n_sites:
        raise ValueError(f"site={site} は 0 以上 {n_sites} 未満である必要があります。")
    m_total = math.prod(out_tuple)
    n_total = math.prod(in_tuple)
    if dense.shape != (m_total, n_total):
        raise ValueError(f"matrix の shape {tuple(dense.shape)} は {(m_total, n_total)} である必要があります。")
    left = _contract_left_group(cores, site)
    right = _contract_right_group(cores, site + 1)
    grouped = dense.reshape(*out_tuple, *in_tuple)
    m_left = math.prod(out_tuple[:site]) if site else 1
    m_right = math.prod(out_tuple[site + 1 :]) if site < n_sites - 1 else 1
    n_left = math.prod(in_tuple[:site]) if site else 1
    n_right = math.prod(in_tuple[site + 1 :]) if site < n_sites - 1 else 1
    order = (
        list(range(0, site))
        + [site]
        + list(range(site + 1, n_sites))
        + list(range(n_sites, n_sites + site))
        + [n_sites + site]
        + list(range(n_sites + site + 1, 2 * n_sites))
    )
    reshaped = grouped.permute(*order).reshape(
        m_left, out_tuple[site], m_right, n_left, in_tuple[site], n_right
    )
    return torch.einsum("apq,piuqjv,buv->aijb", left, reshaped, right)


def _rank1_kronecker_factors(matrix: torch.Tensor, left_n: int, right_n: int) -> tuple[torch.Tensor, torch.Tensor]:
    view = matrix.reshape(left_n, right_n, left_n, right_n)
    rearranged = view.permute(0, 2, 1, 3).reshape(left_n * left_n, right_n * right_n)
    left_vectors, singular_values, right_vectors = torch.linalg.svd(rearranged, full_matrices=False)
    scale = torch.sqrt(singular_values[0].clamp(min=0))
    left = (scale * left_vectors[:, 0]).reshape(left_n, left_n)
    right = (scale * right_vectors[0]).reshape(right_n, right_n)
    return left, right


def _positive_definite_factor(matrix: torch.Tensor) -> tuple[torch.Tensor, bool, bool]:
    symmetric = (matrix + matrix.transpose(-1, -2)) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    sign_flipped = False
    if float(eigenvalues.sum().item()) < 0.0:
        eigenvalues = -eigenvalues
        sign_flipped = True
    largest = float(eigenvalues.max().item())
    floor = largest * 1e-12 if largest > 0.0 else float(torch.finfo(eigenvalues.dtype).tiny)
    clipped = bool(torch.any(eigenvalues < floor).item())
    eigenvalues = torch.clamp(eigenvalues, min=floor)
    rebuilt = (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
    rebuilt = (rebuilt + rebuilt.transpose(-1, -2)) * 0.5
    return rebuilt, clipped, sign_flipped


def kronecker_metric_factors(
    gram: torch.Tensor,
    n_left: int,
    n_site: int,
    n_right: int,
) -> KroneckerMetricFactors:
    """Van Loan–Pitsianis を 2 段階使い、``G`` の Kronecker 近似を作る。"""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"gram は正方である必要があります: shape={tuple(gram.shape)}")
    width = n_left * n_site * n_right
    if gram.shape[0] != width:
        raise ValueError(f"gram の次数 {gram.shape[0]} が {width} と一致しません。")
    left_raw, rest_raw = _rank1_kronecker_factors(gram, n_left, n_site * n_right)
    middle_raw, right_raw = _rank1_kronecker_factors(rest_raw, n_site, n_right)
    left, clip_left, flip_left = _positive_definite_factor(left_raw)
    middle, clip_middle, flip_middle = _positive_definite_factor(middle_raw)
    right, clip_right, flip_right = _positive_definite_factor(right_raw)
    reconstructed = torch.kron(torch.kron(left, middle), right)
    numerator = torch.linalg.matrix_norm(gram - reconstructed, ord="fro")
    denominator = torch.linalg.matrix_norm(gram, ord="fro")
    relative = 0.0 if float(denominator.item()) == 0.0 else float((numerator / denominator).item())
    return KroneckerMetricFactors(
        left=left,
        middle=middle,
        right=right,
        relative_error=relative,
        clipped=clip_left or clip_middle or clip_right,
        sign_flipped=flip_left or flip_middle or flip_right,
    )


def _environment_factor(group: torch.Tensor, metric_factor: torch.Tensor) -> torch.Tensor:
    """``Σ_m group[a,m,n] group[a',m,n'] metric[n,n']`` を ``(r, r)`` で返す。

    先に計量を掛けてから縮約し、``(r, n, r, n)`` の中間配列を作らない。
    """
    weighted = torch.einsum("amn,np->amp", group, metric_factor)
    return torch.einsum("amp,bmp->ab", weighted, group)


def kronecker_preconditioner_matrix(
    cores: Sequence[torch.Tensor],
    site: int,
    factors: KroneckerMetricFactors,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    """前処理 ``M = K_L ⊗ G_j ⊗ K_R`` を、小さい問題用に明示的な行列で返す。"""
    del out_modes, in_modes
    left = _contract_left_group(cores, site)
    right = _contract_right_group(cores, site + 1)
    k_left = _environment_factor(left, factors.left)
    k_right = _environment_factor(right, factors.right)
    return torch.kron(torch.kron(k_left, factors.middle), k_right)


@dataclass(frozen=True)
class _KroneckerEigensystem:
    """``K_L``・``G_j``・``K_R`` の固有値分解。局所問題ごとに 1 回だけ作る。"""

    values_left: torch.Tensor
    vectors_left: torch.Tensor
    values_middle: torch.Tensor
    vectors_middle: torch.Tensor
    values_right: torch.Tensor
    vectors_right: torch.Tensor


def _kronecker_eigensystem(
    k_left: torch.Tensor,
    middle: torch.Tensor,
    k_right: torch.Tensor,
) -> _KroneckerEigensystem:
    eval_left, vec_left = torch.linalg.eigh((k_left + k_left.transpose(0, 1)) * 0.5)
    eval_middle, vec_middle = torch.linalg.eigh((middle + middle.transpose(0, 1)) * 0.5)
    eval_right, vec_right = torch.linalg.eigh((k_right + k_right.transpose(0, 1)) * 0.5)
    return _KroneckerEigensystem(eval_left, vec_left, eval_middle, vec_middle, eval_right, vec_right)


def _apply_kronecker_inverse(residual: torch.Tensor, system: _KroneckerEigensystem) -> torch.Tensor:
    coeffs = torch.einsum(
        "amjb,aA,jJ,bB->AmJB",
        residual,
        system.vectors_left,
        system.vectors_middle,
        system.vectors_right,
    )
    denominator = (
        system.values_left[:, None, None, None]
        * system.values_middle[None, None, :, None]
        * system.values_right[None, None, None, :]
    )
    coeffs = coeffs / denominator
    return torch.einsum(
        "AmJB,aA,jJ,bB->amjb",
        coeffs,
        system.vectors_left,
        system.vectors_middle,
        system.vectors_right,
    )


def _safe_jacobi(diagonal: torch.Tensor) -> torch.Tensor:
    floor = torch.clamp(diagonal.max() * torch.finfo(diagonal.dtype).eps, min=torch.finfo(diagonal.dtype).tiny)
    return torch.where(diagonal > 0, diagonal, floor)


def _build_preconditioner(
    cores: Sequence[torch.Tensor],
    site: int,
    gram: torch.Tensor,
    factors: KroneckerMetricFactors | None,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    apply_time: _Accumulator,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """前処理を 1 回だけ作り、掛けるだけの関数を返す。

    前処理が依存するのは中心以外のコアと ``G`` だけなので、同じ局所問題の中
    （CG の反復と float32 の外側反復）では作り直さない。dtype は ``cores`` に合わせる。
    """
    if factors is None:
        diagonal = _safe_jacobi(metric_tt_jacobi_diagonal_on_device(cores, site, gram, out_modes, in_modes))

        def apply_jacobi(residual: torch.Tensor) -> torch.Tensor:
            with apply_time:
                return residual / diagonal

        return apply_jacobi

    dtype = cores[site].dtype
    left = _contract_left_group(cores, site)
    right = _contract_right_group(cores, site + 1)
    k_left = _environment_factor(left, factors.left.to(dtype=dtype))
    k_right = _environment_factor(right, factors.right.to(dtype=dtype))
    system = _kronecker_eigensystem(k_left, factors.middle.to(dtype=dtype), k_right)

    def apply_kronecker(residual: torch.Tensor) -> torch.Tensor:
        with apply_time:
            return _apply_kronecker_inverse(residual, system)

    return apply_kronecker


def _conjugate_gradient(
    apply_hessian: Callable[[torch.Tensor], torch.Tensor],
    apply_preconditioner: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    initial: torch.Tensor,
    *,
    relative_tolerance: float,
    max_iterations: int,
    check_every: int = 1,
) -> tuple[torch.Tensor, int, float, tuple[float, ...], tuple[float, ...], CgStatusName, int]:
    """共役勾配法。戻り値は解、x を進めた回数、相対残差、履歴、時刻、終わり方、空回り。

    ``check_every=1`` は、反復ごとに曲率と相対残差を CPU へ取り出す今までの計算。空回りは 0。
    ``check_every>1`` は、その回数だけ GPU 上で進めてから残差を確かめる。
    曲率が 0 以下、または値が有限でない反復の更新は解に入れない。
    内積が 0 のときは ``breakdown``。区切りの終わりの残差が許容誤差以下なら ``converged``。
    許容誤差に届いたあとも、区切りの終わりまでは最大 ``check_every - 1`` 回進む。
    """
    if check_every != 1:
        return _conjugate_gradient_blocked(
            apply_hessian,
            apply_preconditioner,
            rhs,
            initial,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            check_every=check_every,
        )
    started = time.perf_counter()
    x_vector = initial.clone()
    residual = rhs - apply_hessian(x_vector)
    b_norm = torch.linalg.vector_norm(rhs.reshape(-1))
    history: list[float] = []
    marks: list[float] = []
    if float(b_norm.item()) == 0.0:
        return x_vector, 0, 0.0, (0.0,), (0.0,), "converged", 0
    relative = float((torch.linalg.vector_norm(residual.reshape(-1)) / b_norm).item())
    history.append(relative)
    marks.append(time.perf_counter() - started)
    if relative <= relative_tolerance:
        return x_vector, 0, relative, tuple(history), tuple(marks), "converged", 0
    preconditioned = apply_preconditioner(residual)
    direction = preconditioned.clone()
    residual_dot = torch.dot(residual.reshape(-1), preconditioned.reshape(-1))
    used = 0
    status: CgStatusName = "max_iterations"
    for iteration in range(1, max_iterations + 1):
        hessian_direction = apply_hessian(direction)
        with torch.profiler.record_function("cg_dot"):
            curvature = torch.dot(direction.reshape(-1), hessian_direction.reshape(-1))
        with torch.profiler.record_function("cg_curvature_item"):
            curvature_value = float(curvature.item())
        if curvature_value <= 0.0:
            status = "nonpositive_curvature"
            break
        alpha = residual_dot / curvature
        x_vector = x_vector + alpha * direction
        residual = residual - alpha * hessian_direction
        with torch.profiler.record_function("cg_residual_item"):
            relative = float((torch.linalg.vector_norm(residual.reshape(-1)) / b_norm).item())
        history.append(relative)
        marks.append(time.perf_counter() - started)
        used = iteration
        if relative <= relative_tolerance:
            status = "converged"
            break
        preconditioned = apply_preconditioner(residual)
        new_dot = torch.dot(residual.reshape(-1), preconditioned.reshape(-1))
        beta = new_dot / residual_dot
        direction = preconditioned + beta * direction
        residual_dot = new_dot
    if status == "max_iterations" and not math.isfinite(relative):
        status = "nonfinite"
    return x_vector, used, relative, tuple(history), tuple(marks), status, 0


def _conjugate_gradient_blocked(
    apply_hessian: Callable[[torch.Tensor], torch.Tensor],
    apply_preconditioner: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    initial: torch.Tensor,
    *,
    relative_tolerance: float,
    max_iterations: int,
    check_every: int,
) -> tuple[torch.Tensor, int, float, tuple[float, ...], tuple[float, ...], CgStatusName, int]:
    """残差の確認を ``check_every`` 回ごとにだけ CPU で行う共役勾配法。

    反復の中の全要素の演算は、HVP、前処理、内積、ベクトルの更新だけにする。
    止まったあとの反復も区切りの終わりまで HVP を回す。返す回数は x を進めた回数で、
    空回りは別の戻り値。``0 * NaN`` は NaN になるので、壊れた途中の x は区切りの
    終わりに、進めた回数のところへ参照を戻す。
    """
    started = time.perf_counter()
    x_vector = initial.clone()
    residual = rhs - apply_hessian(x_vector)
    b_norm = torch.linalg.vector_norm(rhs.reshape(-1))
    history: list[float] = []
    marks: list[float] = []
    b_norm_value = float(b_norm.item())
    if b_norm_value == 0.0:
        return x_vector, 0, 0.0, (0.0,), (0.0,), "converged", 0
    relative = float((torch.linalg.vector_norm(residual.reshape(-1)) / b_norm).item())
    history.append(relative)
    marks.append(time.perf_counter() - started)
    if relative <= relative_tolerance:
        return x_vector, 0, relative, tuple(history), tuple(marks), "converged", 0
    preconditioned = apply_preconditioner(residual)
    direction = preconditioned.clone()
    residual_dot = torch.dot(residual.reshape(-1), preconditioned.reshape(-1))
    device = x_vector.device
    dtype = x_vector.dtype
    zero = torch.zeros((), dtype=dtype, device=device)
    one = torch.ones((), dtype=dtype, device=device)
    active = torch.ones((), dtype=torch.bool, device=device)
    nonpositive = torch.zeros((), dtype=torch.bool, device=device)
    nonfinite = torch.zeros((), dtype=torch.bool, device=device)
    breakdown = torch.zeros((), dtype=torch.bool, device=device)
    advanced = torch.zeros((), dtype=torch.int64, device=device)
    advanced_total = 0
    executed = 0
    status: CgStatusName = "max_iterations"
    iteration = 0
    while iteration < max_iterations:
        chunk = min(check_every, max_iterations - iteration)
        x_before = x_vector
        residual_before = residual
        x_steps: list[torch.Tensor] = []
        residual_steps: list[torch.Tensor] = []
        for _step in range(chunk):
            iteration += 1
            executed += 1
            hessian_direction = apply_hessian(direction)
            curvature = torch.dot(direction.reshape(-1), hessian_direction.reshape(-1))
            curvature_ok = curvature > 0
            dot_finite = torch.isfinite(residual_dot)
            ok = active & curvature_ok & dot_finite
            safe_curvature = torch.where(ok, curvature, one)
            alpha = torch.where(ok, residual_dot / safe_curvature, zero)
            x_vector = x_vector + alpha * direction
            residual = residual - alpha * hessian_direction
            x_steps.append(x_vector)
            residual_steps.append(residual)
            stopped = active & ~ok
            nonpositive = nonpositive | (stopped & torch.isfinite(curvature))
            nonfinite = nonfinite | (stopped & ~torch.isfinite(curvature))
            nonfinite = nonfinite | (active & curvature_ok & ~dot_finite)
            preconditioned = apply_preconditioner(residual)
            new_dot = torch.dot(residual.reshape(-1), preconditioned.reshape(-1))
            zero_dot = residual_dot == 0
            can_beta = ok & ~zero_dot & torch.isfinite(new_dot)
            breakdown = breakdown | (ok & dot_finite & zero_dot)
            nonfinite = nonfinite | (ok & ~zero_dot & ~torch.isfinite(new_dot))
            scale = can_beta.to(dtype=dtype)
            safe_dot = torch.where(can_beta, residual_dot, one)
            beta = torch.where(can_beta, new_dot / safe_dot, zero)
            beta_kept = scale * beta + (one - scale)
            direction = preconditioned * scale + direction * beta_kept
            residual_dot = new_dot * scale + residual_dot * (one - scale)
            active = can_beta
            advanced = advanced + (ok & ~zero_dot).to(dtype=torch.int64)
        with torch.profiler.record_function("cg_block_item"):
            residual_norm = torch.linalg.vector_norm(residual.reshape(-1))
            summary = torch.stack(
                (
                    residual_norm.to(dtype=torch.float64),
                    nonpositive.to(dtype=torch.float64),
                    nonfinite.to(dtype=torch.float64),
                    breakdown.to(dtype=torch.float64),
                    advanced.to(dtype=torch.float64),
                )
            )
            copied = summary.detach().to(device="cpu")
        advanced_now = int(copied[4].item())
        n_good = advanced_now - advanced_total
        advanced_total = advanced_now
        if n_good == chunk:
            relative = float(copied[0].item()) / b_norm_value
        else:
            if n_good <= 0:
                x_vector = x_before
                residual = residual_before
            else:
                x_vector = x_steps[n_good - 1]
                residual = residual_steps[n_good - 1]
            relative = float(torch.linalg.vector_norm(residual.reshape(-1)).detach().cpu()) / b_norm_value
        del x_steps, residual_steps
        history.append(relative)
        marks.append(time.perf_counter() - started)
        # 区切りの途中で許容誤差に届いたあとの余分な反復が印を立てても、
        # 戻した解の残差が基準以下なら収束として扱う。
        if math.isfinite(relative) and relative <= relative_tolerance:
            status = "converged"
            break
        if float(copied[2].item()) != 0.0 or not math.isfinite(relative):
            status = "nonfinite"
            break
        if float(copied[1].item()) != 0.0:
            status = "nonpositive_curvature"
            break
        if float(copied[3].item()) != 0.0:
            status = "breakdown"
            break
    if status == "max_iterations" and not math.isfinite(relative):
        status = "nonfinite"
    return (
        x_vector,
        advanced_total,
        relative,
        tuple(history),
        tuple(marks),
        status,
        executed - advanced_total,
    )


def _site_tolerance(config: MetricTTSolverConfig, gradient_norm: float) -> float:
    if config.cg_tolerance_mode == "fixed":
        return config.cg_relative_tolerance
    unclipped = config.adaptive_eta * gradient_norm
    return float(min(config.cg_tolerance_max, max(config.cg_relative_tolerance, unclipped)))


def _q_lambda(
    cores: Sequence[torch.Tensor],
    weight: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    from .matrix import tt_matrix_to_dense

    approximated = tt_matrix_to_dense(cores, out_modes, in_modes)
    delta = weight - approximated
    metric = (gram + gram.transpose(0, 1)) * 0.5
    return torch.sum(delta * (delta @ metric))


def _residual_and_relative(
    cores: Sequence[torch.Tensor],
    site: int,
    value: torch.Tensor,
    rhs: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> tuple[torch.Tensor, float]:
    """残差 ``b − H x`` とその相対ノルム。HVP を 1 回だけ使う。"""
    residual = rhs - metric_tt_hvp_on_device(cores, site, value, gram, out_modes, in_modes)
    rhs_norm = torch.linalg.vector_norm(rhs.reshape(-1))
    if float(rhs_norm.item()) == 0.0:
        return residual, 0.0
    return residual, float((torch.linalg.vector_norm(residual.reshape(-1)) / rhs_norm).item())


def _relative_residual(
    cores: Sequence[torch.Tensor],
    site: int,
    value: torch.Tensor,
    rhs: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> float:
    return _residual_and_relative(cores, site, value, rhs, gram, out_modes, in_modes)[1]


def _solve_site(
    cores: Sequence[torch.Tensor],
    site: int,
    gram64: torch.Tensor,
    gram32: torch.Tensor | None,
    weight_gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    config: MetricTTSolverConfig,
    factor_cache: dict[tuple[int, int, int], KroneckerMetricFactors],
    hvp_time: _Accumulator,
    preconditioner_time: _Accumulator,
    preconditioner_apply_time: _Accumulator,
    residual_log: list[float] | None = None,
    second_log: list[float] | None = None,
    cg_statuses: list[CgStatusName] | None = None,
    cg_idle: list[int] | None = None,
) -> tuple[torch.Tensor, int, float, float, float, int, float | None, bool]:
    """戻り値は解、CG 回数、最終の float64 相対残差、g_k、許容誤差、外側反復、近似誤差、クリップ。

    ``preconditioner_time`` には前処理を作る時間（Kronecker 因子を含む）、
    ``preconditioner_apply_time`` には前処理を掛ける時間を足す。
    """
    clock = time.perf_counter()
    _out_tuple, in_tuple = _modes(out_modes, in_modes)
    n_left = math.prod(in_tuple[:site]) if site else 1
    n_right = math.prod(in_tuple[site + 1 :]) if site < len(cores) - 1 else 1
    n_site = in_tuple[site]
    rhs = metric_tt_rhs_on_device(cores, site, weight_gram, out_modes, in_modes)
    initial_residual, gradient_norm = _residual_and_relative(
        cores, site, cores[site], rhs, gram64, out_modes, in_modes
    )
    tolerance = _site_tolerance(config, gradient_norm)
    factors: KroneckerMetricFactors | None = None
    if config.preconditioner == "kronecker":
        key = (n_left, n_site, n_right)
        if key not in factor_cache:
            with preconditioner_time:
                factor_cache[key] = kronecker_metric_factors(gram64, n_left, n_site, n_right)
        factors = factor_cache[key]

    def apply64(direction: torch.Tensor) -> torch.Tensor:
        with hvp_time:
            return metric_tt_hvp_on_device(cores, site, direction, gram64, out_modes, in_modes)

    if config.precision == "float64":
        with preconditioner_time:
            precondition64 = _build_preconditioner(
                cores, site, gram64, factors, out_modes, in_modes, preconditioner_apply_time
            )
        cg_offset = time.perf_counter() - clock
        solution, iterations, _inner_residual, history, marks, status, idle = _conjugate_gradient(
            apply64,
            precondition64,
            rhs,
            cores[site],
            relative_tolerance=tolerance,
            max_iterations=config.max_cg_iterations,
            check_every=config.cg_check_every,
        )
        if cg_statuses is not None:
            cg_statuses.append(status)
        if cg_idle is not None:
            cg_idle.append(idle)
        if residual_log is not None:
            residual_log.extend(history)
        if second_log is not None:
            second_log.extend(cg_offset + mark for mark in marks)
        final_residual = _relative_residual(cores, site, solution, rhs, gram64, out_modes, in_modes)
        error = None if factors is None else factors.relative_error
        clipped = False if factors is None else factors.clipped
        return solution, iterations, final_residual, gradient_norm, tolerance, 0, error, clipped

    assert gram32 is not None
    cores32 = [core.to(dtype=torch.float32) for core in cores]
    with preconditioner_time:
        precondition32 = _build_preconditioner(
            cores32, site, gram32, factors, out_modes, in_modes, preconditioner_apply_time
        )

    def apply32(direction: torch.Tensor) -> torch.Tensor:
        with hvp_time:
            return metric_tt_hvp_on_device(cores32, site, direction, gram32, out_modes, in_modes)

    working = cores[site].clone()
    total_iterations = 0
    refinements = 0
    inner_tolerance = max(tolerance, 1e-4)
    reached = False
    # 最初の外側反復の残差は、局所勾配を測ったときのものと同じなので使い回す。
    residual64, final_residual = initial_residual, gradient_norm
    for refinement_index in range(config.max_refinements):
        if refinement_index > 0:
            residual64, final_residual = _residual_and_relative(
                cores, site, working, rhs, gram64, out_modes, in_modes
            )
        if residual_log is not None:
            residual_log.append(final_residual)
        if second_log is not None:
            second_log.append(time.perf_counter() - clock)
        if final_residual <= tolerance:
            reached = True
            break
        cores32[site] = working.to(dtype=torch.float32)
        correction, iterations, _relative, _history, _marks, status, idle = _conjugate_gradient(
            apply32,
            precondition32,
            residual64.to(dtype=torch.float32),
            torch.zeros_like(cores32[site]),
            relative_tolerance=inner_tolerance,
            max_iterations=config.max_cg_iterations,
            check_every=config.cg_check_every,
        )
        if cg_statuses is not None:
            cg_statuses.append(status)
        if cg_idle is not None:
            cg_idle.append(idle)
        working = working + correction.to(dtype=torch.float64)
        total_iterations += iterations
        refinements += 1
    if not reached:
        final_residual = _relative_residual(cores, site, working, rhs, gram64, out_modes, in_modes)
        if residual_log is not None:
            residual_log.append(final_residual)
        if second_log is not None:
            second_log.append(time.perf_counter() - clock)
    error = None if factors is None else factors.relative_error
    clipped = False if factors is None else factors.clipped
    return working, total_iterations, final_residual, gradient_norm, tolerance, refinements, error, clipped


@dataclass(frozen=True)
class LocalSolveRecord:
    """1 サイトの局所解。``residual_history`` の先頭は初期相対残差。

    float64 では CG の各反復後の相対残差が続く。float32_refine では、外側反復の
    前後に測った float64 の相対残差だけが続く。内側 CG の float32 残差は入れない。
    """

    iterations: int
    final_relative_residual: float
    residual_history: tuple[float, ...]
    seconds_history: tuple[float, ...]
    seconds: float
    refinements: int
    kronecker_relative_error: float | None
    kronecker_clipped: bool
    local_gradient_norm: float
    cg_tolerance: float
    seconds_preconditioner_setup: float
    seconds_preconditioner_apply: float


def solve_metric_tt_site_on_device(
    cores: Sequence[torch.Tensor],
    site: int,
    weight: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    config: MetricTTSolverConfig,
) -> LocalSolveRecord:
    """正準中心が ``site`` にあるコアで、局所方程式を 1 回解く。

    入力は変更しない。掃引も正準中心の移動もしない。
    """
    if not isinstance(config, MetricTTSolverConfig):
        raise TypeError("config は MetricTTSolverConfig である必要があります。")
    device = _resolve_device(config)
    with _matmul_without_tf32():
        current = [core.detach().to(device=device, dtype=torch.float64).clone() for core in cores]
        weight64 = weight.detach().to(device=device, dtype=torch.float64)
        gram64 = gram.detach().to(device=device, dtype=torch.float64)
        gram32 = gram64.to(dtype=torch.float32) if config.precision == "float32_refine" else None
        residual_log: list[float] = []
        second_log: list[float] = []
        setup_time = _Accumulator(device, sync=config.timing_sync)
        apply_time = _Accumulator(device, sync=config.timing_sync)
        _sync(device)
        started = time.perf_counter()
        _solution, iterations, final_residual, gradient_norm, tolerance, refinements, error, clipped = _solve_site(
            current,
            site,
            gram64,
            gram32,
            weight64 @ gram64,
            out_modes,
            in_modes,
            config,
            {},
            _Accumulator(device, sync=config.timing_sync),
            setup_time,
            apply_time,
            residual_log,
            second_log,
        )
        _sync(device)
        elapsed = time.perf_counter() - started
    return LocalSolveRecord(
        iterations=iterations,
        final_relative_residual=final_residual,
        residual_history=tuple(residual_log),
        seconds_history=tuple(second_log),
        seconds=elapsed,
        refinements=refinements,
        kronecker_relative_error=error,
        kronecker_clipped=clipped,
        local_gradient_norm=gradient_norm,
        cg_tolerance=tolerance,
        seconds_preconditioner_setup=setup_time.total,
        seconds_preconditioner_apply=apply_time.total,
    )


def run_metric_tt_als_on_device(
    cores: Sequence[torch.Tensor],
    weight: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    config: MetricTTSolverConfig,
    *,
    progress: Callable[[DeviceSweepRecord], None] | None = None,
) -> tuple[list[torch.Tensor], tuple[DeviceSweepRecord, ...]]:
    """正準形を保つ 1 サイト ALS を、``config.device`` で実行する。

    入力のコア・重み・Gram は変更しない。戻り値のコアは CPU の float64。
    ``progress`` には、掃引が終わるたびにその掃引の記録を渡す。
    """
    if not isinstance(config, MetricTTSolverConfig):
        raise TypeError("config は MetricTTSolverConfig である必要があります。")
    device = _resolve_device(config)
    started = time.perf_counter()
    with _matmul_without_tf32():
        current = [
            core.detach().to(device=device, dtype=torch.float64).clone() for core in cores
        ]
        weight64 = weight.detach().to(device=device, dtype=torch.float64)
        gram64 = gram.detach().to(device=device, dtype=torch.float64)
        gram32 = gram64.to(dtype=torch.float32) if config.precision == "float32_refine" else None
        weight_gram = weight64 @ gram64
        initial_qr = _Accumulator(device, sync=config.timing_sync)
        with initial_qr:
            current = canonicalize_tt_matrix_cores(current, out_modes, in_modes, 0)
        q_previous = float(_q_lambda(current, weight64, gram64, out_modes, in_modes).item())
        factor_cache: dict[tuple[int, int, int], KroneckerMetricFactors] = {}
        history: list[DeviceSweepRecord] = []
        n_sites = len(current)
        for sweep in range(1, config.max_sweeps + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            hvp_time = _Accumulator(device, sync=config.timing_sync)
            preconditioner_time = _Accumulator(device, sync=config.timing_sync)
            preconditioner_apply_time = _Accumulator(device, sync=config.timing_sync)
            qr_time = _Accumulator(device, sync=config.timing_sync)
            accept_time = _Accumulator(device, sync=config.timing_sync)
            sweep_clock = time.perf_counter()
            iterations: list[int] = []
            residuals: list[float] = []
            gradients: list[float] = []
            tolerances: list[float] = []
            refinements: list[int] = []
            errors: list[float | None] = []
            clipped: list[bool] = []
            rejected = 0

            def update_site(site: int) -> None:
                nonlocal current, rejected
                solution, cg_iterations, cg_residual, gradient_norm, tolerance, refinement, error, was_clipped = (
                    _solve_site(
                        current,
                        site,
                        gram64,
                        gram32,
                        weight_gram,
                        out_modes,
                        in_modes,
                        config,
                        factor_cache,
                        hvp_time,
                        preconditioner_time,
                        preconditioner_apply_time,
                    )
                )
                with accept_time:
                    candidate = list(current)
                    candidate[site] = solution
                    q_before = float(_q_lambda(current, weight64, gram64, out_modes, in_modes).item())
                    q_after = float(_q_lambda(candidate, weight64, gram64, out_modes, in_modes).item())
                    accepted = q_after <= q_before
                if accepted:
                    current = candidate
                else:
                    rejected += 1
                iterations.append(cg_iterations)
                residuals.append(cg_residual)
                gradients.append(gradient_norm)
                tolerances.append(tolerance)
                refinements.append(refinement)
                errors.append(error)
                clipped.append(was_clipped)

            for site in range(n_sites):
                update_site(site)
                if site < n_sites - 1:
                    with qr_time:
                        current = move_tt_matrix_center(current, out_modes, in_modes, site, site + 1)
            for site in range(n_sites - 2, -1, -1):
                with qr_time:
                    current = move_tt_matrix_center(current, out_modes, in_modes, site + 1, site)
                update_site(site)
            q_next = float(_q_lambda(current, weight64, gram64, out_modes, in_modes).item())
            relative_decrease = 0.0 if q_previous == 0.0 else (q_previous - q_next) / q_previous
            _sync(device)
            elapsed = time.perf_counter() - started
            reason: str | None = None
            if relative_decrease < config.sweep_tolerance:
                reason = "sweep_tolerance"
            elif config.gradient_tolerance is not None and max(gradients) < config.gradient_tolerance:
                reason = "gradient_tolerance"
            elif sweep >= config.max_sweeps:
                reason = "max_sweeps"
            elif config.time_limit_seconds is not None and elapsed >= config.time_limit_seconds:
                reason = "time_limit_seconds"
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            record = DeviceSweepRecord(
                sweep=sweep,
                q_lambda=q_next,
                relative_decrease=relative_decrease,
                cg_iterations=tuple(iterations),
                cg_residuals=tuple(residuals),
                n_rejected=rejected,
                seconds=time.perf_counter() - sweep_clock,
                local_gradient_norms=tuple(gradients),
                cg_tolerances=tuple(tolerances),
                refinements=tuple(refinements),
                seconds_hvp=hvp_time.total,
                seconds_preconditioner=preconditioner_time.total,
                seconds_preconditioner_apply=preconditioner_apply_time.total,
                seconds_qr=qr_time.total + (initial_qr.total if sweep == 1 else 0.0),
                seconds_accept=accept_time.total,
                peak_memory_bytes=peak,
                stop_reason=reason,
                kronecker_relative_errors=tuple(errors),
                kronecker_clipped=tuple(clipped),
            )
            history.append(record)
            if progress is not None:
                progress(record)
            if reason is not None:
                break
            q_previous = q_next
    returned = [core.detach().to(device="cpu", dtype=torch.float64).contiguous() for core in current]
    return returned, tuple(history)


# ---------------------------------------------------------------------------
# 長い ALS：掃引の終わりでの停留性、停止の規則、保存と再開、診断用の摂動
# 設計：plan/50_llm_tt_vs_matrix/10_qwen3_0p6b_base/03_long_als_convergence/design.md
# ---------------------------------------------------------------------------

CONVERGENCE_CHECKPOINT_FORMAT = "metric_tt_convergence_checkpoint/2"
CONVERGENCE_STOP_REASONS = ("converged", "max_sweeps", "local_solver_unresolved")


@dataclass(frozen=True)
class ConvergenceStopRule:
    """長い ALS の停止の規則。既定値は 03 の設計 §5・§6・§8 の値。

    - ``converged``：相対減少が ``patience_sweeps`` 掃引続けて
      ``relative_decrease_tolerance`` 未満、かつ掃引の終わりの s_max が
      ``stationarity_tolerance`` 以下。
    - ``max_sweeps``：``max_sweeps`` 掃引に達した。
    - ``local_solver_unresolved``：局所解の float64 相対残差が
      ``local_residual_tolerance`` に届かず、``retry_max_refinements`` で 1 度だけ
      解き直しても届かなかった。その候補は受理せず、最後に完了した掃引の終わりの
      状態を最終状態として止める。
    - ``monotone_tolerance`` は記録（``monotone_ok``）にだけ使い、停止には使わない。
    """

    relative_decrease_tolerance: float = 1e-7
    patience_sweeps: int = 5
    stationarity_tolerance: float = 1e-6
    max_sweeps: int = 1000
    local_residual_tolerance: float = 1e-8
    retry_max_refinements: int = 10
    monotone_tolerance: float = 1e-10
    checkpoint_every: int = 50

    def __post_init__(self) -> None:
        for name in (
            "relative_decrease_tolerance",
            "stationarity_tolerance",
            "local_residual_tolerance",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} は正である必要があります。")
        if not self.monotone_tolerance >= 0:
            raise ValueError("monotone_tolerance は 0 以上である必要があります。")
        for name in ("patience_sweeps", "max_sweeps", "retry_max_refinements", "checkpoint_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} は 1 以上の整数である必要があります。")


@dataclass(frozen=True)
class ConvergenceSweepRecord:
    """長い ALS の、完了した 1 掃引の記録。

    ``stationarity`` は、この掃引の終わりの状態（中心 0）から更新せずに測った s_k。
    更新ごとの量（``update_sites`` から ``accepted`` まで）は、更新した順に並ぶ。
    解き直した更新では、``cg_iterations`` と ``refinements`` は 2 回分の合計、
    ``cg_residuals`` は解き直したあとの値。

    時間：``seconds`` は掃引全体。ALS 本体の時間は
    ``seconds − seconds_stationarity − seconds_q_unregularized``。
    """

    sweep: int
    q_lambda: float
    q_unregularized: float | None
    relative_decrease: float
    small_decrease_streak: int
    monotone_ok: bool
    stationarity: tuple[float, ...]
    stationarity_max: float
    update_sites: tuple[int, ...]
    cg_iterations: tuple[int, ...]
    cg_residuals: tuple[float, ...]
    refinements: tuple[int, ...]
    retried: tuple[bool, ...]
    accepted: tuple[bool, ...]
    n_rejected: int
    seconds: float
    seconds_hvp: float
    seconds_preconditioner: float
    seconds_preconditioner_apply: float
    seconds_qr: float
    seconds_accept: float
    seconds_stationarity: float
    seconds_q_unregularized: float
    peak_memory_bytes: int | None
    stop_reason: str | None


@dataclass(frozen=True)
class LocalSolverFailure:
    """``local_solver_unresolved`` で止めた掃引の診断情報。最終状態には反映しない。

    ``update_*`` は、失敗した掃引の中で行った更新を順に並べたもの（最後が失敗した更新）。
    途中で受理した更新も含めて、最終状態（直前に完了した掃引の終わり）には入らない。
    ``q_lambda_before`` と ``q_lambda_candidate`` は、失敗した更新の直前の状態と、
    採用しなかった候補の Q_λ。
    """

    sweep: int
    site: int
    first_residual: float
    retry_residual: float
    cg_iterations: int
    refinements: int
    q_lambda_before: float
    q_lambda_candidate: float
    update_sites: tuple[int, ...]
    update_cg_iterations: tuple[int, ...]
    update_cg_residuals: tuple[float, ...]
    update_retried: tuple[bool, ...]
    update_accepted: tuple[bool, ...]


@dataclass(frozen=True)
class ConvergenceRunResult:
    """長い ALS の結果。``cores`` は CPU の float64 で、最後に完了した掃引の終わり（中心 0）。

    ``initial_q_lambda`` は、渡されたコアそのもの（正準化の前）の Q_λ。
    ``diagnostics`` は、``diagnostics`` 関数が保存のたびに返した値に ``"sweep"`` を
    足したもの。``save_timings`` は保存ごとの診断と書き込みの時間（保存ファイルの中では、
    その回の書き込み時間は None）。``resumed_from_sweep`` は再開したときの、
    すでに終わっていた掃引の数（新しく始めたときは None）。
    """

    cores: list[torch.Tensor]
    history: tuple[ConvergenceSweepRecord, ...]
    stop_reason: str
    initial_q_lambda: float
    initial_q_unregularized: float | None
    diagnostics: tuple[dict, ...]
    save_timings: tuple[dict, ...]
    failure: LocalSolverFailure | None
    resumed_from_sweep: int | None
    seconds_total: float


def _stationarity(
    current: Sequence[torch.Tensor],
    center: int,
    gram64: torch.Tensor,
    weight_gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> tuple[float, ...]:
    """更新せずに、正準中心を各サイトへ移して s_k を測る。``current`` は変えない。"""
    working = list(current)
    for site in range(center, 0, -1):
        working = move_tt_matrix_center(working, out_modes, in_modes, site, site - 1)
    values: list[float] = []
    for site in range(len(working)):
        if site:
            working = move_tt_matrix_center(working, out_modes, in_modes, site - 1, site)
        rhs = metric_tt_rhs_on_device(working, site, weight_gram, out_modes, in_modes)
        values.append(_relative_residual(working, site, working[site], rhs, gram64, out_modes, in_modes))
    return tuple(values)


def metric_tt_stationarity_on_device(
    cores: Sequence[torch.Tensor],
    weight: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    center: int,
    device: str | torch.device = "cuda",
) -> tuple[float, ...]:
    """掃引の終わりの局所の停留性 ``s_k = ‖b_k − H_k c_k‖ / ‖b_k‖``（全サイト）。

    ``cores`` は ``center`` を中心とする混合正準形であることを呼び出し側が保証する。
    コアは更新せず、QR で正準中心だけを各サイトへ移して、同じ状態で測る。
    ``‖b_k‖ = 0`` のサイトは 0 を返す。入力は変更しない。
    """
    if not 0 <= center < len(cores):
        raise ValueError(f"center={center} は 0 以上 {len(cores)} 未満である必要があります。")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA が使えません。")
    with _matmul_without_tf32():
        current = [core.detach().to(device=resolved, dtype=torch.float64).clone() for core in cores]
        weight64 = weight.detach().to(device=resolved, dtype=torch.float64)
        gram64 = gram.detach().to(device=resolved, dtype=torch.float64)
        return _stationarity(current, center, gram64, weight64 @ gram64, out_modes, in_modes)


def perturb_tt_matrix_site(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    site: int,
    relative_size: float,
    seed: int,
) -> tuple[list[torch.Tensor], float]:
    """1 つのサイトのコアだけに、大きさを決めた雑音を足す（03 の設計 §9 の診断用の摂動）。

    1. ``site`` を中心とする混合正準形にする。
    2. そのコアと同じ shape のガウス雑音 Z を、CPU の float64 で ``seed`` から作る。
    3. ``C' = C + relative_size · ‖C‖_F · Z / ‖Z‖_F``
    4. 中心 0 の混合正準形に戻す（ゲージの変換だけで、復元行列は変わらない）。

    ``site`` 中心の混合正準形では ``‖ΔW̃‖_F = ‖ΔC‖_F`` かつ ``‖W̃‖_F = ‖C‖_F`` なので、
    実現する相対摂動 δ_W は丸めを除いて ``relative_size`` に等しい。
    ALS は中心 0 から更新を始め、サイト 0 の局所問題の解はそのコアの初期値によらないので、
    ``site=0`` の摂動は最初の更新で消える。診断には ``site >= 1`` を使う。

    戻り値は（中心 0 のコア（CPU の float64）, 実際に測った δ_W）。入力は変更しない。
    """
    from .matrix import tt_matrix_to_dense

    if not 0 <= site < len(cores):
        raise ValueError(f"site={site} は 0 以上 {len(cores)} 未満である必要があります。")
    if not (math.isfinite(relative_size) and relative_size > 0):
        raise ValueError("relative_size は正の有限値である必要があります。")
    base = [core.detach().to(device="cpu", dtype=torch.float64).clone() for core in cores]
    canonical = canonicalize_tt_matrix_cores(base, out_modes, in_modes, site)
    generator = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(canonical[site].shape, generator=generator, dtype=torch.float64)
    noise = noise / torch.linalg.vector_norm(noise.reshape(-1))
    scale = relative_size * torch.linalg.vector_norm(canonical[site].reshape(-1))
    perturbed = list(canonical)
    perturbed[site] = canonical[site] + scale * noise
    reference = tt_matrix_to_dense(canonical, out_modes, in_modes)
    changed = tt_matrix_to_dense(perturbed, out_modes, in_modes)
    delta = torch.linalg.matrix_norm(changed - reference) / torch.linalg.matrix_norm(reference)
    return canonicalize_tt_matrix_cores(perturbed, out_modes, in_modes, 0), float(delta.item())


def _solver_config_record(config: MetricTTSolverConfig) -> dict:
    record = asdict(config)
    record["device"] = str(config.device)
    return record


def _same_solver_config(saved: object, expected: object) -> bool:
    """``timing_sync`` が無い古い保存は、既定の True として比較する。"""
    if not isinstance(saved, Mapping) or not isinstance(expected, Mapping):
        return False
    left = dict(saved)
    right = dict(expected)
    left.setdefault("timing_sync", True)
    right.setdefault("timing_sync", True)
    return left == right


def _input_hashes(
    cores: Sequence[torch.Tensor],
    weight: torch.Tensor,
    gram: torch.Tensor,
    report_gram: torch.Tensor | None,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> dict:
    from ..utils.reproducibility import tensor_sha256

    return {
        "initial_cores": [tensor_sha256(core) for core in cores],
        "weight": tensor_sha256(weight),
        "gram": tensor_sha256(gram),
        "report_gram": None if report_gram is None else tensor_sha256(report_gram),
        "out_modes": [int(mode) for mode in out_modes],
        "in_modes": [int(mode) for mode in in_modes],
    }


def _check_resumable(payload: dict, expected: dict) -> None:
    if payload.get("format") != CONVERGENCE_CHECKPOINT_FORMAT:
        raise ValueError(f"保存ファイルの形式が違います: {payload.get('format')!r}")
    mismatched = [
        key
        for key, value in expected.items()
        if not (_same_solver_config(payload.get(key), value) if key == "solver_config" else payload.get(key) == value)
    ]
    if mismatched:
        raise ValueError(f"保存ファイルと設定・入力が一致しないので再開しません: {mismatched}")


def run_metric_tt_als_until_converged(
    cores: Sequence[torch.Tensor],
    weight: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    config: MetricTTSolverConfig,
    rule: ConvergenceStopRule = ConvergenceStopRule(),
    *,
    report_gram: torch.Tensor | None = None,
    checkpoint_path: str | os.PathLike | None = None,
    metadata: Mapping[str, Any] | None = None,
    progress: Callable[[ConvergenceSweepRecord], None] | None = None,
    diagnostics: Callable[[list[torch.Tensor], int], Mapping[str, Any]] | None = None,
) -> ConvergenceRunResult:
    """停止の規則 ``rule`` で止める、正準形を保つ 1 サイト ALS。

    局所問題の解き方は ``run_metric_tt_als_on_device`` と同じ（掃引の順、受理の規則、
    ``config`` の局所ソルバー設定）。``config`` の ``max_sweeps``・``sweep_tolerance``・
    ``gradient_tolerance``・``time_limit_seconds`` は使わない。

    - ``gram`` は目的の G_λ。``report_gram`` を渡すと、正則化なしの Q も毎掃引記録する。
    - ``checkpoint_path`` を渡すと、``rule.checkpoint_every`` 掃引ごとと停止時に
      ``atomic_torch_save`` で保存する。ファイルがすでにあれば、設定・入力のハッシュ・
      ``metadata`` が一致することを確かめてから続きを回す（止まっていればそのまま返す）。
    - ``diagnostics(cores, sweep)`` は保存のたびに（保存先がなくても同じ掃引で）呼ばれ、
      返した値は記録と保存ファイルに入る。停止の判定には使わない。
    - ``progress`` には、完了した掃引ごとの記録を渡す。
    - ``local_solver_unresolved`` のときは、失敗した掃引を捨て、直前に完了した掃引の
      終わりの状態を最終状態にする。失敗の診断は ``failure`` に入る。

    入力のコア・重み・Gram は変更しない。
    """
    if not isinstance(config, MetricTTSolverConfig):
        raise TypeError("config は MetricTTSolverConfig である必要があります。")
    if not isinstance(rule, ConvergenceStopRule):
        raise TypeError("rule は ConvergenceStopRule である必要があります。")
    if config.cg_tolerance_mode != "fixed" or config.cg_relative_tolerance > rule.local_residual_tolerance:
        raise ValueError(
            "長い ALS では cg_tolerance_mode='fixed' かつ "
            "cg_relative_tolerance <= rule.local_residual_tolerance にしてください。"
        )
    device = _resolve_device(config)
    retry_config = replace(config, max_refinements=rule.retry_max_refinements)
    path = None if checkpoint_path is None else Path(checkpoint_path)
    expected = {
        "solver_config": _solver_config_record(config),
        "stop_rule": asdict(rule),
        "metadata": dict(metadata or {}),
        "input_hashes": _input_hashes(cores, weight, gram, report_gram, out_modes, in_modes),
    }
    payload: dict | None = None
    if path is not None and path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _check_resumable(payload, expected)

    def result_from(state: dict, resumed: int | None) -> ConvergenceRunResult:
        failure_state = state["failure"]
        return ConvergenceRunResult(
            cores=[core.to(dtype=torch.float64).contiguous() for core in state["cores"]],
            history=tuple(ConvergenceSweepRecord(**record) for record in state["history"]),
            stop_reason=state["stop_reason"],
            initial_q_lambda=state["initial_q_lambda"],
            initial_q_unregularized=state["initial_q_unregularized"],
            diagnostics=tuple(dict(item) for item in state["diagnostics"]),
            save_timings=tuple(dict(item) for item in state["save_timings"]),
            failure=None if failure_state is None else LocalSolverFailure(**failure_state),
            resumed_from_sweep=resumed,
            seconds_total=state["seconds_total"],
        )

    if payload is not None and payload["stop_reason"] is not None:
        return result_from(payload, payload["sweeps_completed"])

    started = time.perf_counter()
    with _matmul_without_tf32():
        weight64 = weight.detach().to(device=device, dtype=torch.float64)
        gram64 = gram.detach().to(device=device, dtype=torch.float64)
        gram32 = gram64.to(dtype=torch.float32) if config.precision == "float32_refine" else None
        report64 = None if report_gram is None else report_gram.detach().to(device=device, dtype=torch.float64)
        weight_gram = weight64 @ gram64

        def q_lambda_of(state: Sequence[torch.Tensor]) -> float:
            return float(_q_lambda(state, weight64, gram64, out_modes, in_modes).item())

        def q_unregularized_of(state: Sequence[torch.Tensor]) -> float | None:
            if report64 is None:
                return None
            return float(_q_lambda(state, weight64, report64, out_modes, in_modes).item())

        if payload is None:
            current = [core.detach().to(device=device, dtype=torch.float64).clone() for core in cores]
            initial_q = q_lambda_of(current)
            initial_q_unregularized = q_unregularized_of(current)
            initial_qr = _Accumulator(device, sync=config.timing_sync)
            with initial_qr:
                current = canonicalize_tt_matrix_cores(current, out_modes, in_modes, 0)
            q_previous = q_lambda_of(current)
            history: list[ConvergenceSweepRecord] = []
            saved_diagnostics: list[dict] = []
            save_timings: list[dict] = []
            streak = 0
            done = 0
            seconds_before = 0.0
            initial_qr_seconds = initial_qr.total
            resumed: int | None = None
        else:
            current = [core.to(device=device, dtype=torch.float64) for core in payload["cores"]]
            initial_q = payload["initial_q_lambda"]
            initial_q_unregularized = payload["initial_q_unregularized"]
            q_previous = payload["q_lambda_last"]
            history = [ConvergenceSweepRecord(**record) for record in payload["history"]]
            saved_diagnostics = [dict(item) for item in payload["diagnostics"]]
            save_timings = [dict(item) for item in payload["save_timings"]]
            streak = payload["small_decrease_streak"]
            done = payload["sweeps_completed"]
            seconds_before = payload["seconds_total"]
            initial_qr_seconds = 0.0
            resumed = done

        factor_cache: dict[tuple[int, int, int], KroneckerMetricFactors] = {}
        n_sites = len(current)
        stop_reason: str | None = None
        failure: LocalSolverFailure | None = None

        def save(sweeps_completed: int) -> None:
            entry: dict = {"sweep": sweeps_completed, "seconds_diagnostics": 0.0, "seconds_save": None}
            already = bool(saved_diagnostics) and saved_diagnostics[-1]["sweep"] == sweeps_completed
            if diagnostics is not None and not already:
                clock = time.perf_counter()
                values = dict(diagnostics([core.detach().to("cpu") for core in current], sweeps_completed))
                values["sweep"] = sweeps_completed
                saved_diagnostics.append(values)
                entry["seconds_diagnostics"] = time.perf_counter() - clock
            save_timings.append(entry)
            if path is None:
                return
            from ..utils.artifacts import atomic_torch_save

            clock = time.perf_counter()
            state = {
                "format": CONVERGENCE_CHECKPOINT_FORMAT,
                **expected,
                "cores": [core.detach().to(device="cpu", dtype=torch.float64).contiguous() for core in current],
                "sweeps_completed": sweeps_completed,
                "history": [asdict(record) for record in history],
                "small_decrease_streak": streak,
                "q_lambda_last": q_previous,
                "initial_q_lambda": initial_q,
                "initial_q_unregularized": initial_q_unregularized,
                "diagnostics": saved_diagnostics,
                "save_timings": save_timings,
                "failure": None if failure is None else asdict(failure),
                "seconds_total": seconds_before + time.perf_counter() - started,
                "stop_reason": stop_reason,
            }
            atomic_torch_save(path, state)
            entry["seconds_save"] = time.perf_counter() - clock

        for sweep in range(done + 1, rule.max_sweeps + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            hvp_time = _Accumulator(device, sync=config.timing_sync)
            preconditioner_time = _Accumulator(device, sync=config.timing_sync)
            preconditioner_apply_time = _Accumulator(device, sync=config.timing_sync)
            qr_time = _Accumulator(device, sync=config.timing_sync)
            accept_time = _Accumulator(device, sync=config.timing_sync)
            stationarity_time = _Accumulator(device, sync=config.timing_sync)
            unregularized_time = _Accumulator(device, sync=config.timing_sync)
            sweep_clock = time.perf_counter()
            sweep_start = list(current)
            sites: list[int] = []
            iterations: list[int] = []
            residuals: list[float] = []
            refinements: list[int] = []
            retried: list[bool] = []
            accepted_flags: list[bool] = []

            def update_site(site: int) -> bool:
                """1 サイトを更新する。解き直しても届かなかったら候補を捨てて True を返す。"""
                nonlocal current, failure
                arguments = (site, gram64, gram32, weight_gram, out_modes, in_modes)
                timers = (factor_cache, hvp_time, preconditioner_time, preconditioner_apply_time)
                solution, n_iter, residual, _grad, _tol, n_ref, _err, _clip = _solve_site(
                    current, *arguments, config, *timers
                )
                first_residual = residual
                was_retried = not residual <= rule.local_residual_tolerance
                if was_retried:
                    solution, n_iter2, residual, _grad, _tol, n_ref2, _err, _clip = _solve_site(
                        current, *arguments, retry_config, *timers
                    )
                    n_iter += n_iter2
                    n_ref += n_ref2
                resolved = residual <= rule.local_residual_tolerance
                with accept_time:
                    candidate = list(current)
                    candidate[site] = solution
                    q_before = q_lambda_of(current)
                    q_after = q_lambda_of(candidate)
                    accepted = resolved and q_after <= q_before
                if accepted:
                    current = candidate
                sites.append(site)
                iterations.append(n_iter)
                residuals.append(residual)
                refinements.append(n_ref)
                retried.append(was_retried)
                accepted_flags.append(accepted)
                if not resolved:
                    failure = LocalSolverFailure(
                        sweep=sweep,
                        site=site,
                        first_residual=first_residual,
                        retry_residual=residual,
                        cg_iterations=n_iter,
                        refinements=n_ref,
                        q_lambda_before=q_before,
                        q_lambda_candidate=q_after,
                        update_sites=tuple(sites),
                        update_cg_iterations=tuple(iterations),
                        update_cg_residuals=tuple(residuals),
                        update_retried=tuple(retried),
                        update_accepted=tuple(accepted_flags),
                    )
                return not resolved

            unresolved = False
            for site in range(n_sites):
                unresolved = update_site(site)
                if unresolved:
                    break
                if site < n_sites - 1:
                    with qr_time:
                        current = move_tt_matrix_center(current, out_modes, in_modes, site, site + 1)
            if not unresolved:
                for site in range(n_sites - 2, -1, -1):
                    with qr_time:
                        current = move_tt_matrix_center(current, out_modes, in_modes, site + 1, site)
                    unresolved = update_site(site)
                    if unresolved:
                        break
            if unresolved:
                # 失敗した掃引は捨て、直前に完了した掃引の終わりの状態を最終状態にする。
                current = sweep_start
                stop_reason = "local_solver_unresolved"
                save(sweep - 1)
                break
            q_next = q_lambda_of(current)
            with unregularized_time:
                q_unregularized = q_unregularized_of(current)
            relative_decrease = 0.0 if q_previous == 0.0 else (q_previous - q_next) / q_previous
            monotone_ok = q_next <= q_previous * (1.0 + rule.monotone_tolerance)
            streak = streak + 1 if relative_decrease < rule.relative_decrease_tolerance else 0
            with stationarity_time:
                stationarity = _stationarity(current, 0, gram64, weight_gram, out_modes, in_modes)
            stationarity_max = max(stationarity)
            if streak >= rule.patience_sweeps and stationarity_max <= rule.stationarity_tolerance:
                stop_reason = "converged"
            elif sweep >= rule.max_sweeps:
                stop_reason = "max_sweeps"
            _sync(device)
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            record = ConvergenceSweepRecord(
                sweep=sweep,
                q_lambda=q_next,
                q_unregularized=q_unregularized,
                relative_decrease=relative_decrease,
                small_decrease_streak=streak,
                monotone_ok=monotone_ok,
                stationarity=stationarity,
                stationarity_max=stationarity_max,
                update_sites=tuple(sites),
                cg_iterations=tuple(iterations),
                cg_residuals=tuple(residuals),
                refinements=tuple(refinements),
                retried=tuple(retried),
                accepted=tuple(accepted_flags),
                n_rejected=accepted_flags.count(False),
                seconds=time.perf_counter() - sweep_clock,
                seconds_hvp=hvp_time.total,
                seconds_preconditioner=preconditioner_time.total,
                seconds_preconditioner_apply=preconditioner_apply_time.total,
                seconds_qr=qr_time.total + (initial_qr_seconds if sweep == 1 else 0.0),
                seconds_accept=accept_time.total,
                seconds_stationarity=stationarity_time.total,
                seconds_q_unregularized=unregularized_time.total,
                peak_memory_bytes=peak,
                stop_reason=stop_reason,
            )
            history.append(record)
            q_previous = q_next
            if progress is not None:
                progress(record)
            if stop_reason is not None or sweep % rule.checkpoint_every == 0:
                save(sweep)
            if stop_reason is not None:
                break
        final_state = {
            "cores": [core.detach().to(device="cpu", dtype=torch.float64).contiguous() for core in current],
            "history": [asdict(record) for record in history],
            "stop_reason": stop_reason,
            "initial_q_lambda": initial_q,
            "initial_q_unregularized": initial_q_unregularized,
            "diagnostics": saved_diagnostics,
            "save_timings": save_timings,
            "failure": None if failure is None else asdict(failure),
            "seconds_total": seconds_before + time.perf_counter() - started,
        }
    return result_from(final_state, resumed)


STUDY_CHECKPOINT_FORMAT = "metric_tt_study_checkpoint/2"
# 03 の記録用フラグと同じ幅。停止の判定には使わない。
_STUDY_MONOTONE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class StudyResolutionStopRule:
    """研究上の分解能で止める規則。

    ``d_ref`` は呼び出し側が計算する正の値。
    停止の判断は ``window_sweeps`` の倍数だけで行い、s_max や 1 掃引の相対減少では止めない。
    """

    d_ref: float
    window_sweeps: int = 50
    resolution_tolerance: float = 1e-4
    consecutive_windows: int = 2
    max_sweeps: int = 1000
    local_residual_tolerance: float = 1e-8
    escalation_refinements: tuple[int, ...] = (10, 50)
    warn_residual_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not math.isfinite(self.d_ref) or self.d_ref <= 0:
            raise ValueError(f"d_ref は正の有限値である必要があります: {self.d_ref!r}")
        if isinstance(self.window_sweeps, bool) or not isinstance(self.window_sweeps, int) or self.window_sweeps < 1:
            raise ValueError(f"window_sweeps は 1 以上の int である必要があります: {self.window_sweeps!r}")
        if (
            isinstance(self.consecutive_windows, bool)
            or not isinstance(self.consecutive_windows, int)
            or self.consecutive_windows < 1
        ):
            raise ValueError(
                f"consecutive_windows は 1 以上の int である必要があります: {self.consecutive_windows!r}"
            )
        if isinstance(self.max_sweeps, bool) or not isinstance(self.max_sweeps, int) or self.max_sweeps < 1:
            raise ValueError(f"max_sweeps は 1 以上の int である必要があります: {self.max_sweeps!r}")
        if self.resolution_tolerance <= 0 or self.local_residual_tolerance <= 0 or self.warn_residual_tolerance <= 0:
            raise ValueError("許容誤差は正である必要があります。")
        refinements = self.escalation_refinements
        if (
            isinstance(refinements, (str, bytes))
            or len(tuple(refinements)) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in refinements)
        ):
            raise ValueError(
                "escalation_refinements は、2 段目と 3 段目の外側反復の累計上限を表す長さ 2 の int 列である必要があります。"
                f" {refinements!r}"
            )
        pair = tuple(int(item) for item in refinements)
        if pair[0] >= pair[1]:
            raise ValueError(
                "escalation_refinements は累計の上限で、2 段目より 3 段目が大きい必要があります。"
                f" {pair!r}"
            )
        object.__setattr__(self, "escalation_refinements", pair)


@dataclass(frozen=True)
class StudyStageRecord:
    """局所問題の 1 段。``q_lambda`` は 3 段目まで進んだときの 2 段目・3 段目だけ入る。

    ``cg_idle_iterations`` は、区切りの CG が印のあと空回りした回数。x は進めていない。
    ``initial_residual`` は、その段を始めるときに測り直した相対残差。
    ``cumulative_refinements`` は、その段が終わった時点の外側反復の累計。
    ``max_refinements`` は、その段に渡した追加の上限。
    """

    stage: int
    max_refinements: int
    residual: float
    cg_iterations: int
    refinements: int
    seconds: float
    cg_statuses: tuple[CgStatusName, ...]
    nonpositive_curvature_count: int
    q_lambda: float | None
    cumulative_refinements: int
    initial_residual: float
    cg_idle_iterations: int = 0


@dataclass(frozen=True)
class StudyUpdateRecord:
    """1 回のサイト更新。``outcome`` は ``target_met`` / ``local_target_missed`` / ``local_solver_unresolved``。"""

    site: int
    stages: tuple[StudyStageRecord, ...]
    outcome: str
    accepted: bool
    confirmation_a: float | None


@dataclass(frozen=True)
class StudySweepRecord:
    """完了した 1 掃引。``delta_50`` は窓の境界だけで、それ以外は None。"""

    sweep: int
    q_lambda: float
    q_unregularized: float | None
    relative_decrease: float
    monotone_ok: bool
    stationarity: tuple[float, ...]
    stationarity_max: float
    delta_50: float | None
    consecutive_small_windows: int
    n_warnings: int
    n_stage2: int
    n_stage3: int
    updates: tuple[StudyUpdateRecord, ...]
    cg_iterations: tuple[int, ...]
    cg_residuals: tuple[float, ...]
    hvp_calls: int
    n_rejected: int
    seconds: float
    seconds_hvp: float
    seconds_preconditioner: float
    seconds_preconditioner_apply: float
    seconds_qr: float
    seconds_accept: float
    seconds_stationarity: float
    seconds_q_unregularized: float
    peak_memory_bytes: int | None
    stop_reason: str | None


@dataclass(frozen=True)
class StudyLocalFailure:
    """解けなかった局所問題。失敗した掃引は最終状態に入れない。"""

    sweep: int
    site: int
    stages: tuple[StudyStageRecord, ...]
    q_lambda_before: float
    q_lambda_candidate: float
    update_sites: tuple[int, ...]
    update_outcomes: tuple[str, ...]
    update_accepted: tuple[bool, ...]


@dataclass(frozen=True)
class StudyRunResult:
    """``run_metric_tt_als_study`` の結果。コアは最後に完了した掃引の終わり。"""

    cores: list[torch.Tensor]
    history: tuple[StudySweepRecord, ...]
    stop_reason: str
    initial_q_lambda: float
    initial_q_unregularized: float | None
    d_ref: float
    diagnostics: tuple[dict, ...]
    save_timings: tuple[dict, ...]
    failure: StudyLocalFailure | None
    resumed_from_sweep: int | None
    seconds_total: float
    n_warnings: int
    n_stage2: int
    n_stage3: int


def study_resolution_delta(q_before_window: float, q_after_window: float, d_ref: float) -> float:
    """Δ = (Q_λ(s − 窓) − Q_λ(s)) / D_ref。"""
    if not math.isfinite(d_ref) or d_ref <= 0:
        raise ValueError(f"d_ref は正の有限値である必要があります: {d_ref!r}")
    return (q_before_window - q_after_window) / d_ref


def decide_study_stop(
    q_history: Sequence[float],
    rule: StudyResolutionStopRule,
    *,
    unresolved: bool = False,
) -> tuple[str | None, float | None, int]:
    """掃引 0 から現在までの Q_λ で、停止理由と直近の Δ、連続回数を返す。

    ``q_history[0]`` は正準化する前の Q_λ、``q_history[s]`` は掃引 s の終わり。
    窓の境界でなければ Δ は None。連続回数は、最後に完了した窓までの値。
    ``unresolved=True`` のときは Q の列を見ずに ``local_solver_unresolved`` を返す。
    分解能と掃引上限が同時に成り立つときは ``study_resolution_reached`` を優先する。
    """
    if not isinstance(rule, StudyResolutionStopRule):
        raise TypeError("rule は StudyResolutionStopRule である必要があります。")
    if unresolved:
        return "local_solver_unresolved", None, 0
    sweep = len(q_history) - 1
    if sweep < 0:
        raise ValueError("q_history には Q_λ(0) が必要です。")
    streak = 0
    last_delta: float | None = None
    window = rule.window_sweeps
    for end in range(window, sweep + 1, window):
        delta = study_resolution_delta(q_history[end - window], q_history[end], rule.d_ref)
        streak = streak + 1 if delta < rule.resolution_tolerance else 0
        last_delta = delta
    on_boundary = sweep > 0 and sweep % window == 0
    reason: str | None = None
    if on_boundary and streak >= rule.consecutive_windows:
        reason = "study_resolution_reached"
    elif sweep >= rule.max_sweeps:
        reason = "max_sweeps"
    return reason, (last_delta if on_boundary else None), streak


def _study_stage_from_dict(item: Mapping[str, Any]) -> StudyStageRecord:
    q_lambda = item["q_lambda"]
    return StudyStageRecord(
        stage=int(item["stage"]),
        max_refinements=int(item["max_refinements"]),
        residual=float(item["residual"]),
        cg_iterations=int(item["cg_iterations"]),
        refinements=int(item["refinements"]),
        seconds=float(item["seconds"]),
        cg_statuses=tuple(item["cg_statuses"]),
        nonpositive_curvature_count=int(item["nonpositive_curvature_count"]),
        q_lambda=None if q_lambda is None else float(q_lambda),
        cumulative_refinements=int(item["cumulative_refinements"]),
        initial_residual=float(item["initial_residual"]),
        cg_idle_iterations=int(item.get("cg_idle_iterations", 0)),
    )


def _study_update_from_dict(item: Mapping[str, Any]) -> StudyUpdateRecord:
    confirmation = item["confirmation_a"]
    return StudyUpdateRecord(
        site=int(item["site"]),
        stages=tuple(_study_stage_from_dict(stage) for stage in item["stages"]),
        outcome=str(item["outcome"]),
        accepted=bool(item["accepted"]),
        confirmation_a=None if confirmation is None else float(confirmation),
    )


def _study_sweep_from_dict(item: Mapping[str, Any]) -> StudySweepRecord:
    delta = item["delta_50"]
    q_unregularized = item["q_unregularized"]
    peak = item["peak_memory_bytes"]
    return StudySweepRecord(
        sweep=int(item["sweep"]),
        q_lambda=float(item["q_lambda"]),
        q_unregularized=None if q_unregularized is None else float(q_unregularized),
        relative_decrease=float(item["relative_decrease"]),
        monotone_ok=bool(item["monotone_ok"]),
        stationarity=tuple(float(value) for value in item["stationarity"]),
        stationarity_max=float(item["stationarity_max"]),
        delta_50=None if delta is None else float(delta),
        consecutive_small_windows=int(item["consecutive_small_windows"]),
        n_warnings=int(item["n_warnings"]),
        n_stage2=int(item["n_stage2"]),
        n_stage3=int(item["n_stage3"]),
        updates=tuple(_study_update_from_dict(update) for update in item["updates"]),
        cg_iterations=tuple(int(value) for value in item["cg_iterations"]),
        cg_residuals=tuple(float(value) for value in item["cg_residuals"]),
        hvp_calls=int(item["hvp_calls"]),
        n_rejected=int(item["n_rejected"]),
        seconds=float(item["seconds"]),
        seconds_hvp=float(item["seconds_hvp"]),
        seconds_preconditioner=float(item["seconds_preconditioner"]),
        seconds_preconditioner_apply=float(item["seconds_preconditioner_apply"]),
        seconds_qr=float(item["seconds_qr"]),
        seconds_accept=float(item["seconds_accept"]),
        seconds_stationarity=float(item["seconds_stationarity"]),
        seconds_q_unregularized=float(item["seconds_q_unregularized"]),
        peak_memory_bytes=None if peak is None else int(peak),
        stop_reason=None if item["stop_reason"] is None else str(item["stop_reason"]),
    )


def _study_failure_from_dict(item: Mapping[str, Any] | None) -> StudyLocalFailure | None:
    if item is None:
        return None
    return StudyLocalFailure(
        sweep=int(item["sweep"]),
        site=int(item["site"]),
        stages=tuple(_study_stage_from_dict(stage) for stage in item["stages"]),
        q_lambda_before=float(item["q_lambda_before"]),
        q_lambda_candidate=float(item["q_lambda_candidate"]),
        update_sites=tuple(int(site) for site in item["update_sites"]),
        update_outcomes=tuple(str(outcome) for outcome in item["update_outcomes"]),
        update_accepted=tuple(bool(flag) for flag in item["update_accepted"]),
    )


def _check_study_resumable(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if payload.get("format") != STUDY_CHECKPOINT_FORMAT:
        raise ValueError(f"保存ファイルの形式が違います: {payload.get('format')!r}")
    mismatched = [
        key
        for key, value in expected.items()
        if not (_same_solver_config(payload.get(key), value) if key == "solver_config" else payload.get(key) == value)
    ]
    if mismatched:
        raise ValueError(f"保存ファイルと今回の入力が違います: {mismatched}")


def run_metric_tt_als_study(
    cores: Sequence[torch.Tensor],
    weight: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    config: MetricTTSolverConfig,
    rule: StudyResolutionStopRule,
    *,
    report_gram: torch.Tensor | None = None,
    checkpoint_path: str | os.PathLike | None = None,
    metadata: Mapping[str, Any] | None = None,
    progress: Callable[[StudySweepRecord], None] | None = None,
    diagnostics: Callable[[list[torch.Tensor], int], Mapping[str, Any]] | None = None,
) -> StudyRunResult:
    """``StudyResolutionStopRule`` で止める 1 サイト ALS。

    掃引の順、正準形、Q_λ が増えない更新だけを受理する点は
    ``run_metric_tt_als_until_converged`` と同じ。局所問題は 1 段目を
    ``config.max_refinements`` 回まで解く。届かなければ、その解から続けて
    累計が ``rule.escalation_refinements`` になるまで足す。右辺・ヘッセ・前処理は
    中心以外のコアと G で決まるので、段が変わっても作り直した値は同じである。

    目標の残差に届いた更新は受理の判定へ進む。3 段目が ``warn_residual_tolerance``
    以下なら ``local_target_missed`` を記録して、同じ受理の判定へ進む。
    それを超える、値が有限でない、float64 の CG で曲率が 0 以下なら
    ``local_solver_unresolved`` で止め、その掃引は捨てる。
    float32 の内側 CG の曲率 0 以下と、``breakdown``（内積が 0。float64 も同じ）は
    記録するだけで、それだけでは ``local_solver_unresolved`` にしない。

    保存は窓の倍数と停止時だけ。止まったファイルは、設定と入力が一致すれば計算し直さない。
    """
    if not isinstance(config, MetricTTSolverConfig):
        raise TypeError("config は MetricTTSolverConfig である必要があります。")
    if not isinstance(rule, StudyResolutionStopRule):
        raise TypeError("rule は StudyResolutionStopRule である必要があります。")
    if config.cg_tolerance_mode != "fixed" or config.cg_relative_tolerance > rule.local_residual_tolerance:
        raise ValueError(
            "研究用 ALS では cg_tolerance_mode='fixed' かつ "
            "cg_relative_tolerance <= rule.local_residual_tolerance にしてください。"
        )
    cumulative_caps = (config.max_refinements, *rule.escalation_refinements)
    if not (cumulative_caps[0] < cumulative_caps[1] < cumulative_caps[2]):
        raise ValueError(
            "外側反復の累計の上限は段ごとに大きくする必要があります: "
            f"max_refinements={config.max_refinements}, escalation_refinements={rule.escalation_refinements}"
        )
    device = _resolve_device(config)
    path = None if checkpoint_path is None else Path(checkpoint_path)
    expected = {
        "solver_config": _solver_config_record(config),
        "stop_rule": asdict(rule),
        "metadata": dict(metadata or {}),
        "input_hashes": _input_hashes(cores, weight, gram, report_gram, out_modes, in_modes),
    }
    payload: dict | None = None
    if path is not None and path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _check_study_resumable(payload, expected)

    def result_from(state: Mapping[str, Any], resumed: int | None) -> StudyRunResult:
        return StudyRunResult(
            cores=[core.to(dtype=torch.float64).contiguous() for core in state["cores"]],
            history=tuple(_study_sweep_from_dict(record) for record in state["history"]),
            stop_reason=str(state["stop_reason"]),
            initial_q_lambda=float(state["initial_q_lambda"]),
            initial_q_unregularized=(
                None if state["initial_q_unregularized"] is None else float(state["initial_q_unregularized"])
            ),
            d_ref=rule.d_ref,
            diagnostics=tuple(dict(item) for item in state["diagnostics"]),
            save_timings=tuple(dict(item) for item in state["save_timings"]),
            failure=_study_failure_from_dict(state["failure"]),
            resumed_from_sweep=resumed,
            seconds_total=float(state["seconds_total"]),
            n_warnings=int(state["n_warnings"]),
            n_stage2=int(state["n_stage2"]),
            n_stage3=int(state["n_stage3"]),
        )

    if payload is not None and payload["stop_reason"] is not None:
        return result_from(payload, payload["sweeps_completed"])

    started = time.perf_counter()
    with _matmul_without_tf32():
        weight64 = weight.detach().to(device=device, dtype=torch.float64)
        gram64 = gram.detach().to(device=device, dtype=torch.float64)
        gram32 = gram64.to(dtype=torch.float32) if config.precision == "float32_refine" else None
        report64 = None if report_gram is None else report_gram.detach().to(device=device, dtype=torch.float64)
        weight_gram = weight64 @ gram64

        def q_lambda_of(state: Sequence[torch.Tensor]) -> float:
            return float(_q_lambda(state, weight64, gram64, out_modes, in_modes).item())

        def q_unregularized_of(state: Sequence[torch.Tensor]) -> float | None:
            if report64 is None:
                return None
            return float(_q_lambda(state, weight64, report64, out_modes, in_modes).item())

        if payload is None:
            current = [core.detach().to(device=device, dtype=torch.float64).clone() for core in cores]
            initial_q = q_lambda_of(current)
            initial_q_unregularized = q_unregularized_of(current)
            initial_qr = _Accumulator(device, sync=config.timing_sync)
            with initial_qr:
                current = canonicalize_tt_matrix_cores(current, out_modes, in_modes, 0)
            q_previous = q_lambda_of(current)
            history: list[StudySweepRecord] = []
            saved_diagnostics: list[dict] = []
            save_timings: list[dict] = []
            done = 0
            seconds_before = 0.0
            initial_qr_seconds = initial_qr.total
            resumed: int | None = None
            n_warnings = 0
            n_stage2 = 0
            n_stage3 = 0
        else:
            current = [core.to(device=device, dtype=torch.float64) for core in payload["cores"]]
            initial_q = float(payload["initial_q_lambda"])
            initial_q_unregularized = payload["initial_q_unregularized"]
            q_previous = float(payload["q_lambda_last"])
            history = [_study_sweep_from_dict(record) for record in payload["history"]]
            saved_diagnostics = [dict(item) for item in payload["diagnostics"]]
            save_timings = [dict(item) for item in payload["save_timings"]]
            done = int(payload["sweeps_completed"])
            seconds_before = float(payload["seconds_total"])
            initial_qr_seconds = 0.0
            resumed = done
            n_warnings = int(payload["n_warnings"])
            n_stage2 = int(payload["n_stage2"])
            n_stage3 = int(payload["n_stage3"])

        q_history = [initial_q, *[record.q_lambda for record in history]]
        factor_cache: dict[tuple[int, int, int], KroneckerMetricFactors] = {}
        n_sites = len(current)
        stop_reason: str | None = None
        failure: StudyLocalFailure | None = None
        stage_caps = cumulative_caps

        def save(sweeps_completed: int) -> None:
            entry: dict = {"sweep": sweeps_completed, "seconds_diagnostics": 0.0, "seconds_save": None}
            already = bool(saved_diagnostics) and saved_diagnostics[-1]["sweep"] == sweeps_completed
            if diagnostics is not None and not already:
                clock = time.perf_counter()
                values = dict(diagnostics([core.detach().to("cpu") for core in current], sweeps_completed))
                values["sweep"] = sweeps_completed
                saved_diagnostics.append(values)
                entry["seconds_diagnostics"] = time.perf_counter() - clock
            save_timings.append(entry)
            if path is None:
                return
            from ..utils.artifacts import atomic_torch_save

            clock = time.perf_counter()
            state = {
                "format": STUDY_CHECKPOINT_FORMAT,
                **expected,
                "cores": [core.detach().to(device="cpu", dtype=torch.float64).contiguous() for core in current],
                "sweeps_completed": sweeps_completed,
                "history": [asdict(record) for record in history],
                "q_lambda_last": q_previous,
                "initial_q_lambda": initial_q,
                "initial_q_unregularized": initial_q_unregularized,
                "diagnostics": saved_diagnostics,
                "save_timings": save_timings,
                "failure": None if failure is None else asdict(failure),
                "seconds_total": seconds_before + time.perf_counter() - started,
                "stop_reason": stop_reason,
                "n_warnings": n_warnings,
                "n_stage2": n_stage2,
                "n_stage3": n_stage3,
                "consecutive_small_windows": 0 if not history else history[-1].consecutive_small_windows,
                "delta_50": None if not history else history[-1].delta_50,
            }
            atomic_torch_save(path, state)
            entry["seconds_save"] = time.perf_counter() - clock

        for sweep in range(done + 1, rule.max_sweeps + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            hvp_time = _Accumulator(device, sync=config.timing_sync)
            preconditioner_time = _Accumulator(device, sync=config.timing_sync)
            preconditioner_apply_time = _Accumulator(device, sync=config.timing_sync)
            qr_time = _Accumulator(device, sync=config.timing_sync)
            accept_time = _Accumulator(device, sync=config.timing_sync)
            stationarity_time = _Accumulator(device, sync=config.timing_sync)
            unregularized_time = _Accumulator(device, sync=config.timing_sync)
            hvp_counter = [0]
            hvp_token = _hvp_calls.set(hvp_counter)
            sweep_clock = time.perf_counter()
            sweep_start = list(current)
            updates: list[StudyUpdateRecord] = []

            def update_site(site: int) -> bool:
                """1 サイトを更新する。解けなかったら True。"""
                nonlocal current, failure, n_warnings, n_stage2, n_stage3
                arguments = (site, gram64, gram32, weight_gram, out_modes, in_modes)
                timers = (factor_cache, hvp_time, preconditioner_time, preconditioner_apply_time)
                stages: list[StudyStageRecord] = []
                solutions: list[torch.Tensor] = []
                outcome = "local_solver_unresolved"
                used_refinements = 0
                for stage_index, cumulative_cap in enumerate(stage_caps, start=1):
                    statuses: list[CgStatusName] = []
                    idle_counts: list[int] = []
                    budget = cumulative_cap - used_refinements
                    stage_cores = list(current)
                    if solutions:
                        stage_cores[site] = solutions[-1]
                    stage_config = replace(config, max_refinements=budget)
                    stage_clock = time.perf_counter()
                    solution, n_iter, residual, initial_residual, _tol, n_ref, _err, _clip = _solve_site(
                        stage_cores,
                        *arguments,
                        stage_config,
                        *timers,
                        cg_statuses=statuses,
                        cg_idle=idle_counts,
                    )
                    used_refinements += n_ref
                    stages.append(
                        StudyStageRecord(
                            stage=stage_index,
                            max_refinements=budget,
                            residual=residual,
                            cg_iterations=n_iter,
                            refinements=n_ref,
                            seconds=time.perf_counter() - stage_clock,
                            cg_statuses=tuple(statuses),
                            nonpositive_curvature_count=sum(
                                status == "nonpositive_curvature" for status in statuses
                            ),
                            q_lambda=None,
                            cumulative_refinements=used_refinements,
                            initial_residual=initial_residual,
                            cg_idle_iterations=sum(idle_counts),
                        )
                    )
                    solutions.append(solution)
                    # breakdown はちょうど解けた印なので、未解決の条件に入れない。
                    nonfinite = (not math.isfinite(residual)) or ("nonfinite" in statuses)
                    float64_curvature = config.precision == "float64" and "nonpositive_curvature" in statuses
                    if nonfinite or float64_curvature:
                        outcome = "local_solver_unresolved"
                        break
                    if residual <= rule.local_residual_tolerance:
                        outcome = "target_met"
                        break
                    if stage_index == len(stage_caps):
                        if residual <= rule.warn_residual_tolerance:
                            outcome = "local_target_missed"
                        else:
                            outcome = "local_solver_unresolved"
                confirmation: float | None = None
                with accept_time:
                    if len(solutions) == len(stage_caps):
                        q_by_stage: list[float | None] = [None] * len(stages)
                        for stage_index in (2, 3):
                            candidate_cores = list(current)
                            candidate_cores[site] = solutions[stage_index - 1]
                            q_by_stage[stage_index - 1] = q_lambda_of(candidate_cores)
                        stages = [
                            replace(stage, q_lambda=q_value) for stage, q_value in zip(stages, q_by_stage, strict=True)
                        ]
                        q2 = q_by_stage[1]
                        q3 = q_by_stage[2]
                        if q2 is not None and q3 is not None and math.isfinite(q2) and math.isfinite(q3):
                            confirmation = (q2 - q3) / rule.d_ref
                    candidate = list(current)
                    candidate[site] = solutions[-1]
                    q_before = q_lambda_of(current)
                    q_after = q_lambda_of(candidate)
                    if outcome != "local_solver_unresolved" and not math.isfinite(q_after):
                        outcome = "local_solver_unresolved"
                    accepted = outcome != "local_solver_unresolved" and q_after <= q_before
                    if accepted:
                        current = candidate
                if len(stages) >= 2:
                    n_stage2 += 1
                if len(stages) >= 3:
                    n_stage3 += 1
                if outcome == "local_target_missed":
                    n_warnings += 1
                updates.append(
                    StudyUpdateRecord(
                        site=site,
                        stages=tuple(stages),
                        outcome=outcome,
                        accepted=accepted,
                        confirmation_a=confirmation,
                    )
                )
                if outcome == "local_solver_unresolved":
                    failure = StudyLocalFailure(
                        sweep=sweep,
                        site=site,
                        stages=tuple(stages),
                        q_lambda_before=q_before,
                        q_lambda_candidate=q_after,
                        update_sites=tuple(update.site for update in updates),
                        update_outcomes=tuple(update.outcome for update in updates),
                        update_accepted=tuple(update.accepted for update in updates),
                    )
                    return True
                return False

            try:
                unresolved = False
                for site in range(n_sites):
                    unresolved = update_site(site)
                    if unresolved:
                        break
                    if site < n_sites - 1:
                        with qr_time:
                            current = move_tt_matrix_center(current, out_modes, in_modes, site, site + 1)
                if not unresolved:
                    for site in range(n_sites - 2, -1, -1):
                        with qr_time:
                            current = move_tt_matrix_center(current, out_modes, in_modes, site + 1, site)
                        unresolved = update_site(site)
                        if unresolved:
                            break
                if unresolved:
                    current = sweep_start
                    stop_reason = "local_solver_unresolved"
                    save(sweep - 1)
                    break
                q_next = q_lambda_of(current)
                with unregularized_time:
                    q_unregularized = q_unregularized_of(current)
                relative_decrease = 0.0 if q_previous == 0.0 else (q_previous - q_next) / q_previous
                monotone_ok = q_next <= q_previous * (1.0 + _STUDY_MONOTONE_TOLERANCE)
                q_history.append(q_next)
                stop_reason, delta_50, streak = decide_study_stop(q_history, rule)
                with stationarity_time:
                    stationarity = _stationarity(current, 0, gram64, weight_gram, out_modes, in_modes)
                _sync(device)
                peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
                record = StudySweepRecord(
                    sweep=sweep,
                    q_lambda=q_next,
                    q_unregularized=q_unregularized,
                    relative_decrease=relative_decrease,
                    monotone_ok=monotone_ok,
                    stationarity=stationarity,
                    stationarity_max=max(stationarity),
                    delta_50=delta_50,
                    consecutive_small_windows=streak,
                    n_warnings=n_warnings,
                    n_stage2=n_stage2,
                    n_stage3=n_stage3,
                    updates=tuple(updates),
                    cg_iterations=tuple(sum(stage.cg_iterations for stage in update.stages) for update in updates),
                    cg_residuals=tuple(update.stages[-1].residual for update in updates),
                    hvp_calls=hvp_counter[0],
                    n_rejected=sum(not update.accepted for update in updates),
                    seconds=time.perf_counter() - sweep_clock,
                    seconds_hvp=hvp_time.total,
                    seconds_preconditioner=preconditioner_time.total,
                    seconds_preconditioner_apply=preconditioner_apply_time.total,
                    seconds_qr=qr_time.total + (initial_qr_seconds if sweep == 1 else 0.0),
                    seconds_accept=accept_time.total,
                    seconds_stationarity=stationarity_time.total,
                    seconds_q_unregularized=unregularized_time.total,
                    peak_memory_bytes=peak,
                    stop_reason=stop_reason,
                )
                history.append(record)
                q_previous = q_next
                if progress is not None:
                    progress(record)
                if stop_reason is not None or sweep % rule.window_sweeps == 0:
                    save(sweep)
                if stop_reason is not None:
                    break
            finally:
                _hvp_calls.reset(hvp_token)
        if stop_reason is None:
            raise RuntimeError("停止理由が決まりませんでした。")
        final_state = {
            "cores": [core.detach().to(device="cpu", dtype=torch.float64).contiguous() for core in current],
            "history": [asdict(record) for record in history],
            "stop_reason": stop_reason,
            "initial_q_lambda": initial_q,
            "initial_q_unregularized": initial_q_unregularized,
            "diagnostics": saved_diagnostics,
            "save_timings": save_timings,
            "failure": None if failure is None else asdict(failure),
            "seconds_total": seconds_before + time.perf_counter() - started,
            "n_warnings": n_warnings,
            "n_stage2": n_stage2,
            "n_stage3": n_stage3,
        }
    return result_from(final_state, resumed)
