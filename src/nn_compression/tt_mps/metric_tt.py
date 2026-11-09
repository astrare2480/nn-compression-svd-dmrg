"""入力 Gram 行列を目的にした TT-matrix の rank 選択と 1 サイト ALS。

コアの形は :math:`G^{(k)}\\in\\mathbb{R}^{r_{k-1}\\times m_k\\times n_k\\times r_k}`、
重みは :math:`W\\in\\mathbb{R}^{m\\times n}`。計算はすべて CPU の float64。

HVP は、方向 ``v`` から密な行列を作り、``G_λ`` を右から掛け、その行列を
中心コアへ縮約して戻す。局所方程式は前処理つき共役勾配法で解く。
``K_k \\otimes I`` を使ったブロック CG はここには置かない。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .canonical import tt_move_center
from .matrix import (
    dense_to_tt_matrix_cores,
    tt_cores_to_tt_matrix_cores,
    tt_matrix_cores_to_tt_cores,
    tt_matrix_num_parameters,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)
from .matrix_two_site import canonicalize_tt_matrix_cores
from .matrix_validation import validate_tt_matrix_cores
from .modes import (
    ordered_factorizations,
    tt_matrix_num_parameters_from_ranks,
    tt_matrix_shape_imbalance,
)


def _require_cpu_f64(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} は torch.Tensor である必要があります: {type(tensor)!r}")
    if tensor.dtype != torch.float64:
        raise TypeError(f"{name} は torch.float64 である必要があります: dtype={tensor.dtype}")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} は CPU 上である必要があります: device={tensor.device}")
    return tensor


def _modes(out_modes: Sequence[int], in_modes: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(out_modes, (str, bytes)) or not isinstance(out_modes, Sequence):
        raise TypeError("out_modes は int の列である必要があります。")
    if isinstance(in_modes, (str, bytes)) or not isinstance(in_modes, Sequence):
        raise TypeError("in_modes は int の列である必要があります。")
    out_tuple = tuple(int(mode) for mode in out_modes)
    in_tuple = tuple(int(mode) for mode in in_modes)
    if len(out_tuple) != len(in_tuple) or len(out_tuple) == 0:
        raise ValueError("out_modes と in_modes は空でなく、同じ長さである必要があります。")
    return out_tuple, in_tuple


def select_balanced_tt_matrix_modes(
    out_features: int,
    in_features: int,
    n_cores: int,
    *,
    min_factor: int = 2,
) -> dict[str, object]:
    """physical size の偏りが最小の mode 分解を 1 つ返す。

    候補は ``ordered_factorizations`` の全組み合わせ。評価値は
    ``tt_matrix_shape_imbalance``（physical size の max/min）。
    同点では ``(out_modes, in_modes)`` を辞書順に見て、最初のものを採る。
    ``ordered_factorizations`` 自体は辞書順なので、出力候補・入力候補の
    順に走査したときの最初の最小値と同じ結果になる。

    戻り値のキーは ``out_modes``、``in_modes``、``imbalance``。
    mode は ``tuple[int, ...]``、``imbalance`` は ``float``。
    """
    out_candidates = ordered_factorizations(out_features, n_cores, min_factor)
    in_candidates = ordered_factorizations(in_features, n_cores, min_factor)
    if not out_candidates or not in_candidates:
        raise ValueError(
            f"mode 分解がありません: out={out_features}, in={in_features}, "
            f"n_cores={n_cores}, min_factor={min_factor}"
        )
    best_key: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
    for out_modes in out_candidates:
        for in_modes in in_candidates:
            imbalance = float(tt_matrix_shape_imbalance(out_modes, in_modes))
            key = (imbalance, out_modes, in_modes)
            if best_key is None or key < best_key:
                best_key = key
    assert best_key is not None
    return {
        "out_modes": best_key[1],
        "in_modes": best_key[2],
        "imbalance": best_key[0],
    }


def structural_tt_matrix_ranks(
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> tuple[int, ...]:
    """各 bond の構造上の上限 ``(r_0, ..., r_d)`` を返す。

    境界は 1。内部 bond ``r_k`` は、左右の physical size の積の小さい方。
    physical size は ``m_j * n_j``。
    """
    out_tuple, in_tuple = _modes(out_modes, in_modes)
    physical = [out_mode * in_mode for out_mode, in_mode in zip(out_tuple, in_tuple)]
    total = math.prod(physical)
    ranks = [1]
    left = 1
    for site, size in enumerate(physical[:-1]):
        left *= size
        right = total // left
        ranks.append(min(left, right))
        del site
    ranks.append(1)
    return tuple(ranks)


def uniform_tt_matrix_ranks(
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    rank_cap: int,
) -> tuple[int, ...]:
    """一様な上限 ``rank_cap`` を、構造上の上限で頭打ちにした rank 列を返す。

    境界 ``r_0 = r_d = 1`` はそのまま。
    """
    if isinstance(rank_cap, bool) or not isinstance(rank_cap, int) or rank_cap < 1:
        raise ValueError(f"rank_cap は 1 以上の int である必要があります: {rank_cap!r}")
    structural = structural_tt_matrix_ranks(out_modes, in_modes)
    capped = [1]
    for bond in structural[1:-1]:
        capped.append(min(rank_cap, bond))
    capped.append(1)
    return tuple(capped)


def select_uniform_tt_rank(
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    budget: int,
    *,
    extra_parameters: int = 0,
) -> dict[str, object]:
    """パラメータ数が予算以下になる最大の一様 rank 上限を返す。

    数えるパラメータは TT コアの要素数に ``extra_parameters`` を足したもの。
    M4 の対角スケールは、この引数に入力次元（1024 など）を渡す。
    戻り値は ``rank_cap``、``ranks``、``tt_parameters``、``total_parameters``。
    rank 1 でも予算を超えるときは ``ValueError``。
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f"budget は 1 以上の int である必要があります: {budget!r}")
    if isinstance(extra_parameters, bool) or not isinstance(extra_parameters, int) or extra_parameters < 0:
        raise ValueError(f"extra_parameters は 0 以上の int である必要があります: {extra_parameters!r}")
    structural = structural_tt_matrix_ranks(out_modes, in_modes)
    max_cap = max(structural[1:-1], default=1)
    best: dict[str, object] | None = None
    for cap in range(1, max_cap + 1):
        ranks = uniform_tt_matrix_ranks(out_modes, in_modes, cap)
        tt_parameters = tt_matrix_num_parameters_from_ranks(out_modes, in_modes, ranks)
        total = tt_parameters + extra_parameters
        if total <= budget:
            best = {
                "rank_cap": cap,
                "ranks": ranks,
                "tt_parameters": tt_parameters,
                "extra_parameters": extra_parameters,
                "total_parameters": total,
            }
        else:
            break
    if best is None:
        raise ValueError(
            f"予算 {budget}（追加 {extra_parameters}）では rank 1 の TT も入りません。"
        )
    return best


def expand_tt_matrix_bonds(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    target_ranks: Sequence[int],
) -> list[torch.Tensor]:
    """bond 次元をゼロ埋めで ``target_ranks`` まで広げる。

    既存の成分は各 bond の先頭に置く。増えた成分は 0 なので、密な ``W`` は
    （丸めを除いて）変わらない。``target_ranks`` は境界を含む長さ ``d+1`` で、
    現在の rank 以上、境界は 1。

    戻り値は新しいコア列。dtype / device は入力と同じ。入力は変更しない。
    """
    out_tuple, in_tuple = _modes(out_modes, in_modes)
    current = tt_matrix_ranks(cores)
    if len(target_ranks) != len(current):
        raise ValueError(
            f"target_ranks の長さ {len(target_ranks)} は {len(current)} である必要があります。"
        )
    target = tuple(int(rank) for rank in target_ranks)
    if target[0] != 1 or target[-1] != 1:
        raise ValueError(f"境界 rank は 1 である必要があります: {target}")
    if any(new < old for new, old in zip(target, current)):
        raise ValueError(f"target_ranks {target} は現在の rank {current} 以上である必要があります。")
    expanded: list[torch.Tensor] = []
    for site, core in enumerate(cores):
        _require_cpu_f64(core, f"cores[{site}]")
        left = target[site]
        right = target[site + 1]
        if core.shape[0] == left and core.shape[-1] == right:
            expanded.append(core.clone())
            continue
        padded = core.new_zeros(left, core.shape[1], core.shape[2], right)
        padded[: core.shape[0], :, :, : core.shape[-1]] = core
        expanded.append(padded)
    validate_tt_matrix_cores(expanded, out_tuple, in_tuple)
    return expanded


def _contract_left_group(cores: Sequence[torch.Tensor], end: int) -> torch.Tensor:
    """``cores[:end]`` を ``(r_end, m_left, n_left)`` に縮約する。

    ``end=0`` のときは ``[[[1]]]``。入力 mode は最後の site が最速になる
    C-order でまとめる。
    """
    if end == 0:
        return cores[0].new_ones((1, 1, 1))
    accumulator = cores[0][0]
    m_so_far = int(cores[0].shape[1])
    n_so_far = int(cores[0].shape[2])
    for core in cores[1:end]:
        accumulator = torch.tensordot(accumulator, core, dims=([-1], [0]))
        m_site = int(core.shape[1])
        n_site = int(core.shape[2])
        r_next = int(core.shape[3])
        accumulator = accumulator.reshape(m_so_far, n_so_far, m_site, n_site, r_next)
        accumulator = accumulator.permute(0, 2, 1, 3, 4).reshape(
            m_so_far * m_site, n_so_far * n_site, r_next
        )
        m_so_far *= m_site
        n_so_far *= n_site
    return accumulator.permute(2, 0, 1).contiguous()


def _contract_right_group(cores: Sequence[torch.Tensor], start: int) -> torch.Tensor:
    """``cores[start:]`` を ``(r_start, m_right, n_right)`` に縮約する。

    ``start=len(cores)`` のときは ``[[[1]]]``。
    """
    if start == len(cores):
        return cores[0].new_ones((1, 1, 1))
    accumulator = cores[start]
    m_so_far = int(accumulator.shape[1])
    n_so_far = int(accumulator.shape[2])
    for core in cores[start + 1 :]:
        accumulator = torch.tensordot(accumulator, core, dims=([-1], [0]))
        m_site = int(core.shape[1])
        n_site = int(core.shape[2])
        r_next = int(core.shape[3])
        accumulator = accumulator.reshape(
            accumulator.shape[0], m_so_far, n_so_far, m_site, n_site, r_next
        )
        accumulator = accumulator.permute(0, 1, 3, 2, 4, 5).reshape(
            accumulator.shape[0], m_so_far * m_site, n_so_far * n_site, r_next
        )
        m_so_far *= m_site
        n_so_far *= n_site
    return accumulator[..., 0].contiguous()


def tt_matrix_core_adjoint(
    cores: Sequence[torch.Tensor],
    site: int,
    matrix: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    """密な行列を、site ``site`` のコア空間へ縮約して戻す。

    これは、他のコアを固定したときの線形写像 ``コア → W`` の随伴。
    ``matrix`` は shape ``(m, n)`` の float64 CPU Tensor。
    戻り値は ``cores[site]`` と同じ shape。
    """
    out_tuple, in_tuple = _modes(out_modes, in_modes)
    dense = _require_cpu_f64(matrix, "matrix")
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


def _dense_with_site(
    cores: Sequence[torch.Tensor],
    site: int,
    replacement: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    replaced = list(cores)
    replaced[site] = replacement
    return tt_matrix_to_dense(replaced, out_modes, in_modes)


def metric_tt_hvp(
    cores: Sequence[torch.Tensor],
    site: int,
    direction: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    """中心コア方向 ``v`` に局所ヘッセ行列を掛ける。

    ``H v = P^\\top \\mathrm{vec}((P v) G)``。``P v`` は site を ``v`` に
    置き換えた密行列。``gram`` は ``G`` または ``G_λ`` で、shape ``(n, n)``。
    戻り値は ``direction`` と同じ shape の float64 CPU Tensor。
    """
    vector = _require_cpu_f64(direction, "direction")
    metric = _require_cpu_f64(gram, "gram")
    if vector.shape != cores[site].shape:
        raise ValueError(
            f"direction の shape {tuple(vector.shape)} が cores[{site}] の {tuple(cores[site].shape)} と一致しません。"
        )
    dense = _dense_with_site(cores, site, vector, out_modes, in_modes)
    return tt_matrix_core_adjoint(cores, site, dense @ metric, out_modes, in_modes)


def metric_tt_rhs(
    cores: Sequence[torch.Tensor],
    site: int,
    weight: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    """局所正規方程式の右辺 ``b = P^\\top \\mathrm{vec}(W G)`` を返す。"""
    target = _require_cpu_f64(weight, "weight")
    metric = _require_cpu_f64(gram, "gram")
    return tt_matrix_core_adjoint(cores, site, target @ metric, out_modes, in_modes)


def metric_tt_jacobi_diagonal(
    cores: Sequence[torch.Tensor],
    site: int,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    """局所ヘッセ行列の対角成分を、コアと同じ shape で返す。

    対角は出力 mode 方向に繰り返す（``K \\otimes I_{m_k}`` の対角）。
    この繰り返しをブロック CG には使わない。対角そのものは前処理に使う。
    """
    out_tuple, in_tuple = _modes(out_modes, in_modes)
    metric = _require_cpu_f64(gram, "gram")
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
    m_site = out_tuple[site]
    return reduced.unsqueeze(1).expand(-1, m_site, -1, -1).contiguous()


def q_lambda_of_cores(
    cores: Sequence[torch.Tensor],
    weight: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    """コアが表す行列の :math:`Q=\\mathrm{tr}(\\Delta W\\, G\\, \\Delta W^\\top)` を返す。"""
    approximated = tt_matrix_to_dense(cores, out_modes, in_modes)
    delta = _require_cpu_f64(weight, "weight") - approximated
    metric = _require_cpu_f64(gram, "gram")
    metric = (metric + metric.transpose(0, 1)) * 0.5
    return torch.sum(delta * (delta @ metric))


@dataclass(frozen=True)
class PCGResult:
    """前処理つき共役勾配法の結果。

    solution:
        コアと同じ shape の float64 CPU Tensor。
    iterations:
        実行した反復回数。初期残差で止まったときは 0。
    final_relative_residual:
        ``||r||_2 / ||b||_2``。``b=0`` のときは 0。
    converged:
        相対残差が許容値以下なら True。
    residual_history:
        各反復の後の相対残差。初期残差は含めない。
    solution_history:
        ``record_solutions=True`` のとき、初期値と各反復後の解。
    """

    solution: torch.Tensor
    iterations: int
    final_relative_residual: float
    converged: bool
    residual_history: tuple[float, ...]
    solution_history: tuple[torch.Tensor, ...] | None = None


def preconditioned_conjugate_gradient(
    apply_hessian: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    initial: torch.Tensor,
    diagonal: torch.Tensor,
    *,
    relative_tolerance: float = 1e-8,
    max_iterations: int = 500,
    record_solutions: bool = False,
) -> PCGResult:
    """行列を作らない Jacobi 前処理つき共役勾配法。

    ``initial`` から始める。停止は相対残差 ``1e-8`` または ``max_iterations``。
    対角が非正の成分は、対角の最大値に float64 の epsilon を掛けた値で下駄を履かせる。
    これは零割を避けるためで、正定値の対角はそのまま使う。
    """
    b_vector = _require_cpu_f64(rhs, "rhs")
    x_vector = _require_cpu_f64(initial, "initial").clone()
    diag = _require_cpu_f64(diagonal, "diagonal")
    if x_vector.shape != b_vector.shape or diag.shape != b_vector.shape:
        raise ValueError("rhs, initial, diagonal の shape は一致する必要があります。")
    if relative_tolerance <= 0 or max_iterations < 1:
        raise ValueError("relative_tolerance は正、max_iterations は 1 以上である必要があります。")
    residual = b_vector - apply_hessian(x_vector)
    b_norm = torch.linalg.vector_norm(b_vector.reshape(-1))
    floor = torch.clamp(diag.max() * torch.finfo(torch.float64).eps, min=torch.finfo(torch.float64).tiny)
    safe_diag = torch.where(diag > 0, diag, floor)
    solutions: list[torch.Tensor] | None = [x_vector.clone()] if record_solutions else None
    if float(b_norm.item()) == 0.0:
        return PCGResult(x_vector, 0, 0.0, True, (), tuple(solutions) if solutions is not None else None)
    relative = float((torch.linalg.vector_norm(residual.reshape(-1)) / b_norm).item())
    if relative <= relative_tolerance:
        return PCGResult(
            x_vector, 0, relative, True, (), tuple(solutions) if solutions is not None else None
        )
    preconditioned = residual / safe_diag
    direction = preconditioned.clone()
    residual_dot = torch.dot(residual.reshape(-1), preconditioned.reshape(-1))
    history: list[float] = []
    used = 0
    converged = False
    for iteration in range(1, max_iterations + 1):
        hessian_direction = apply_hessian(direction)
        curvature = torch.dot(direction.reshape(-1), hessian_direction.reshape(-1))
        if float(curvature.item()) <= 0.0:
            break
        alpha = residual_dot / curvature
        x_vector = x_vector + alpha * direction
        residual = residual - alpha * hessian_direction
        relative = float((torch.linalg.vector_norm(residual.reshape(-1)) / b_norm).item())
        history.append(relative)
        used = iteration
        if solutions is not None:
            solutions.append(x_vector.clone())
        if relative <= relative_tolerance:
            converged = True
            break
        preconditioned = residual / safe_diag
        new_dot = torch.dot(residual.reshape(-1), preconditioned.reshape(-1))
        beta = new_dot / residual_dot
        direction = preconditioned + beta * direction
        residual_dot = new_dot
    return PCGResult(
        x_vector,
        used,
        relative,
        converged,
        tuple(history),
        tuple(solutions) if solutions is not None else None,
    )


def move_tt_matrix_center(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    source: int,
    target: int,
) -> list[torch.Tensor]:
    """混合正準形の中心を、3 階 TT の ``tt_move_center`` で隣へ移す。

    ``source`` が現在の中心であることは呼び出し側が保証する。密な ``W`` は
    QR の丸めを除いて変わらない。
    """
    tt_cores = tt_matrix_cores_to_tt_cores(cores, out_modes, in_modes)
    moved = tt_move_center(tt_cores, source, target)
    return tt_cores_to_tt_matrix_cores(moved, out_modes, in_modes)


@dataclass(frozen=True)
class SiteUpdate:
    """1 サイト更新の記録。"""

    accepted: bool
    iterations: int
    final_relative_residual: float
    converged: bool
    q_before: float
    q_after: float
    jacobi_seconds: float
    cg_seconds: float
    accept_seconds: float


def update_tt_matrix_site(
    cores: Sequence[torch.Tensor],
    site: int,
    weight: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    relative_tolerance: float = 1e-8,
    max_iterations: int = 500,
    center: int | None = None,
) -> tuple[list[torch.Tensor], SiteUpdate]:
    """正準中心 ``site`` のコアを、PCG で 1 回更新する。

    更新後の :math:`Q_\\lambda` が減っていれば受理し、増えていれば元のコアに戻す。
    ``center`` を渡したときは、その中心の正準形であることを前提にする
    （この関数は中心を動かさない）。
    """
    del center
    started = time.perf_counter()
    diagonal = metric_tt_jacobi_diagonal(cores, site, gram, out_modes, in_modes)
    jacobi_seconds = time.perf_counter() - started
    rhs = metric_tt_rhs(cores, site, weight, gram, out_modes, in_modes)

    def apply_hessian(direction: torch.Tensor) -> torch.Tensor:
        return metric_tt_hvp(cores, site, direction, gram, out_modes, in_modes)

    cg_started = time.perf_counter()
    solved = preconditioned_conjugate_gradient(
        apply_hessian,
        rhs,
        cores[site],
        diagonal,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    cg_seconds = time.perf_counter() - cg_started
    accept_started = time.perf_counter()
    q_before = q_lambda_of_cores(cores, weight, gram, out_modes, in_modes)
    candidate = list(cores)
    candidate[site] = solved.solution
    q_after = q_lambda_of_cores(candidate, weight, gram, out_modes, in_modes)
    accepted = float(q_after.item()) <= float(q_before.item())
    accept_seconds = time.perf_counter() - accept_started
    info = SiteUpdate(
        accepted=accepted,
        iterations=solved.iterations,
        final_relative_residual=solved.final_relative_residual,
        converged=solved.converged,
        q_before=float(q_before.item()),
        q_after=float(q_after.item()) if accepted else float(q_before.item()),
        jacobi_seconds=jacobi_seconds,
        cg_seconds=cg_seconds,
        accept_seconds=accept_seconds,
    )
    if not accepted:
        return list(cores), info
    return candidate, info


@dataclass(frozen=True)
class SweepRecord:
    """1 掃引の履歴。

    1 掃引は、左から右へ全サイトを更新し、その後右から左へ端以外を更新する
    往復（サイト更新は ``2d-1`` 回）。``q_lambda`` は掃引終了時の値。
    """

    sweep: int
    q_lambda: float
    relative_decrease: float
    cg_iterations: tuple[int, ...]
    cg_residuals: tuple[float, ...]
    n_rejected: int
    seconds: float


def run_metric_tt_als(
    cores: Sequence[torch.Tensor],
    weight: torch.Tensor,
    gram: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    relative_tolerance: float = 1e-8,
    max_cg_iterations: int = 500,
    sweep_tolerance: float = 1e-6,
    max_sweeps: int = 50,
) -> tuple[list[torch.Tensor], tuple[SweepRecord, ...]]:
    """正準形を保つ 1 サイト ALS。

    停止は、掃引ごとの :math:`Q_\\lambda` の相対減少が ``1e-6`` 未満、
    または 50 掃引。検証データは見ない。入力コアは変更しない。
    """
    current = canonicalize_tt_matrix_cores(cores, out_modes, in_modes, 0)
    q_previous = float(q_lambda_of_cores(current, weight, gram, out_modes, in_modes).item())
    n_sites = len(current)
    history: list[SweepRecord] = []
    for sweep in range(1, max_sweeps + 1):
        started = time.perf_counter()
        iterations: list[int] = []
        residuals: list[float] = []
        rejected = 0
        center = 0
        for site in range(n_sites):
            current, info = update_tt_matrix_site(
                current,
                site,
                weight,
                gram,
                out_modes,
                in_modes,
                relative_tolerance=relative_tolerance,
                max_iterations=max_cg_iterations,
                center=center,
            )
            iterations.append(info.iterations)
            residuals.append(info.final_relative_residual)
            rejected += int(not info.accepted)
            if site < n_sites - 1:
                current = move_tt_matrix_center(current, out_modes, in_modes, site, site + 1)
                center = site + 1
        for site in range(n_sites - 2, -1, -1):
            current = move_tt_matrix_center(current, out_modes, in_modes, site + 1, site)
            center = site
            current, info = update_tt_matrix_site(
                current,
                site,
                weight,
                gram,
                out_modes,
                in_modes,
                relative_tolerance=relative_tolerance,
                max_iterations=max_cg_iterations,
                center=center,
            )
            iterations.append(info.iterations)
            residuals.append(info.final_relative_residual)
            rejected += int(not info.accepted)
        q_next = float(q_lambda_of_cores(current, weight, gram, out_modes, in_modes).item())
        relative_decrease = 0.0 if q_previous == 0.0 else (q_previous - q_next) / q_previous
        history.append(
            SweepRecord(
                sweep=sweep,
                q_lambda=q_next,
                relative_decrease=relative_decrease,
                cg_iterations=tuple(iterations),
                cg_residuals=tuple(residuals),
                n_rejected=rejected,
                seconds=time.perf_counter() - started,
            )
        )
        if relative_decrease < sweep_tolerance:
            break
        q_previous = q_next
    return current, tuple(history)


def diagonal_rms_scales(gram: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """M4 の対角スケール :math:`s_i=\\sqrt{G_{ii}/N}` を返す。

    戻り値は shape ``(n,)`` の float64 CPU Tensor。0 以下の成分があれば
    ``ValueError``（逆数を作れないため、値を置き換えない）。
    """
    metric = _require_cpu_f64(gram, "gram")
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 1:
        raise ValueError(f"n_tokens は 1 以上の int である必要があります: {n_tokens!r}")
    scales = torch.sqrt(torch.diagonal(metric).clamp(min=0) / n_tokens)
    if torch.any(scales <= 0):
        raise ValueError("対角 RMS に 0 以下のチャネルがあります。クランプはしません。")
    return scales


def scale_matrix_columns(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``W @ diag(scales)`` を返す。``scales`` は shape ``(n,)``。"""
    matrix = _require_cpu_f64(weight, "weight")
    diagonal = _require_cpu_f64(scales, "scales")
    if diagonal.ndim != 1 or diagonal.shape[0] != matrix.shape[1]:
        raise ValueError(f"scales の shape {tuple(diagonal.shape)} が入力次元 {matrix.shape[1]} と一致しません。")
    return matrix * diagonal.unsqueeze(0)


@dataclass(frozen=True)
class DiagonalMetricTT:
    """M4 の 2 つの表現。

    scaled_cores:
        ``W @ diag(s)`` を TT-SVD したコア。推論ではこれと ``input_scale_inv``
        を組にして、入力に ``D^{-1}`` を掛けてから TT を適用する。
    input_scale_inv:
        ``1/s``。shape ``(n,)`` の float64 CPU Tensor。
    absorbed_cores:
        M4 の密行列を TT-SVD で作り直したコア。一般の対角は bond 次元を
        保ったままコアへ入らないので、``dense(scaled_cores) @ diag(1/s)`` を
        同じ ``max_rank`` で作り直す。属性名は保存済みのファイルに合わせてある。
        ``max_rank=None`` なら切り詰めなし。
    scales:
        ``s``。shape ``(n,)``。
    """

    scaled_cores: list[torch.Tensor]
    input_scale_inv: torch.Tensor
    absorbed_cores: list[torch.Tensor]
    scales: torch.Tensor
    max_rank: int | None


def fit_diagonal_metric_tt(
    weight: torch.Tensor,
    gram: torch.Tensor,
    n_tokens: int,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    max_rank: int | None,
) -> DiagonalMetricTT:
    """対角 RMS で列をスケールしてから TT-SVD する（M4）。"""
    scales = diagonal_rms_scales(gram, n_tokens)
    scaled = scale_matrix_columns(weight, scales)
    scaled_cores = dense_to_tt_matrix_cores(scaled, out_modes, in_modes, max_rank=max_rank)
    inverse = 1.0 / scales
    deployed = scale_matrix_columns(
        tt_matrix_to_dense(scaled_cores, out_modes, in_modes),
        inverse,
    )
    absorbed = dense_to_tt_matrix_cores(deployed, out_modes, in_modes, max_rank=max_rank)
    return DiagonalMetricTT(
        scaled_cores=scaled_cores,
        input_scale_inv=inverse,
        absorbed_cores=absorbed,
        scales=scales,
        max_rank=max_rank,
    )


def prepare_m5_from_diagonal_tt(
    diagonal_tt: DiagonalMetricTT,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    target_ranks: Sequence[int],
) -> list[torch.Tensor]:
    """M4 の密行列を TT-SVD で作り直したコアを、M5 の初期値にする。

    作り直したコアの bond を M3 と同じ rank までゼロで広げ、site 0 を中心とする
    正準形にして返す。対角スケールそのものをコアへ吸収する操作ではない。
    """
    expanded = expand_tt_matrix_bonds(
        diagonal_tt.absorbed_cores,
        out_modes,
        in_modes,
        target_ranks,
    )
    return canonicalize_tt_matrix_cores(expanded, out_modes, in_modes, 0)


def tt_parameter_count(cores: Sequence[torch.Tensor]) -> int:
    """コアの要素数。``tt_matrix_num_parameters`` と同じ。"""
    return tt_matrix_num_parameters(cores)
