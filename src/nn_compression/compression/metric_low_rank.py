"""入力 Gram 行列を使った、行列の低ランク近似。

最適化と診断は CPU の float64 だけで行う。活性化は行がトークン、
最後の次元が入力特徴 ``n`` である（``x @ W.T`` の ``x`` と同じ並び）。
トークン ``t`` の特徴ベクトルを :math:`x_t` とすると

.. math::

    G = \\sum_t x_t x_t^\\top

で、平均は引かない。``G`` は対称化する（:math:`(G+G^\\top)/2`）。
これは float64 の丸めで崩れた対称性を戻すだけで、式は変えない。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _require_cpu_f64(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """float64 かつ CPU の Tensor だけを受け取る。"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} は torch.Tensor である必要があります: {type(tensor)!r}")
    if tensor.dtype != torch.float64:
        raise TypeError(f"{name} は torch.float64 である必要があります: dtype={tensor.dtype}")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} は CPU 上である必要があります: device={tensor.device}")
    return tensor


def _symmetrize(gram: torch.Tensor) -> torch.Tensor:
    return (gram + gram.transpose(-1, -2)) * 0.5


def _as_token_feature_matrix(activations: torch.Tensor, name: str) -> torch.Tensor:
    """``(..., n)`` を ``(T, n)`` の float64 CPU 行列にする。"""
    matrix = _require_cpu_f64(activations, name)
    if matrix.ndim < 2:
        raise ValueError(f"{name} は (トークン, 特徴) の 2 次元以上である必要があります: ndim={matrix.ndim}")
    if matrix.shape[-1] < 1:
        raise ValueError(f"{name} の特徴次元が空です: shape={tuple(matrix.shape)}")
    return matrix.reshape(-1, matrix.shape[-1])


def gram_matrix(activations: torch.Tensor) -> torch.Tensor:
    """活性化から Gram 行列 :math:`G=\\sum_t x_t x_t^\\top` を返す。

    activations:
        最後の次元が入力特徴 ``n``。先頭の次元はトークン軸。
        dtype は ``torch.float64``、device は CPU。
        平均は引かない。

    戻り値:
        shape ``(n, n)`` の float64 CPU Tensor。対称。
    """
    features = _as_token_feature_matrix(activations, "activations")
    gram = features.transpose(0, 1) @ features
    return _symmetrize(gram)


def gram_matrices_from_documents(
    documents: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """文書ごとの Gram 行列と、その合計を返す。

    documents:
        各要素は 1 文書の活性化。最後の次元 ``n`` は全文書で一致する。
        dtype は ``torch.float64``、device は CPU。

    戻り値:
        ``(per_document, total)``。
        ``per_document`` は shape ``(D, n, n)``、``total`` は shape ``(n, n)``。
        どちらも float64 CPU で対称。``D=0`` のときは ``total`` を作れないので
        ``ValueError``。
    """
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise TypeError("documents は活性化 Tensor の列である必要があります。")
    if len(documents) == 0:
        raise ValueError("documents が空です。")
    grams = [gram_matrix(document) for document in documents]
    n_features = grams[0].shape[0]
    for index, gram in enumerate(grams):
        if gram.shape != (n_features, n_features):
            raise ValueError(
                f"documents[{index}] の特徴数 {gram.shape[0]} が"
                f" documents[0] の {n_features} と一致しません。"
            )
    stacked = torch.stack(grams, dim=0)
    return stacked, _symmetrize(stacked.sum(dim=0))


def regularize_gram(gram: torch.Tensor, lam: float = 1e-6) -> torch.Tensor:
    """:math:`G_\\lambda = G + \\lambda \\cdot \\mathrm{tr}(G)/n \\cdot I` を返す。

    gram:
        shape ``(n, n)`` の float64 CPU Tensor。
    lam:
        正則化係数。この実験では ``1e-6`` を変えない。

    戻り値:
        shape ``(n, n)`` の float64 CPU Tensor。入力は変更しない。
    """
    matrix = _symmetrize(_require_cpu_f64(gram, "gram"))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"gram は正方行列である必要があります: shape={tuple(matrix.shape)}")
    if isinstance(lam, bool) or not isinstance(lam, (int, float)):
        raise TypeError(f"lam は実数である必要があります: {lam!r}")
    if not math.isfinite(float(lam)) or float(lam) < 0:
        raise ValueError(f"lam は 0 以上の有限値である必要があります: {lam}")
    n_features = matrix.shape[0]
    shift = float(lam) * torch.trace(matrix) / n_features
    eye = torch.eye(n_features, dtype=torch.float64)
    return matrix + shift * eye


def regularization_shift(gram: torch.Tensor, lam: float = 1e-6) -> float:
    """正則化で足す対角成分 :math:`\\lambda \\cdot \\mathrm{tr}(G)/n` を返す。"""
    matrix = _symmetrize(_require_cpu_f64(gram, "gram"))
    n_features = matrix.shape[0]
    return float(lam) * float(torch.trace(matrix).item()) / n_features


def _eigenvalues(gram: torch.Tensor) -> torch.Tensor:
    matrix = _symmetrize(_require_cpu_f64(gram, "gram"))
    return torch.linalg.eigvalsh(matrix)


def effective_rank_from_eigenvalues(eigenvalues: torch.Tensor) -> float:
    """Roy & Vetterli の effective rank :math:`\\exp(H(p))` を返す。

    :math:`p_i = \\lambda_i^+ / \\sum_j \\lambda_j^+`。負の固有値は 0 にする。
    正の固有値が無いときは 0。
    """
    positive = eigenvalues.clamp(min=0)
    total = positive.sum()
    if float(total.item()) == 0.0:
        return 0.0
    probabilities = positive / total
    probabilities = probabilities[probabilities > 0]
    entropy = -(probabilities * probabilities.log()).sum()
    return float(torch.exp(entropy).item())


def channel_rms(gram: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """チャネルごとの RMS :math:`\\sqrt{G_{ii}/N}` を返す。

    戻り値は shape ``(n,)`` の float64 CPU Tensor。``N`` はトークン数。
    """
    matrix = _symmetrize(_require_cpu_f64(gram, "gram"))
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, int):
        raise TypeError(f"n_tokens は int である必要があります: {n_tokens!r}")
    if n_tokens < 1:
        raise ValueError(f"n_tokens は 1 以上である必要があります: {n_tokens}")
    diagonal = torch.diagonal(matrix).clamp(min=0)
    return torch.sqrt(diagonal / n_tokens)


def gram_diagnostics(
    gram: torch.Tensor,
    *,
    lam: float = 1e-6,
    n_tokens: int,
    quantiles: Sequence[float] = (0.5, 0.9, 0.99),
) -> dict[str, float]:
    """``G`` と ``G_λ`` の診断量を、CSV に書ける float の辞書で返す。

    キーの接頭辞 ``G_`` は正則化前、``G_lambda_`` は正則化後。
    RMS は正則化前の対角から計算する。分位点は ``rms_q50`` のように百分率で付ける。
    """
    matrix = _symmetrize(_require_cpu_f64(gram, "gram"))
    regularized = regularize_gram(matrix, lam)
    eigenvalues = _eigenvalues(matrix)
    eigenvalues_reg = _eigenvalues(regularized)
    lam_min = float(eigenvalues[0].item())
    lam_max = float(eigenvalues[-1].item())
    reg_min = float(eigenvalues_reg[0].item())
    reg_max = float(eigenvalues_reg[-1].item())
    shift = regularization_shift(matrix, lam)
    rms = channel_rms(matrix, n_tokens)
    diagnostics: dict[str, float] = {
        "n_features": float(matrix.shape[0]),
        "n_tokens": float(n_tokens),
        "lambda": float(lam),
        "regularization_shift": shift,
        "G_eigenvalue_min": lam_min,
        "G_eigenvalue_max": lam_max,
        "G_condition_number": math.inf if lam_min <= 0 else lam_max / lam_min,
        "G_effective_rank": effective_rank_from_eigenvalues(eigenvalues),
        "G_lambda_eigenvalue_min": reg_min,
        "G_lambda_eigenvalue_max": reg_max,
        "G_lambda_condition_number": math.inf if reg_min <= 0 else reg_max / reg_min,
        "G_lambda_effective_rank": effective_rank_from_eigenvalues(eigenvalues_reg),
        "regularization_to_min_eigenvalue": math.inf if lam_min == 0 else shift / lam_min,
        "rms_max": float(rms.max().item()),
        "rms_median": float(rms.median().item()),
    }
    for quantile in quantiles:
        if not 0.0 <= float(quantile) <= 1.0:
            raise ValueError(f"quantile は 0 から 1 です: {quantile}")
        percent = int(round(float(quantile) * 100))
        diagnostics[f"rms_q{percent}"] = float(torch.quantile(rms, float(quantile)).item())
    return diagnostics


def squared_output_error_from_gram(delta_weight: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    """:math:`Q=\\mathrm{tr}(\\Delta W\\, G\\, \\Delta W^\\top)` を 0 次元 Tensor で返す。

    delta_weight:
        shape ``(m, n)`` の float64 CPU Tensor。
    gram:
        shape ``(n, n)`` の float64 CPU Tensor。
    """
    delta = _require_cpu_f64(delta_weight, "delta_weight")
    matrix = _symmetrize(_require_cpu_f64(gram, "gram"))
    if delta.ndim != 2:
        raise ValueError(f"delta_weight は 2 次元である必要があります: ndim={delta.ndim}")
    if matrix.shape != (delta.shape[1], delta.shape[1]):
        raise ValueError(
            f"gram の shape {tuple(matrix.shape)} は delta_weight の入力次元"
            f" {delta.shape[1]} と一致する必要があります。"
        )
    return torch.sum(delta * (delta @ matrix))


def squared_output_errors_from_documents(
    delta_weight: torch.Tensor,
    document_grams: torch.Tensor,
) -> torch.Tensor:
    """文書ごとの :math:`Q_d=\\mathrm{tr}(\\Delta W\\, G_d\\, \\Delta W^\\top)` を返す。

    document_grams:
        shape ``(D, n, n)``。戻り値は shape ``(D,)`` の float64 CPU Tensor。
    """
    delta = _require_cpu_f64(delta_weight, "delta_weight")
    grams = _symmetrize(_require_cpu_f64(document_grams, "document_grams"))
    if grams.ndim != 3 or grams.shape[-1] != grams.shape[-2]:
        raise ValueError(f"document_grams は (D, n, n) である必要があります: shape={tuple(grams.shape)}")
    if delta.shape[1] != grams.shape[-1]:
        raise ValueError(
            f"delta_weight の入力次元 {delta.shape[1]} と document_grams の {grams.shape[-1]} が一致しません。"
        )
    mixed = torch.einsum("mn,dnk->dmk", delta, grams)
    return torch.einsum("dmk,mk->d", mixed, delta)


def squared_output_error_from_activations(
    delta_weight: torch.Tensor,
    activations: torch.Tensor,
) -> torch.Tensor:
    """活性化から :math:`\\|\\Delta W X\\|_F^2` を直接計算して返す。

    activations の最後の次元は入力特徴。戻り値は 0 次元の float64 CPU Tensor。
    """
    delta = _require_cpu_f64(delta_weight, "delta_weight")
    features = _as_token_feature_matrix(activations, "activations")
    if features.shape[1] != delta.shape[1]:
        raise ValueError(
            f"activations の特徴数 {features.shape[1]} と delta_weight の {delta.shape[1]} が一致しません。"
        )
    residual = features @ delta.transpose(0, 1)
    return torch.sum(residual * residual)


def max_matrix_rank_within_budget(out_features: int, in_features: int, budget: int) -> int:
    """:math:`r(m+n)\\le` 予算 を満たす最大の行列 rank を返す。

    バイアスは数えない。予算が ``m+n`` 未満なら ``ValueError``。
    """
    for name, value in (
        ("out_features", out_features),
        ("in_features", in_features),
        ("budget", budget),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} は int である必要があります: {value!r}")
        if value < 1:
            raise ValueError(f"{name} は 1 以上である必要があります: {value}")
    width = out_features + in_features
    rank = budget // width
    if rank < 1:
        raise ValueError(f"予算 {budget} では rank 1 の行列（{width} パラメータ）も入りません。")
    return min(rank, out_features, in_features)


def matrix_factor_parameter_count(out_features: int, in_features: int, rank: int) -> int:
    """バイアスなし低ランク因子 :math:`r(m+n)` のパラメータ数を返す。"""
    if rank < 1:
        raise ValueError(f"rank は 1 以上である必要があります: {rank}")
    return rank * (out_features + in_features)


def truncated_svd_weight(weight: torch.Tensor, rank: int) -> torch.Tensor:
    """通常の SVD で rank ``r`` に切った行列（M1）を返す。

    weight:
        shape ``(m, n)`` の float64 CPU Tensor。変更しない。
    戻り値:
        同じ shape・dtype・device の近似行列。
    """
    matrix = _require_cpu_f64(weight, "weight")
    if matrix.ndim != 2:
        raise ValueError(f"weight は 2 次元である必要があります: ndim={matrix.ndim}")
    max_rank = min(matrix.shape)
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(f"rank は int である必要があります: {rank!r}")
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rank は 1〜{max_rank} である必要があります: {rank}")
    u_factor, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u_factor[:, :rank] * singular_values[:rank]) @ vh[:rank, :]


@dataclass(frozen=True)
class WhitenedSvdResult:
    """M2 の結果。

    weight:
        近似行列。shape は入力と同じ ``(m, n)``、float64 CPU。
    discarded_singular_value_energy:
        捨てた白色化特異値の二乗和。0 次元 float64 Tensor。
    q_lambda:
        :math:`Q_\\lambda`。0 次元 float64 Tensor。
        数学的には ``discarded_singular_value_energy`` と一致する。
    """

    weight: torch.Tensor
    discarded_singular_value_energy: torch.Tensor
    q_lambda: torch.Tensor


def gram_sqrt_and_inv_sqrt(gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """対称行列の :math:`G^{1/2}` と :math:`G^{-1/2}` を固有値分解で返す。

    戻り値は ``(G_sqrt, G_inv_sqrt, eigenvalues)``。固有値は昇順。
    最小固有値が 0 以下なら ``ValueError``（正定値だけを平方根の対象にする）。
    """
    matrix = _symmetrize(_require_cpu_f64(gram, "gram"))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    if float(eigenvalues[0].item()) <= 0.0:
        raise ValueError(
            "平方根を作る Gram 行列は正定値である必要があります。"
            f" 最小固有値={float(eigenvalues[0].item())}"
        )
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    inv_sqrt_eigenvalues = 1.0 / sqrt_eigenvalues
    gram_sqrt = (eigenvectors * sqrt_eigenvalues.unsqueeze(0)) @ eigenvectors.transpose(0, 1)
    gram_inv_sqrt = (eigenvectors * inv_sqrt_eigenvalues.unsqueeze(0)) @ eigenvectors.transpose(0, 1)
    return gram_sqrt, gram_inv_sqrt, eigenvalues


def whitened_truncated_svd(weight: torch.Tensor, gram_lambda: torch.Tensor, rank: int) -> WhitenedSvdResult:
    """白色化した SVD（M2）で rank ``r`` の行列を返す。

    :math:`W G_\\lambda^{1/2}` を SVD で rank ``r`` に切り、
    :math:`G_\\lambda^{-1/2}` を右から掛けて戻す。
    :math:`G_\\lambda^{\\pm 1/2}` は固有値分解で作る。
    """
    matrix = _require_cpu_f64(weight, "weight")
    gram_sqrt, gram_inv_sqrt, _eigenvalues = gram_sqrt_and_inv_sqrt(gram_lambda)
    if gram_sqrt.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"G_λ の次元 {gram_sqrt.shape[0]} と weight の入力次元 {matrix.shape[1]} が一致しません。"
        )
    max_rank = min(matrix.shape)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= max_rank:
        raise ValueError(f"rank は 1〜{max_rank} の int である必要があります: {rank!r}")
    transformed = matrix @ gram_sqrt
    u_factor, singular_values, vh = torch.linalg.svd(transformed, full_matrices=False)
    kept = (u_factor[:, :rank] * singular_values[:rank]) @ vh[:rank, :]
    approximated = kept @ gram_inv_sqrt
    discarded = torch.sum(singular_values[rank:] ** 2)
    q_lambda = squared_output_error_from_gram(matrix - approximated, gram_lambda)
    return WhitenedSvdResult(
        weight=approximated,
        discarded_singular_value_energy=discarded,
        q_lambda=q_lambda,
    )


def gated_error_parts(
    weight: torch.Tensor,
    approximated: torch.Tensor,
    gate_weight: torch.Tensor,
    activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """E_gated の分子と分母を返す。

    .. math::

        \\|\\mathrm{SiLU}(W_{gate} X) \\odot (\\Delta W X)\\|_F^2,\\quad
        \\|\\mathrm{SiLU}(W_{gate} X) \\odot (W X)\\|_F^2

    ``W_gate`` は圧縮しない。``activations`` は ``(T, n)`` またはトークン軸を
    持つ float64 CPU Tensor。戻り値はどちらも 0 次元 float64 Tensor。
    """
    target = _require_cpu_f64(weight, "weight")
    approx = _require_cpu_f64(approximated, "approximated")
    gate = _require_cpu_f64(gate_weight, "gate_weight")
    features = _as_token_feature_matrix(activations, "activations")
    if approx.shape != target.shape or gate.shape != target.shape:
        raise ValueError(
            "weight, approximated, gate_weight の shape は一致する必要があります: "
            f"{tuple(target.shape)}, {tuple(approx.shape)}, {tuple(gate.shape)}"
        )
    if features.shape[1] != target.shape[1]:
        raise ValueError("activations の特徴数が weight の入力次元と一致しません。")
    gate_activation = F.silu(features @ gate.transpose(0, 1))
    delta_out = features @ (target - approx).transpose(0, 1)
    base_out = features @ target.transpose(0, 1)
    numerator = torch.sum((gate_activation * delta_out) ** 2)
    denominator = torch.sum((gate_activation * base_out) ** 2)
    return numerator, denominator


def gated_error_parts_per_document(
    weight: torch.Tensor,
    approximated: torch.Tensor,
    gate_weight: torch.Tensor,
    documents: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """文書ごとの E_gated 分子・分母を shape ``(D,)`` で返す。"""
    if len(documents) == 0:
        raise ValueError("documents が空です。")
    numerators = []
    denominators = []
    for document in documents:
        numerator, denominator = gated_error_parts(weight, approximated, gate_weight, document)
        numerators.append(numerator)
        denominators.append(denominator)
    return torch.stack(numerators), torch.stack(denominators)


def _error_ratio_record(numerator: torch.Tensor, denominator: torch.Tensor) -> dict[str, float]:
    if float(denominator.item()) == 0.0:
        raise ValueError("分母 ||W X||_F^2 が 0 です。")
    relative_square = numerator / denominator
    return {
        "numerator": float(numerator.item()),
        "denominator": float(denominator.item()),
        "e": float(relative_square.item()),
        "E_rel": float(torch.sqrt(relative_square).item()),
    }


def output_error_ratios_from_gram(
    weight: torch.Tensor,
    approximated: torch.Tensor,
    gram: torch.Tensor,
) -> dict[str, float]:
    """``e`` と ``E_rel`` を Gram 行列から計算する。

    ``e = tr(ΔW G ΔW^T) / tr(W G W^T)``、``E_rel = sqrt(e)``。
    ``gram`` は ``(n, n)``。活性化が同じ形でも、この関数は Gram として扱う。
    """
    target = _require_cpu_f64(weight, "weight")
    approx = _require_cpu_f64(approximated, "approximated")
    if approx.shape != target.shape:
        raise ValueError("approximated の shape が weight と一致しません。")
    numerator = squared_output_error_from_gram(target - approx, gram)
    denominator = squared_output_error_from_gram(target, gram)
    return _error_ratio_record(numerator, denominator)


def output_error_ratios_from_activations(
    weight: torch.Tensor,
    approximated: torch.Tensor,
    activations: torch.Tensor,
) -> dict[str, float]:
    """``e`` と ``E_rel`` を活性化から直接計算する。

    ``e = ||ΔW X||_F^2 / ||W X||_F^2``。トークン数と特徴数が同じ正方でも、
    Gram 行列にはしない。
    """
    target = _require_cpu_f64(weight, "weight")
    approx = _require_cpu_f64(approximated, "approximated")
    if approx.shape != target.shape:
        raise ValueError("approximated の shape が weight と一致しません。")
    numerator = squared_output_error_from_activations(target - approx, activations)
    denominator = squared_output_error_from_activations(target, activations)
    return _error_ratio_record(numerator, denominator)
