"""重み摂動（圧縮）による出力誤差の理論上界。

参照元:
    notebooks/30_tt_mps/10_fashion_mnist_mlp/01_ttlinear_output_error_norm_bounds.ipynb
    notebooks/30_tt_mps/10_fashion_mnist_mlp/02_relu_hidden_activation_error_todo.ipynb
    notebooks/30_tt_mps/10_fashion_mnist_mlp/03_second_linear_logits_error_propagation.ipynb
    notebooks/30_tt_mps/10_fashion_mnist_mlp/05_logits_bound_to_margin_certificate.ipynb
    notebooks/30_tt_mps/10_fashion_mnist_mlp/06_minimum_margin_batchwide_stability.ipynb

上界の計算だけを担当する。実測誤差との比較（assert）、表示、DataFrame 化、
plot は呼び出し側（Notebook・実験コード）に置く。TT 固有の処理ではなく、
重み差分 ``delta_W = W_approx - W`` を持つ任意の低ランク・量子化・TT 近似に使える。

規約: 重みは ``nn.Linear`` と同じ ``(out_features, in_features)``、入力は
``(batch, in_features)``、``Y = X W^T + b``。bias は近似前後で同じなら誤差に現れない。

NaN / inf を含む入力は ``ValueError`` にする。メッセージには、どの引数が
非有限かが入る。spectral norm 内部の SVD は NaN に対して ``_LinAlgError`` を
出すことがあり、呼び出し側が backend 依存の例外を見ないように、計算の前に
拒否する。検査は ``torch.isfinite(...).all()`` の bool 評価で、CUDA では
同期する。正常な入力の dtype / device / autograd は維持し、検査のために
入力 Tensor は変更しない。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_SUPPORTED_DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True)
class LinearOutputErrorBounds:
    """Linear 出力誤差 ``||Delta_Y||_F`` の2通りの上界（どちらも 0 階 Tensor）。

    ``Delta_Y = X Delta_W^T`` に対し、

    - ``frobenius_x_spectral_dw``: ``||X||_F * ||Delta_W||_2``
    - ``spectral_x_frobenius_dw``: ``||X||_2 * ||Delta_W||_F``

    はどちらも ``||Delta_Y||_F`` の上界。どちらが小さいかは入力に依存する。
    """

    frobenius_x_spectral_dw: torch.Tensor
    spectral_x_frobenius_dw: torch.Tensor


def _check_tensor(value, *, name: str, ndim: int) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} は torch.Tensor である必要があります: {type(value)!r}")
    if value.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"{name}.dtype={value.dtype} は未対応です"
            "（torch.float32 / torch.float64 のみ）。"
        )
    if value.ndim != ndim:
        raise ValueError(
            f"{name} は{ndim}階である必要があります: "
            f"ndim={value.ndim}, shape={tuple(value.shape)}"
        )
    if value.numel() == 0:
        raise ValueError(f"{name} は空にできません: shape={tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} に NaN / inf が含まれています。")


def _check_same_dtype_and_device(reference: torch.Tensor, other: torch.Tensor, *, names) -> None:
    ref_name, other_name = names
    if other.dtype != reference.dtype:
        raise TypeError(
            f"{other_name}.dtype={other.dtype} は {ref_name}.dtype={reference.dtype} と"
            "一致する必要があります。"
        )
    if other.device != reference.device:
        raise ValueError(
            f"{other_name}.device={other.device} は {ref_name}.device={reference.device} と"
            "一致する必要があります。"
        )


def _check_x_and_delta_w(X: torch.Tensor, delta_W: torch.Tensor, *, delta_name: str) -> None:
    _check_tensor(X, name="X", ndim=2)
    _check_tensor(delta_W, name=delta_name, ndim=2)
    _check_same_dtype_and_device(X, delta_W, names=("X", delta_name))
    if X.shape[1] != delta_W.shape[1]:
        raise ValueError(
            f"X.shape[1]={X.shape[1]} は {delta_name}.shape[1]={delta_W.shape[1]}"
            "（in_features）と一致する必要があります。"
        )


def _anchored_zero(*tensors: torch.Tensor) -> torch.Tensor:
    """入力に接続された 0 階の 0。

    有限値に 0 を掛けてから足す。``finfo.max * 0`` は 0 であり、
    ``inf * 0`` にはならない。device / dtype は最初のテンソルに従う。
    """
    zero = (tensors[0] * 0).sum()
    for tensor in tensors[1:]:
        zero = zero + (tensor * 0).sum()
    return zero


def _posinf_like(like: torch.Tensor) -> torch.Tensor:
    """``like`` と同じ shape / dtype / device の ``+inf``。

    非有限要素は 0 に置き換えてから足すので、``inf * 0`` や ``NaN`` を作らない。
    """
    finite = torch.where(torch.isfinite(like), like, torch.zeros_like(like))
    return finite * 0 + torch.full_like(like, float("inf"))


def _is_nonzero(matrix: torch.Tensor) -> bool:
    return bool(torch.any(matrix))


def _upper_norm(norm: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """有限なノルムはそのまま、overflow の ``inf`` もそのまま、backend の ``NaN`` は ``+inf``。

    有限な通常範囲では ``norm`` 自体を返すので、値も autograd も直接計算と同じ。
    巨大な有限入力の SVD が CUDA で ``NaN`` を返すことがある。それは表現範囲を
    超えた上界として ``+inf`` に置き換え、呼び出し側へ ``NaN`` を出さない。
    """
    if bool(torch.isfinite(norm)):
        return norm
    if not bool(torch.isnan(norm)):
        return norm
    return _posinf_like(_anchored_zero(matrix))


def _fro(matrix: torch.Tensor) -> torch.Tensor:
    return _upper_norm(torch.linalg.vector_norm(matrix), matrix)


def _spectral(matrix: torch.Tensor) -> torch.Tensor:
    return _upper_norm(torch.linalg.matrix_norm(matrix, ord=2), matrix)


def _mul_nonnegative(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """非負な 0 階量の積。

    両方が有限なら ``left * right`` そのもの（片方が 0 でも ``0 * 有限 = 0``）。
    片方が ``inf`` で他方が正なら積は ``inf``。``inf * 0`` は、零行列の早期 return
    の後に残る underflow したノルムと overflow の組み合わせなので、上界として
    ``+inf`` を返す（``NaN`` にしない。厳密な 0 を過小評価もしない）。
    """
    if bool(torch.isfinite(left)) and bool(torch.isfinite(right)):
        return left * right
    if bool(left == 0) or bool(right == 0):
        finite_side = left if bool(torch.isfinite(left)) else right
        return _posinf_like(finite_side)
    return left * right


def linear_output_error_bounds(
    X: torch.Tensor,
    delta_W: torch.Tensor,
) -> LinearOutputErrorBounds:
    r"""重み差分 ``delta_W`` による Linear 出力誤差 ``||Delta_Y||_F`` の上界を返す。

    ``Delta_Y = X Delta_W^T`` について

    .. math::

        \|\Delta Y\|_F \le \|X\|_F \|\Delta W\|_2, \qquad
        \|\Delta Y\|_F \le \|X\|_2 \|\Delta W\|_F

    X:
        入力。shape ``(batch, in_features)``。
    delta_W:
        重み差分 ``W_approx - W``。shape ``(out_features, in_features)``。
        ``X`` と dtype / device が一致している必要がある。

    戻り値:
        ``LinearOutputErrorBounds``。各上界は 0 階 Tensor（batch 全体の
        Frobenius 誤差に対する上界であり、sample ごとの上界ではない）。
        dtype / device は入力を維持し、autograd の計算グラフは切らない。
        ``X == 0`` または ``delta_W == 0`` なら両方とも厳密に 0（入力に接続した
        0）。どちらも非零で、ノルムの積が dtype の範囲を超えるときは ``inf`` に
        なり得る。有限入力から ``NaN`` は返さない。

    例外:
        Tensor でない・dtype が float32 / float64 以外なら ``TypeError``。
        階数・shape・dtype / device の不一致、空 Tensor は
        ``ValueError`` / ``TypeError``。NaN / inf を含む引数は、どの引数かを
        示した ``ValueError``（SVD の ``_LinAlgError`` ではない）。
    """
    _check_x_and_delta_w(X, delta_W, delta_name="delta_W")
    # 厳密な零因子は、相手のノルムが inf になっても上界 0。巨大な側の SVD はしない。
    if not _is_nonzero(X) or not _is_nonzero(delta_W):
        zero = _anchored_zero(X, delta_W)
        return LinearOutputErrorBounds(zero, zero)

    return LinearOutputErrorBounds(
        frobenius_x_spectral_dw=_mul_nonnegative(_fro(X), _spectral(delta_W)),
        spectral_x_frobenius_dw=_mul_nonnegative(_spectral(X), _fro(delta_W)),
    )


def _check_mlp_inputs(
    X: torch.Tensor,
    delta_W1: torch.Tensor,
    W2: torch.Tensor,
) -> None:
    _check_x_and_delta_w(X, delta_W1, delta_name="delta_W1")
    _check_tensor(W2, name="W2", ndim=2)
    _check_same_dtype_and_device(X, W2, names=("X", "W2"))
    if W2.shape[1] != delta_W1.shape[0]:
        raise ValueError(
            f"W2.shape[1]={W2.shape[1]} は delta_W1.shape[0]={delta_W1.shape[0]}"
            "（hidden_features）と一致する必要があります。"
        )


def mlp_logits_error_bound(
    X: torch.Tensor,
    delta_W1: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    r"""``Linear -> ReLU -> Linear`` の logits 誤差の batch 全体の上界を返す。

    第1層だけが ``delta_W1`` で摂動された（第2層 ``W2`` と両 bias は同一）とき、

    .. math::

        \|\Delta L\|_F \le \|X\|_F \|\Delta W_1\|_2 \|W_2\|_2

    ``Delta Z = X Delta W_1^T``、ReLU は要素ごとに 1-Lipschitz なので
    ``||Delta H||_F <= ||Delta Z||_F``、``Delta L = Delta H W_2^T`` より従う。

    X:
        入力。shape ``(batch, in_features)``。
    delta_W1:
        第1層の重み差分。shape ``(hidden_features, in_features)``。
    W2:
        摂動していない第2層の重み。shape ``(num_classes, hidden_features)``。

    戻り値:
        0 階 Tensor（batch 全体の ``||Delta L||_F`` に対する上界）。
        sample ごとの上界ではない（sample ごとなら
        ``mlp_sample_logits_error_bounds``）。dtype / device / autograd を維持する。
        ``X``、``delta_W1``、``W2`` のいずれかが零行列なら厳密に 0。
        どれも非零で積が表現範囲を超えるときは ``inf`` になり得る。``NaN`` は返さない。

    例外:
        ``linear_output_error_bounds`` と同じ。NaN / inf は ``ValueError``。
    """
    _check_mlp_inputs(X, delta_W1, W2)
    if not _is_nonzero(X) or not _is_nonzero(delta_W1) or not _is_nonzero(W2):
        return _anchored_zero(X, delta_W1, W2)
    return _mul_nonnegative(_mul_nonnegative(_fro(X), _spectral(delta_W1)), _spectral(W2))


def mlp_sample_logits_error_bounds(
    X: torch.Tensor,
    delta_W1: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    r"""``Linear -> ReLU -> Linear`` の logits 誤差の sample ごとの上界を返す。

    .. math::

        U_n = \|x_n\|_2 \|\Delta W_1\|_2 \|W_2\|_2 \;\ge\; \|\Delta l_n\|_2

    sample ``n`` の誤差ベクトル ``Delta l_n`` について成り立つ（上の batch 上界と
    同じ導出を1行ずつ適用）。``||Delta l_n||_inf <= ||Delta l_n||_2`` なので、
    argmax 不変の判定（``U_n < gamma_n / 2``）にも使える。

    引数は ``mlp_logits_error_bound`` と同じ。

    戻り値:
        shape ``(batch,)`` の Tensor。``X`` の dtype / device / autograd を維持する。
        ``sum_n U_n**2 = ||X||_F^2 ||Delta W_1||_2^2 ||W_2||_2^2`` なので、
        ``sqrt(sum(U**2))`` は batch 上界に一致する。
        ``delta_W1`` または ``W2`` が零行列なら全 sample が厳密に 0。
        共有ノルムが ``inf`` でも、零入力行の上界は 0。非零行の上界が表現範囲を
        超えるときは ``inf`` になり得る。``NaN`` は返さない。

    例外:
        ``linear_output_error_bounds`` と同じ。NaN / inf は ``ValueError``。
    """
    _check_mlp_inputs(X, delta_W1, W2)
    if not _is_nonzero(delta_W1) or not _is_nonzero(W2):
        return (X * 0).sum(dim=1) + (delta_W1 * 0).sum() + (W2 * 0).sum()

    factor = _mul_nonnegative(_spectral(delta_W1), _spectral(W2))
    row_norms = torch.linalg.vector_norm(X, dim=1)
    # 有限で正の共有因子は、零行も ``0 * factor = 0``、極端な行は ``inf`` になり得る。
    if bool(torch.isfinite(factor)) and not bool(factor == 0):
        return row_norms * factor

    row_zero = torch.all(X == 0, dim=1)
    anchored_rows = (X * 0).sum(dim=1)
    if bool(torch.isfinite(factor)):
        # 共有因子が underflow で 0。有限な行ノルムとの積は従来どおり 0。
        # 非有限な行ノルムは ``inf * 0`` を避け、上界として ``+inf``。
        finite_rows = torch.where(torch.isfinite(row_norms), row_norms, torch.zeros_like(row_norms))
        finite_product = finite_rows * factor
        return torch.where(torch.isfinite(row_norms), finite_product, _posinf_like(row_norms))
    # 共有因子が inf。零入力行は厳密に 0、非零行は inf（``0 * inf`` を作らない）。
    return torch.where(row_zero, anchored_rows, _posinf_like(row_norms))
