"""TT/MPS の canonical form（mixed-canonical form）と center 移動。

参照元:
    notebooks/30_tt_mps/00_fundamentals/04_gauge_freedom_qr_L2.ipynb
    notebooks/30_tt_mps/00_fundamentals/05_right_orthogonalization_right_block.ipynb
    notebooks/30_tt_mps/00_fundamentals/06_mixed_canonical_form_fixed.ipynb
    notebooks/30_tt_mps/00_fundamentals/07_orthogonality_center_qr_move.ipynb
    notebooks/30_tt_mps/00_fundamentals/08_svd_center_move_and_schmidt_form.ipynb

Notebook側は3コア固定・小さいshape固定の演習用実装（``reconstruct_tt3`` 等）
だったのに対し、ここでは一般の ``d`` 階 TT へ一般化する。左右のQR step自体は
Notebook 04/07/08 のアルゴリズムのまま変更しない。
"""

from __future__ import annotations

import torch

from .svd import coerce_integer_scalar
from .tt_validation import validate_tt_cores, validate_tt_dtype


def _scale_safe_frobenius_norm(matrix: torch.Tensor) -> torch.Tensor:
    """行列のFrobenius normを、最大絶対値で正規化して計算する。"""
    scale = matrix.abs().amax()
    is_zero = scale == 0
    safe_scale = torch.where(is_zero, torch.ones_like(scale), scale)
    normalized_norm = torch.linalg.vector_norm((matrix / safe_scale).reshape(-1))
    return torch.where(is_zero, torch.zeros_like(scale), scale * normalized_norm)


def _qr_step_left_to_right(cores: list[torch.Tensor], k: int) -> None:
    """``cores[k]`` を左直交化し、余りを ``cores[k + 1]`` へ吸収する（in-place）。

    ``cores[k]`` を ``(r_left * n_k, r_right)`` へ展開して
    ``torch.linalg.qr(..., mode="reduced")`` を適用し、``Q`` を新しい
    ``cores[k]``（左直交）に、``R`` を ``cores[k + 1]`` の左 bond へ吸収する。

    ``cores`` は呼び出し側で複製済みのリストを渡すこと（この関数はリスト
    要素を書き換える）。
    """
    r_left, n_k, r_right = cores[k].shape
    mat = cores[k].reshape(r_left * n_k, r_right)

    Q, R = torch.linalg.qr(mat, mode="reduced")
    new_rank = Q.shape[1]

    # Q: 新しい左直交 core。R: 右隣 core の左 bond へ吸収する残差。
    cores[k] = Q.reshape(r_left, n_k, new_rank)
    cores[k + 1] = torch.tensordot(R, cores[k + 1], dims=([1], [0]))


def _qr_step_right_to_left(cores: list[torch.Tensor], k: int) -> None:
    """``cores[k]`` を右直交化し、余りを ``cores[k - 1]`` へ吸収する（in-place）。

    ``cores[k]`` を ``(r_left, n_k * r_right)`` へ展開し、その転置
    （``.mT``）に ``torch.linalg.qr(..., mode="reduced")`` を適用する。
    ``Q.mT`` を新しい ``cores[k]``（右直交）に、``R.mT`` を
    ``cores[k - 1]`` の右 bond へ吸収する。

    ``cores`` は呼び出し側で複製済みのリストを渡すこと（この関数はリスト
    要素を書き換える）。
    """
    r_left, n_k, r_right = cores[k].shape
    mat = cores[k].reshape(r_left, n_k * r_right)

    # 行を直交化したいので、転置してから列直交な reduced QR を使う。
    Q, R = torch.linalg.qr(mat.mT, mode="reduced")
    new_rank = Q.shape[1]

    cores[k] = Q.mT.reshape(new_rank, n_k, r_right)
    cores[k - 1] = torch.tensordot(cores[k - 1], R.mT, dims=([-1], [0]))


def tt_canonicalize(cores: list[torch.Tensor], center: int) -> list[torch.Tensor]:
    """任意の d 階 TT を、指定 ``center`` の mixed-canonical form へ変換する。

    cores:
        ``validate_tt_cores`` の contract を満たす TT core 列。この
        リストおよび各 Tensor は変更しない。
    center:
        orthogonality center とするサイト index。``0 <= center < d``
        （``d = len(cores)``）。bool は整数として受け付けない。

    戻り値:
        新しい TT core 列。``center`` より左の core はすべて左直交
        （``core.reshape(r_left * n_k, r_right)`` の列が正規直交）、
        ``center`` より右の core はすべて右直交（``core.reshape(r_left,
        n_k * r_right)`` の行が正規直交）になる。``center`` 自身の core
        には直交条件を課さない（一般の中心係数テンソル）。

    Note:
        ``d == 1`` の場合、内部 bond が存在しないため入力の複製をそのまま
        返す。dense 再構成値・dtype・device は変換前後で変わらない
        （QR は可逆な gauge 変換であり、情報を捨てない）。

        既知の制約（今回のsrc化では対応しない）: ``torch.linalg.qr`` と、
        その余り ``R`` / ``R.mT`` の隣接coreへの吸収（``torch.tensordot``）
        は、入力の dtype（float32/float64）のまま行われる。coreごとの
        スケールが極端に不均衡なgauge（例: あるcoreは``1e30``、隣の
        coreは``1e-30``程度）では、この中間演算自体がoverflow・
        underflowし得る。最終的なdense tensorの値が表現可能な範囲でも、
        中間のcore同士の積が同じ範囲に収まるとは限らない（例: float32の
        rank-1 TTで、core scaleが``1e30``, ``1e-30``, ``1e-30``の場合、
        denseな値は``1e-30``程度で表現できるが、right-to-left
        canonicalization中に``1e-30 * 1e-30 -> 1e-60 -> 0``へ
        underflowし、``_scale_safe_norm`` 等のscale-safeな処理へ到達する
        前に情報が失われる）。この関数は、通常の正準化済み・
        scale-balancedなTT（学習済みNNの重みをTT分解したもの等）を対象
        としており、任意のdynamic range（core間のスケールが極端に
        不均衡な入力）に対するscale-invariant性は保証しない。
    """
    validate_tt_cores(cores)
    validate_tt_dtype(cores[0], name="cores[0]")

    d = len(cores)
    center = coerce_integer_scalar(center, name="center")
    if not 0 <= center < d:
        raise ValueError(f"center={center} は 0 <= center < {d} を満たす必要があります。")

    result = [core.clone() for core in cores]
    if d == 1:
        return result

    # center より左を左から順に左直交化し、Rを右隣（最終的にcenter）へ吸収する。
    for k in range(0, center):
        _qr_step_left_to_right(result, k)

    # center より右を右から順に右直交化し、R.mTを左隣（最終的にcenter）へ吸収する。
    for k in range(d - 1, center, -1):
        _qr_step_right_to_left(result, k)

    return result


def tt_canonicality_errors(
    cores: list[torch.Tensor],
    center: int,
) -> dict[str, list[torch.Tensor]]:
    r"""指定centerに対する左右canonicalityのFrobenius誤差を返す。

    ``center`` より左のcoreでは左展開行列 ``M_k`` に対する
    ``||M_k.mT @ M_k - I||_F``、右のcoreでは右展開行列 ``N_k`` に
    対する ``||N_k @ N_k.mT - I||_F`` を計算する。center自身は一般の
    中心係数Tensorなので評価対象に含めない。

    戻り値は ``{"left": [...], "right": [...]}``。``left`` はサイト
    ``0, ..., center-1``、``right`` はサイト ``center+1, ..., d-1``
    の順で並ぶ。各要素は入力と同じdtype/deviceのスカラーTensorで、
    autograd graphを維持する。入力core列は変更しない。
    """
    validate_tt_cores(cores)
    validate_tt_dtype(cores[0], name="cores[0]")

    d = len(cores)
    center = coerce_integer_scalar(center, name="center")
    if not 0 <= center < d:
        raise ValueError(f"center={center} は 0 <= center < {d} を満たす必要があります。")

    left_errors: list[torch.Tensor] = []
    for core in cores[:center]:
        r_left, n_k, r_right = core.shape
        mat = core.reshape(r_left * n_k, r_right)
        identity = torch.eye(r_right, dtype=core.dtype, device=core.device)
        gram_error = mat.mT @ mat - identity
        left_errors.append(_scale_safe_frobenius_norm(gram_error))

    right_errors: list[torch.Tensor] = []
    for core in cores[center + 1 :]:
        r_left, n_k, r_right = core.shape
        mat = core.reshape(r_left, n_k * r_right)
        identity = torch.eye(r_left, dtype=core.dtype, device=core.device)
        gram_error = mat @ mat.mT - identity
        right_errors.append(_scale_safe_frobenius_norm(gram_error))

    return {"left": left_errors, "right": right_errors}


def tt_move_center(
    cores: list[torch.Tensor],
    source: int,
    target: int,
) -> list[torch.Tensor]:
    """center=``source`` の mixed-canonical TT の center を ``target`` へ移動する。

    cores:
        ``validate_tt_cores`` の contract を満たす TT core 列。この
        リストおよび各 Tensor は変更しない。

        **前提（呼び出し側の責務）**: ``cores`` は既に
        ``tt_canonicalize(cores, source)`` などによって center=``source``
        の mixed-canonical form になっていること。この関数はこの前提を
        検証しない（全 core の直交性を検査するコストを避けるため）。
        前提が崩れている入力に対しては、dense 再構成値は変換前後で
        一致するが、``target`` 以外の位置での左右直交性は保証しない。
    source:
        現在の center サイト index。``0 <= source < d``。
    target:
        移動先の center サイト index。``0 <= target < d``。

    戻り値:
        ``source`` から ``target`` まで、QR step を1サイトずつ繰り返して
        center を移動した新しい TT core 列。``source == target`` なら
        入力の複製をそのまま返す。

    Note:
        bool は整数として受け付けない。``d == 1`` では ``source ==
        target == 0`` しか valid ではないため、常に複製を返す。
    """
    validate_tt_cores(cores)
    validate_tt_dtype(cores[0], name="cores[0]")

    d = len(cores)
    source = coerce_integer_scalar(source, name="source")
    target = coerce_integer_scalar(target, name="target")

    if not 0 <= source < d:
        raise ValueError(f"source={source} は 0 <= source < {d} を満たす必要があります。")
    if not 0 <= target < d:
        raise ValueError(f"target={target} は 0 <= target < {d} を満たす必要があります。")

    result = [core.clone() for core in cores]
    if source == target:
        return result

    if source < target:
        # center を1サイトずつ右へ移動: 現center coreを左直交化し、Rを右隣へ吸収。
        for k in range(source, target):
            _qr_step_left_to_right(result, k)
    else:
        # center を1サイトずつ左へ移動: 現center coreを右直交化し、R.mTを左隣へ吸収。
        for k in range(source, target, -1):
            _qr_step_right_to_left(result, k)

    return result
