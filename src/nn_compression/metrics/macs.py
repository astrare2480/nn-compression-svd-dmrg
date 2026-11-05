"""層単位の理論 MACs（モデル構造に依存しない）。

Linear は ``in * out``、Conv2d は出力空間サイズ × カーネル積和で見積もる。
"""

import operator
from collections.abc import Sequence

import torch


def _coerce_positive_int_scalar(value, *, name: str) -> int:
    """bool / Tensor scalar を拒否し、正の整数 scalar を ``int`` として返す。"""
    if isinstance(value, bool):
        raise TypeError(
            f"{name} は bool 以外の正の整数である必要があります: {value!r}"
        )
    if torch.is_tensor(value):
        raise TypeError(
            f"{name} に Tensor scalar は指定できません: {value!r}"
        )
    try:
        value_int = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} は正の整数である必要があります: {value!r}"
        ) from exc
    if value_int <= 0:
        raise ValueError(
            f"{name} は正の整数である必要があります: {value_int}"
        )
    return value_int


def _coerce_macs_rank(rank: int, max_rank: int, *, name: str = "rank") -> int:
    """compressed MACs 計算用に rank を整数化し、``1 <= rank <= max_rank`` を検証する。

    SVD 側の ``coerce_rank`` と同じ contract（bool / Tensor scalar 拒否・
    範囲チェック）をここでも独立に確認する。実際に生成できない分解
    （rank=0 / 負 / 元の行列の最大 rank 超過）に対して、もっともらしい
    MACs 値を返さない。metrics モジュールが compression パッケージへ
    依存しないよう、``coerce_rank`` を import せずここで小さく再実装する。
    """
    if isinstance(rank, bool):
        raise TypeError(
            f"{name} は bool 以外の整数である必要があります: {rank!r}"
        )
    if torch.is_tensor(rank):
        raise TypeError(
            f"{name} に Tensor scalar は指定できません: {rank!r}"
        )
    try:
        rank_int = operator.index(rank)
    except TypeError as exc:
        raise TypeError(f"{name} は整数である必要があります: {rank!r}") from exc
    if not 1 <= rank_int <= max_rank:
        raise ValueError(
            f"{name} は 1〜{max_rank} の範囲で指定してください: {rank_int}"
        )
    return rank_int


def tt_linear_direct_contraction_macs(
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    ranks: Sequence[int],
) -> tuple[int, ...]:
    """左から右へ ``torch.tensordot`` する TTLinear の、1 sample あたり site 別 MACs。

    site ``k``（1 始まり）の積和回数は

    ``(prod_{ell < k} m_ell) * (prod_{ell > k} n_ell) * m_k * r_{k-1} * n_k * r_k``

    である。bias、活性化、reshape、backward、SVD は含めない。
    戻り値は site 1 から site ``d`` までの整数列。
    """
    out_tuple = tuple(
        _coerce_positive_int_scalar(mode, name=f"out_modes[{index}]")
        for index, mode in enumerate(out_modes)
    )
    in_tuple = tuple(
        _coerce_positive_int_scalar(mode, name=f"in_modes[{index}]")
        for index, mode in enumerate(in_modes)
    )
    rank_tuple = tuple(
        _coerce_positive_int_scalar(rank, name=f"ranks[{index}]")
        for index, rank in enumerate(ranks)
    )
    if len(out_tuple) < 1:
        raise ValueError("out_modes は 1 site 以上必要です。")
    if len(out_tuple) != len(in_tuple):
        raise ValueError(
            "out_modes と in_modes の長さが一致しません: "
            f"{len(out_tuple)} と {len(in_tuple)}"
        )
    if len(rank_tuple) != len(out_tuple) + 1:
        raise ValueError(
            "ranks の長さは mode 数 + 1 である必要があります: "
            f"ranks={len(rank_tuple)} modes={len(out_tuple)}"
        )
    if rank_tuple[0] != 1 or rank_tuple[-1] != 1:
        raise ValueError(
            "境界 rank は両方 1 である必要があります: "
            f"r_0={rank_tuple[0]} r_d={rank_tuple[-1]}"
        )
    site_count = len(out_tuple)
    costs: list[int] = []
    for site in range(site_count):
        left_out = 1
        for earlier in range(site):
            left_out *= out_tuple[earlier]
        right_in = 1
        for later in range(site + 1, site_count):
            right_in *= in_tuple[later]
        costs.append(
            left_out
            * right_in
            * out_tuple[site]
            * rank_tuple[site]
            * in_tuple[site]
            * rank_tuple[site + 1]
        )
    return tuple(costs)


def tt_conv2d_direct_contraction_macs(
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    ranks: Sequence[int],
    out_height: int,
    out_width: int,
) -> int:
    """TTConv2d の直接縮約 forward に対応する、1 画像あたりの理論 MACs を返す。

    ``tt_conv2d_forward`` は各出力位置の patch に ``tt_linear_forward`` を
    1 回適用する。戻り値は、その site 別 MACs の和に出力位置数
    ``out_height * out_width`` を掛けた整数である。

    ``out_height`` と ``out_width`` は、stride・padding・dilation・入力空間サイズから
    決まる実際の出力サイズを呼び出し側が渡す。rank の位置は site 別の式が区別する。
    unfold、bias、ReLU、pooling、backward は含めない。dense kernel 再構成の
    MACs でも、2 層 Conv 分解の MACs でもない。
    """
    height = _coerce_positive_int_scalar(out_height, name="out_height")
    width = _coerce_positive_int_scalar(out_width, name="out_width")
    site_costs = tt_linear_direct_contraction_macs(out_modes, in_modes, ranks)
    return height * width * sum(site_costs)


def linear_macs(layer) -> int:
    """1サンプルを Linear 層へ通すときの MACs を返す。

    Linear(in_features -> out_features) では、各出力要素が
    ``in_features`` 回の積和演算を行うため、概算は
    ``in_features * out_features`` となる。
    """
    return layer.in_features * layer.out_features


def compressed_linear_macs(in_features: int, out_features: int, rank: int) -> int:
    """rank ``r`` の 2 層分解 Linear の MACs を返す。

    元の ``in -> out`` を ``in -> r -> out`` に置き換えるため、
    演算量は ``in * r + r * out`` で見積もる。
    rank は SVD 側と同じ contract（bool / Tensor scalar 拒否、
    ``1 <= rank <= min(in_features, out_features)``）で検証する。
    """
    rank = _coerce_macs_rank(
        rank, min(in_features, out_features), name="rank"
    )
    return in_features * rank + rank * out_features


def conv2d_macs(conv, out_h: int, out_w: int) -> int:
    """1サンプルが Conv2d を通るときの MACs を返す。

    出力の各位置・各チャネルが ``in_ch * kH * kW`` 回の積和をするので
    ``out_h * out_w * out_ch * in_ch * kH * kW``。
    ``weight.shape[1]`` は grouped convolution ではすでに
    ``in_channels / groups`` なので、さらに ``groups`` では割らない。
    ``out_h`` / ``out_w`` は bool / Tensor scalar 以外の正の整数を要求する。
    """
    out_h = _coerce_positive_int_scalar(out_h, name="out_h")
    out_w = _coerce_positive_int_scalar(out_w, name="out_w")
    out_ch, in_ch_per_group, k_h, k_w = conv.weight.shape
    return out_h * out_w * out_ch * in_ch_per_group * k_h * k_w


def _reject_grouped_conv_for_compressed(conv, *, context: str) -> None:
    """compressed / estimated MACs は groups=1 の Conv2d のみ対応する。"""
    if conv.groups != 1:
        raise ValueError(
            f"{context} は groups=1 の Conv2d のみ対応しています: "
            f"groups={conv.groups}"
        )


def compressed_conv2d_macs(conv, rank: int, out_h: int, out_w: int) -> int:
    """``factorize_conv2d_layer`` と同じ 2 層 Conv の MACs を返す。

    1 層目: ``Conv(in -> rank, kH×kW)``
    2 層目: ``Conv(rank -> out, 1×1)``
    空間サイズは分解前の出力と同じ。
    rank は SVD 側と同じ contract（bool / Tensor scalar 拒否、
    ``1 <= rank <= min(out_ch, in_ch * kH * kW)``）で検証する。
    ``out_h`` / ``out_w`` は bool / Tensor scalar 以外の正の整数を要求する。
    """
    _reject_grouped_conv_for_compressed(conv, context="compressed_conv2d_macs")
    out_ch, in_ch, k_h, k_w = conv.weight.shape
    rank = _coerce_macs_rank(rank, min(out_ch, in_ch * k_h * k_w), name="rank")
    out_h = _coerce_positive_int_scalar(out_h, name="out_h")
    out_w = _coerce_positive_int_scalar(out_w, name="out_w")
    first_layer_macs = out_h * out_w * rank * in_ch * k_h * k_w
    second_layer_macs = out_h * out_w * out_ch * rank
    return first_layer_macs + second_layer_macs


def estimate_conv2d_macs(
    conv,
    rank: int,
    out_hw: tuple[int, int],
    verbose: bool = True,
) -> tuple[int, int, float]:
    """1つの Conv2d を低ランク 2 層にしたときの MACs と削減率。

    モデル構造には依存しない。空間サイズ ``out_hw`` は呼び出し側が渡す。
    ``out_hw`` の各値は bool / Tensor scalar 以外の正の整数を要求する。
    ``compute_reduction`` は ``1 - compressed / baseline``。
    """
    _reject_grouped_conv_for_compressed(conv, context="estimate_conv2d_macs")
    out_h, out_w = out_hw
    baseline_macs = conv2d_macs(conv, out_h, out_w)
    compressed_macs = compressed_conv2d_macs(conv, rank, out_h, out_w)
    compute_reduction = 1.0 - compressed_macs / baseline_macs

    if verbose:
        print("Baseline MACs:", baseline_macs)
        print("Compressed MACs:", compressed_macs)
        print(f"Compute reduction: {compute_reduction:.2%}")

    return baseline_macs, compressed_macs, compute_reduction


factorized_linear_macs = compressed_linear_macs
factorized_conv2d_macs = compressed_conv2d_macs
