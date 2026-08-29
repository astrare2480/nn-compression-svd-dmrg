"""層単位の理論 MACs（モデル構造に依存しない）。

Linear は ``in * out``、Conv2d は出力空間サイズ × カーネル積和で見積もる。
"""

import operator

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
