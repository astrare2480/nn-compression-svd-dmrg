"""層単位の理論 MACs（モデル構造に依存しない）。

Linear は ``in * out``、Conv2d は出力空間サイズ × カーネル積和で見積もる。
"""


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
    """
    return in_features * rank + rank * out_features


def conv2d_macs(conv, out_h: int, out_w: int) -> int:
    """1サンプルが Conv2d を通るときの MACs を返す。

    出力の各位置・各チャネルが ``in_ch * kH * kW`` 回の積和をするので
    ``out_h * out_w * out_ch * in_ch * kH * kW``。
    ``weight.shape[1]`` は grouped convolution ではすでに
    ``in_channels / groups`` なので、さらに ``groups`` では割らない。
    """
    out_ch, in_ch_per_group, k_h, k_w = conv.weight.shape
    return out_h * out_w * out_ch * in_ch_per_group * k_h * k_w


def compressed_conv2d_macs(conv, rank: int, out_h: int, out_w: int) -> int:
    """``factorize_conv2d_layer`` と同じ 2 層 Conv の MACs を返す。

    1 層目: ``Conv(in -> rank, kH×kW)``
    2 層目: ``Conv(rank -> out, 1×1)``
    空間サイズは分解前の出力と同じ。
    """
    out_ch, in_ch, k_h, k_w = conv.weight.shape
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
    ``compute_reduction`` は ``1 - compressed / baseline``。
    """
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
