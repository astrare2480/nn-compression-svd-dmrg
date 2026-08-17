"""CNN の理論 MACs 集計。

層単位の計算は ``metrics.macs``。
空間サイズや圧縮対象層は引数で渡す。省略時だけ Fashion-MNIST
CNN（conv1 28×28、conv2 14×14、fc1 / fc2）になる。
"""

from .macs import (
    compressed_conv2d_macs,
    compressed_linear_macs,
    conv2d_macs,
    estimate_conv2d_macs,
    linear_macs,
)

CONV1_OUTPUT_HW = (28, 28)
CONV2_OUTPUT_HW = (14, 14)
FASHION_MNIST_CONV_OUTPUT_HW = {
    "conv1": CONV1_OUTPUT_HW,
    "conv2": CONV2_OUTPUT_HW,
}
FASHION_MNIST_LINEAR_LAYER_NAMES = ("fc1", "fc2")


def estimate_cnn_linear_macs(model, fc1_rank, verbose=True):
    """fc1 だけを分解したときの Linear 部 MACs・削減率を比較する。

    02 / 03 と同じく、比較対象は ``fc1`` と ``fc2`` のみ。
    ``fc1`` は ``in → rank → out``、``fc2`` は元の Linear のまま。
    """
    baseline_macs = linear_macs(model.fc1) + linear_macs(model.fc2)
    compressed_macs = (
        compressed_linear_macs(
            model.fc1.weight.shape[0],
            model.fc1.weight.shape[1],
            fc1_rank,
        )
        + linear_macs(model.fc2)
    )
    compute_reduction_rate = 1.0 - compressed_macs / baseline_macs

    if verbose:
        print("Baseline MACs:", baseline_macs)
        print("Compressed MACs:", compressed_macs)
        print(f"Compute reduction: {compute_reduction_rate:.2%}")

    return baseline_macs, compressed_macs, compute_reduction_rate


def estimate_cnn_conv2_macs(
    model,
    rank,
    verbose=True,
    *,
    layer_name: str = "conv2",
    out_hw: tuple[int, int] | None = None,
):
    """指定した Conv2d だけを分解したときの MACs・削減率を比較する。

    他層は圧縮しないので、比較対象はその 1 層のみ。
    ``layer_name`` と ``out_hw`` を省略すると Fashion-MNIST 04 と同じ
    ``conv2`` / 14×14 になる。
    """
    if out_hw is None:
        out_hw = CONV2_OUTPUT_HW

    conv = getattr(model, layer_name)
    return estimate_conv2d_macs(
        conv,
        rank,
        out_hw,
        verbose=verbose,
    )


def estimate_cnn_macs(
    model,
    conv2_rank: int | None = None,
    fc1_rank: int | None = None,
    verbose: bool = True,
    *,
    conv_ranks: dict[str, int] | None = None,
    linear_ranks: dict[str, int] | None = None,
    conv_output_hw: dict[str, tuple[int, int]] | None = None,
    linear_layer_names: tuple[str, ...] = FASHION_MNIST_LINEAR_LAYER_NAMES,
) -> tuple[int, int, float]:
    """CNN 全体の MACs・削減率を比較する。

    ``conv_output_hw`` に載せた Conv と ``linear_layer_names`` の Linear
    を合計する。``conv_ranks`` / ``linear_ranks`` にある層だけ低ランク
    2 層として数え、他は元の層のままにする。

    位置引数 ``conv2_rank`` / ``fc1_rank`` だけ渡すと Fashion-MNIST 05
    と同じ conv2 + fc1 圧縮、空間サイズ 28×28 / 14×14 になる。
    """
    if conv_output_hw is None:
        conv_output_hw = FASHION_MNIST_CONV_OUTPUT_HW
    if conv_ranks is None:
        if conv2_rank is None:
            raise TypeError("conv2_rank または conv_ranks を指定してください。")
        conv_ranks = {"conv2": conv2_rank}
    if linear_ranks is None:
        if fc1_rank is None:
            raise TypeError("fc1_rank または linear_ranks を指定してください。")
        linear_ranks = {"fc1": fc1_rank}

    baseline_macs = 0
    compressed_macs = 0

    for layer_name, out_hw in conv_output_hw.items():
        conv = getattr(model, layer_name)
        out_h, out_w = out_hw
        layer_baseline = conv2d_macs(conv, out_h, out_w)
        baseline_macs += layer_baseline
        if layer_name in conv_ranks:
            compressed_macs += compressed_conv2d_macs(
                conv,
                conv_ranks[layer_name],
                out_h,
                out_w,
            )
        else:
            compressed_macs += layer_baseline

    for layer_name in linear_layer_names:
        linear = getattr(model, layer_name)
        layer_baseline = linear_macs(linear)
        baseline_macs += layer_baseline
        if layer_name in linear_ranks:
            compressed_macs += compressed_linear_macs(
                linear.in_features,
                linear.out_features,
                linear_ranks[layer_name],
            )
        else:
            compressed_macs += layer_baseline

    reduction = 1.0 - compressed_macs / baseline_macs

    if verbose:
        print("Baseline total MACs:", baseline_macs)
        print("Compressed total MACs:", compressed_macs)
        print(f"Compute reduction: {reduction:.2%}")

    return baseline_macs, compressed_macs, reduction
