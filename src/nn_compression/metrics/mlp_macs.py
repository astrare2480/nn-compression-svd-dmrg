"""現在の MLP 圧縮前後の理論計算量（MACs）を見積もる関数。

移植元:
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
    「MACs (Multiply-Accumulate Operations)」のセル

このモジュールは現在の 784 -> 512 -> 256 -> 10 MLP を対象にする。
層単位の計算は ``metrics.macs`` に置く。
"""

from .macs import compressed_linear_macs, linear_macs


def estimate_mlp_macs(model, fc1_rank, fc2_rank, verbose=True):
    """現在の MLP と SVD 圧縮版の MACs・削減率を比較する。

    ``fc1`` と ``fc2`` は低ランク2層へ置換済み、``fc3`` は元の Linear
    のまま、という元Notebookと同じ前提で計算する。
    """
    baseline_macs = (
        linear_macs(model.fc1)
        + linear_macs(model.fc2)
        + linear_macs(model.fc3)
    )

    compressed_macs = (
        compressed_linear_macs(784, 512, fc1_rank)
        + compressed_linear_macs(512, 256, fc2_rank)
        + linear_macs(model.fc3)
    )
    compute_reduction_rate = 1.0 - compressed_macs / baseline_macs

    if verbose:
        print("Baseline MACs:", baseline_macs)
        print("Compressed MACs:", compressed_macs)
        print(f"Compute reduction: {compute_reduction_rate:.2%}")

    return baseline_macs, compressed_macs, compute_reduction_rate
