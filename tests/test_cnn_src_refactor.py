"""Fashion-MNIST CNN 実験の src 切り出しが、Notebook 時代の数値と一致することを確認する。"""

import torch
from torch import nn

from nn_compression.compression import (
    RebuildSVD,
    SVD,
    factorize_Conv2d_layer,
    factorize_conv2d_layer,
    factorize_linear_layer,
    factorize_named_conv2d,
    factorize_named_linear,
    rebuild_linear_from_svd,
    retained_energy,
    truncated_svd,
)
from nn_compression.compression.mlp_svd import (
    make_one_layer_svd_model,
    make_two_layer_svd_model,
)
from nn_compression.metrics import (
    compressed_conv2d_macs,
    compressed_linear_macs,
    conv2d_macs,
    count_parameters,
    estimate_cnn_macs,
    estimate_mlp_macs,
    linear_macs,
)
from nn_compression.models import FashionMNISTCNN, MNISTMLP


def test_linear_full_rank_matches_original():
    torch.manual_seed(0)
    layer = nn.Linear(12, 8)
    rank = min(layer.in_features, layer.out_features)
    factorized = factorize_linear_layer(layer, rank)
    x = torch.randn(4, 12)
    torch.testing.assert_close(factorized(x), layer(x), atol=1e-5, rtol=1e-5)


def test_conv2d_full_rank_matches_original():
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 16, kernel_size=3, padding=1)
    rank = min(conv.out_channels, conv.in_channels * 3 * 3)
    factorized = factorize_conv2d_layer(conv, rank)
    x = torch.randn(2, 8, 14, 14)
    out_factorized = factorized(x)
    out_original = conv(x)
    torch.testing.assert_close(out_factorized, out_original, atol=1e-4, rtol=1e-4)
    assert out_factorized.shape == out_original.shape


def test_retained_energy_linear_and_conv():
    torch.manual_seed(0)
    linear = nn.Linear(10, 6)
    conv = nn.Conv2d(4, 8, kernel_size=3, padding=1)
    linear_full = min(linear.in_features, linear.out_features)
    conv_full = min(conv.out_channels, conv.in_channels * 9)
    assert retained_energy(linear, linear_full) == 1.0
    assert retained_energy(conv, conv_full) == 1.0
    assert 0.0 < retained_energy(linear, 1) <= 1.0
    assert 0.0 < retained_energy(conv, 1) <= 1.0


def test_fashion_mnist_cnn_parameter_count():
    model = FashionMNISTCNN()
    assert count_parameters(model) == 421_642


def test_05_macs_and_compressed_parameter_count():
    model = FashionMNISTCNN()
    baseline, compressed, reduction = estimate_cnn_macs(
        model,
        conv2_rank=28,
        fc1_rank=24,
        verbose=False,
    )
    assert baseline == 4_241_152
    assert compressed == 2_237_184
    assert abs(reduction - (1.0 - 2_237_184 / 4_241_152)) < 1e-12

    same = estimate_cnn_macs(
        model,
        verbose=False,
        conv_ranks={"conv2": 28},
        linear_ranks={"fc1": 24},
        conv_output_hw={"conv1": (28, 28), "conv2": (14, 14)},
    )
    assert same == (baseline, compressed, reduction)

    compressed_model = factorize_named_linear(model, "fc1", 24)
    compressed_model.conv2 = factorize_conv2d_layer(model.conv2, 28)
    assert count_parameters(compressed_model) == 89_994


def test_named_conv2d_does_not_mutate_original():
    model = FashionMNISTCNN()
    original = model.conv2
    compressed = factorize_named_conv2d(model, "conv2", 8)
    assert compressed.conv2 is not original
    assert isinstance(model.conv2, nn.Conv2d)


def test_mlp_imports_and_full_rank_two_layer():
    torch.manual_seed(0)
    model = MNISTMLP()
    compressed = make_two_layer_svd_model(model, 512, 256)
    x = torch.randn(2, 1, 28, 28)
    torch.testing.assert_close(compressed(x), model(x), atol=1e-4, rtol=1e-4)
    baseline, _, _ = estimate_mlp_macs(model, 128, 64, verbose=False)
    assert baseline == (
        linear_macs(model.fc1) + linear_macs(model.fc2) + linear_macs(model.fc3)
    )
    one_layer = make_one_layer_svd_model(model, 16, 16)
    assert one_layer.fc1.out_features == 512


def test_backward_compat_aliases():
    assert SVD is truncated_svd
    assert RebuildSVD is rebuild_linear_from_svd
    assert factorize_Conv2d_layer is factorize_conv2d_layer
    assert compressed_linear_macs(10, 4, 2) == 28
    conv = nn.Conv2d(3, 5, 3)
    assert conv2d_macs(conv, 8, 8) == 8 * 8 * 5 * 3 * 9
    assert compressed_conv2d_macs(conv, 2, 8, 8) == (
        8 * 8 * 2 * 3 * 9 + 8 * 8 * 5 * 2
    )


def test_linear_bias_false_full_rank_matches_original():
    torch.manual_seed(0)
    layer = nn.Linear(12, 8, bias=False)
    rank = min(layer.in_features, layer.out_features)
    factorized = factorize_linear_layer(layer, rank)
    rebuilt = rebuild_linear_from_svd(
        *truncated_svd(layer.weight.detach(), rank),
        layer,
    )
    x = torch.randn(4, 12)
    torch.testing.assert_close(factorized(x), layer(x), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rebuilt(x), layer(x), atol=1e-5, rtol=1e-5)
    assert factorized[1].bias is None
    assert rebuilt.bias is None
    assert factorized[0].weight.dtype == layer.weight.dtype
    assert rebuilt.weight.dtype == layer.weight.dtype


def test_conv2d_macs_groups_two():
    conv = nn.Conv2d(8, 16, kernel_size=3, groups=2)
    out_h, out_w = 10, 12
    out_ch, in_ch_per_group, k_h, k_w = conv.weight.shape
    assert (out_ch, in_ch_per_group, k_h, k_w) == (16, 4, 3, 3)
    expected = out_h * out_w * 16 * 4 * 3 * 3
    assert conv2d_macs(conv, out_h, out_w) == expected


if __name__ == "__main__":
    tests = [
        test_linear_full_rank_matches_original,
        test_conv2d_full_rank_matches_original,
        test_retained_energy_linear_and_conv,
        test_fashion_mnist_cnn_parameter_count,
        test_05_macs_and_compressed_parameter_count,
        test_named_conv2d_does_not_mutate_original,
        test_mlp_imports_and_full_rank_two_layer,
        test_backward_compat_aliases,
        test_linear_bias_false_full_rank_matches_original,
        test_conv2d_macs_groups_two,
    ]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("all passed")
