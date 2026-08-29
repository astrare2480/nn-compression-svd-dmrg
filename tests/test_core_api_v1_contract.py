"""Core API v1 のself reviewで追加した境界contractテスト。"""

import pytest
import torch
from torch import nn

from nn_compression.compression import core_from_factors, reconstruct_tucker
from nn_compression.metrics import (
    compressed_conv2d_macs,
    conv2d_macs,
    estimate_conv2d_macs,
)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5])
def test_conv2d_macs_rejects_invalid_out_h(value):
    conv = nn.Conv2d(3, 5, kernel_size=3)
    with pytest.raises((TypeError, ValueError)):
        conv2d_macs(conv, out_h=value, out_w=8)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5])
def test_conv2d_macs_rejects_invalid_out_w(value):
    conv = nn.Conv2d(3, 5, kernel_size=3)
    with pytest.raises((TypeError, ValueError)):
        conv2d_macs(conv, out_h=8, out_w=value)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5])
def test_compressed_conv2d_macs_rejects_invalid_out_h(value):
    conv = nn.Conv2d(3, 5, kernel_size=3)
    with pytest.raises((TypeError, ValueError)):
        compressed_conv2d_macs(conv, rank=3, out_h=value, out_w=8)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5])
def test_compressed_conv2d_macs_rejects_invalid_out_w(value):
    conv = nn.Conv2d(3, 5, kernel_size=3)
    with pytest.raises((TypeError, ValueError)):
        compressed_conv2d_macs(conv, rank=3, out_h=8, out_w=value)


def test_conv2d_macs_accepts_positive_integer_boundaries():
    conv = nn.Conv2d(3, 5, kernel_size=3)
    assert conv2d_macs(conv, out_h=1, out_w=1) == 5 * 3 * 3 * 3
    assert compressed_conv2d_macs(conv, rank=3, out_h=1, out_w=1) > 0


@pytest.mark.parametrize("out_hw", [(True, 8), (8, False), (0, 8), (8, 0), (1.5, 8), (8, 1.5)])
def test_estimate_conv2d_macs_rejects_invalid_out_hw(out_hw):
    conv = nn.Conv2d(3, 5, kernel_size=3)
    with pytest.raises((TypeError, ValueError)):
        estimate_conv2d_macs(conv, rank=3, out_hw=out_hw, verbose=False)


def test_reconstruct_tucker_rejects_1d_core_even_without_factors():
    with pytest.raises(ValueError, match="2次元以上"):
        reconstruct_tucker(torch.randn(5), {})


def test_core_from_factors_rejects_1d_input_even_without_factors():
    with pytest.raises(ValueError, match="2次元以上"):
        core_from_factors(torch.randn(5), {})
