"""TTConv の直接縮約 MACs と、二乗和からの relative L2。"""

import math

import pytest
import torch

from nn_compression.metrics import (
    relative_l2_from_squared_sums,
    tt_conv2d_direct_contraction_macs,
    tt_linear_direct_contraction_macs,
)


OUT_MODES = (4, 4, 4)
IN_MODES = (6, 6, 8)


def test_spatial_product_matches_tt_linear_site_sum():
    ranks = (1, 16, 20, 1)
    site = sum(tt_linear_direct_contraction_macs(OUT_MODES, IN_MODES, ranks))
    assert tt_conv2d_direct_contraction_macs(OUT_MODES, IN_MODES, ranks, 14, 14) == 14 * 14 * site


def test_rank_position_changes_the_conv_cost():
    left = tt_conv2d_direct_contraction_macs(OUT_MODES, IN_MODES, (1, 12, 16, 1), 14, 14)
    right = tt_conv2d_direct_contraction_macs(OUT_MODES, IN_MODES, (1, 16, 12, 1), 14, 14)
    assert left != right
    assert left == 33_216_512
    assert right == 33_718_272


def test_rejects_non_positive_spatial_size():
    with pytest.raises(ValueError):
        tt_conv2d_direct_contraction_macs(OUT_MODES, IN_MODES, (1, 12, 12, 1), 0, 14)
    with pytest.raises(TypeError):
        tt_conv2d_direct_contraction_macs(OUT_MODES, IN_MODES, (1, 12, 12, 1), True, 14)


def test_relative_l2_matches_vector_norm_ratio():
    baseline = torch.tensor([[3.0, 4.0], [0.0, 12.0]])
    difference = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    expected = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(baseline)
    relative = relative_l2_from_squared_sums(
        float(torch.sum(difference * difference)),
        float(torch.sum(baseline * baseline)),
    )
    assert relative == pytest.approx(float(expected))
    assert relative == pytest.approx(5.0 / 13.0)


def test_relative_l2_rejects_zero_or_negative_denominator_without_clipping():
    with pytest.raises(ValueError, match="ゼロ"):
        relative_l2_from_squared_sums(1.0, 0.0)
    with pytest.raises(ValueError):
        relative_l2_from_squared_sums(-1.0, 4.0)
    with pytest.raises(TypeError):
        relative_l2_from_squared_sums(True, 4.0)
    assert math.sqrt(4.0) / math.sqrt(16.0) == relative_l2_from_squared_sums(4.0, 16.0)
