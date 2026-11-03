"""TTLinear の直接縮約 MACs。"""

import pytest
import torch
from torch import nn

from nn_compression.metrics import (
    linear_macs,
    tt_linear_direct_contraction_macs,
)
from nn_compression.models import MNISTMLP


OUT_MODES = (4, 2, 2, 2, 4, 4)
IN_MODES = (2, 2, 2, 2, 7, 7)
RANKS = (1, 8, 12, 12, 12, 12, 1)


def test_current_six_core_setting_matches_expected_site_costs():
    costs = tt_linear_direct_contraction_macs(OUT_MODES, IN_MODES, RANKS)
    assert costs == (25088, 301056, 451584, 451584, 903168, 43008)
    assert sum(costs) == 2_175_488


def test_one_core_boundary_matches_dense_product():
    costs = tt_linear_direct_contraction_macs((8,), (4,), (1, 1))
    assert costs == (32,)


def test_rejects_bad_mode_product_rank_length_and_boundary():
    with pytest.raises(ValueError, match="長さ"):
        tt_linear_direct_contraction_macs((4, 2), (2,), (1, 1, 1))
    with pytest.raises(ValueError, match="ranks"):
        tt_linear_direct_contraction_macs((4, 2), (2, 2), (1, 1))
    with pytest.raises(ValueError, match="境界"):
        tt_linear_direct_contraction_macs((4,), (2,), (2, 1))
    with pytest.raises(ValueError, match="out_modes"):
        tt_linear_direct_contraction_macs((), (2,), (1, 1))


def test_rejects_zero_or_negative_mode_or_rank():
    with pytest.raises(ValueError):
        tt_linear_direct_contraction_macs((0,), (2,), (1, 1))
    with pytest.raises(ValueError):
        tt_linear_direct_contraction_macs((4,), (-1,), (1, 1))
    with pytest.raises(ValueError):
        tt_linear_direct_contraction_macs((4,), (2,), (1, 0))
    with pytest.raises(TypeError):
        tt_linear_direct_contraction_macs((True,), (2,), (1, 1))


def test_actual_ranks_change_the_cost():
    reduced = (1, 8, 4, 12, 12, 12, 1)
    full = sum(tt_linear_direct_contraction_macs(OUT_MODES, IN_MODES, RANKS))
    changed = sum(tt_linear_direct_contraction_macs(OUT_MODES, IN_MODES, reduced))
    assert changed != full
    assert changed > 0


def test_sums_with_dense_linear_macs():
    model = MNISTMLP()
    fc1 = sum(tt_linear_direct_contraction_macs(OUT_MODES, IN_MODES, RANKS))
    total = fc1 + linear_macs(model.fc2) + linear_macs(model.fc3)
    dense = linear_macs(model.fc1) + linear_macs(model.fc2) + linear_macs(model.fc3)
    assert linear_macs(model.fc1) == 401_408
    assert dense == 535_040
    assert total == 2_309_120
    assert isinstance(model.fc2, nn.Linear)
    assert torch.float32 == torch.float32
