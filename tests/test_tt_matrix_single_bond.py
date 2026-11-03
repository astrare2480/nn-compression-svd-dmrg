"""1 bond の TT-matrix 切り詰め。CPU の小さい例だけを使う。"""

from __future__ import annotations

import pytest
import torch

from nn_compression.tt_mps import (
    dense_to_tt_matrix_cores,
    truncate_tt_matrix_bond,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)

OUT_MODES = (2, 2, 2)
IN_MODES = (2, 2, 2)


def _cores(max_rank=4, requires_grad=False):
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    if requires_grad:
        weight.requires_grad_()
    return weight, dense_to_tt_matrix_cores(weight, OUT_MODES, IN_MODES, max_rank=max_rank)


def test_truncate_one_bond_keeps_inputs_and_limits_rank():
    _weight, cores = _cores()
    before = [core.detach().clone() for core in cores]
    identities = [core.data_ptr() for core in cores]
    updated, split = truncate_tt_matrix_bond(
        cores,
        OUT_MODES,
        IN_MODES,
        1,
        max_bond=2,
        direction="right",
    )
    assert [core.data_ptr() for core in cores] == identities
    for core, original in zip(cores, before):
        assert torch.equal(core, original)
    assert updated[1].dtype == cores[1].dtype
    assert updated[1].device == cores[1].device
    assert int(tt_matrix_ranks(updated)[2]) == split.retained_rank
    assert split.retained_rank <= 2
    assert updated[1].shape[3] == split.retained_rank
    assert updated[2].shape[0] == split.retained_rank
    assert updated[0].data_ptr() != cores[0].data_ptr() or torch.equal(updated[0], cores[0])
    dense = tt_matrix_to_dense(updated, OUT_MODES, IN_MODES)
    assert dense.shape == (8, 8)
    assert dense.dtype == torch.float64


def test_truncate_preserves_autograd():
    weight, cores = _cores(requires_grad=True)
    updated, _split = truncate_tt_matrix_bond(cores, OUT_MODES, IN_MODES, 0, max_bond=2)
    updated[0].sum().backward()
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()


@pytest.mark.parametrize("pair_index", [-1, 2, 3])
def test_truncate_rejects_pair_outside_the_chain(pair_index):
    _weight, cores = _cores()
    with pytest.raises(ValueError, match="pair_index"):
        truncate_tt_matrix_bond(cores, OUT_MODES, IN_MODES, pair_index, max_bond=2)


def test_truncate_rejects_bad_rank_cap_without_mutating_cores():
    _weight, cores = _cores()
    before = [core.detach().clone() for core in cores]
    with pytest.raises(ValueError):
        truncate_tt_matrix_bond(cores, OUT_MODES, IN_MODES, 0, max_bond=0)
    for core, original in zip(cores, before):
        assert torch.equal(core, original)
