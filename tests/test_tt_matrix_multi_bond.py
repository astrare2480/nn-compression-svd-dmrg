"""複数 bond の順次切り詰め。CPU の小さい Tensor だけを使う。"""

from __future__ import annotations

import torch

from nn_compression.tt_mps import (
    dense_to_tt_matrix_cores,
    truncate_tt_matrix_bond,
    truncate_tt_matrix_bonds,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)

OUT_MODES = (2, 2, 2)
IN_MODES = (2, 2, 2)


def _cores(requires_grad=False):
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    if requires_grad:
        weight.requires_grad_()
    cores = dense_to_tt_matrix_cores(weight, OUT_MODES, IN_MODES, max_rank=4)
    return weight, cores


def test_sequential_truncation_matches_repeated_single_bond_calls():
    _weight, cores = _cores()
    before = [core.detach().clone() for core in cores]
    updated, splits = truncate_tt_matrix_bonds(
        cores,
        OUT_MODES,
        IN_MODES,
        ((0, 2), (1, 1)),
    )
    manual, first = truncate_tt_matrix_bond(cores, OUT_MODES, IN_MODES, 0, max_bond=2)
    manual, second = truncate_tt_matrix_bond(manual, OUT_MODES, IN_MODES, 1, max_bond=1)
    for core, original in zip(cores, before):
        assert torch.equal(core, original)
    assert len(splits) == 2
    for left, right in zip(updated, manual):
        assert torch.equal(left, right)
    assert splits[0].retained_rank == first.retained_rank
    assert splits[1].retained_rank == second.retained_rank
    assert tt_matrix_ranks(updated)[1] <= 2
    assert tt_matrix_ranks(updated)[2] <= 1
    assert tt_matrix_to_dense(updated, OUT_MODES, IN_MODES).dtype == torch.float64


def test_bond_order_can_change_the_result():
    _weight, cores = _cores()
    forward, _forward_splits = truncate_tt_matrix_bonds(cores, OUT_MODES, IN_MODES, ((0, 1), (1, 1)))
    reverse, _reverse_splits = truncate_tt_matrix_bonds(cores, OUT_MODES, IN_MODES, ((1, 1), (0, 1)))
    forward_dense = tt_matrix_to_dense(forward, OUT_MODES, IN_MODES)
    reverse_dense = tt_matrix_to_dense(reverse, OUT_MODES, IN_MODES)
    assert not torch.allclose(forward_dense, reverse_dense)


def test_truncation_preserves_autograd_and_leaves_inputs_attached():
    weight, cores = _cores(requires_grad=True)
    updated, _splits = truncate_tt_matrix_bonds(cores, OUT_MODES, IN_MODES, ((1, 2),))
    updated[1].sum().backward()
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()
