"""rank 割当の列挙と予算内選択。数値はテスト内だけで決める。"""

from __future__ import annotations

import pytest

from nn_compression.selection import (
    RankAllocationSelection,
    enumerate_bond_rank_allocations,
    select_best_rank_allocation_under_budget,
)


def test_enumeration_is_deterministic_and_drops_duplicate_tuples():
    base = [1, 4, 4, 1]
    bonds = [1, 2]
    choices = ([2, 2, 3], [3, 2])
    found = enumerate_bond_rank_allocations(base, bonds, choices)
    assert found == (
        (1, 2, 3, 1),
        (1, 2, 2, 1),
        (1, 3, 3, 1),
        (1, 3, 2, 1),
    )
    assert base == [1, 4, 4, 1]
    assert bonds == [1, 2]
    assert choices[0] == [2, 2, 3]


def test_enumeration_rejects_invalid_bonds_and_choices():
    with pytest.raises(TypeError):
        enumerate_bond_rank_allocations((1, 2, 1), (True,), ((1,),))
    with pytest.raises(ValueError, match="範囲外"):
        enumerate_bond_rank_allocations((1, 2, 1), (3,), ((1,),))
    with pytest.raises(ValueError, match="重複"):
        enumerate_bond_rank_allocations((1, 2, 2, 1), (1, 1), ((2,), (3,)))
    with pytest.raises(ValueError, match="空"):
        enumerate_bond_rank_allocations((1, 2, 1), (1,), ((),))
    with pytest.raises(ValueError, match="正の整数"):
        enumerate_bond_rank_allocations((1, 0, 1), (1,), ((1,),))


def test_budget_selection_uses_score_then_cost_then_rank_tuple():
    candidates = [
        ((1, 3, 1), 4.0, 1.0),
        ((1, 9, 1), 9.0, 0.0),
        ((1, 2, 1), 4.0, 1.0),
        ((1, 4, 1), 3.0, 2.0),
    ]
    reversed_rows = list(reversed(candidates))
    selected = select_best_rank_allocation_under_budget(candidates, 5)
    assert selected == select_best_rank_allocation_under_budget(reversed_rows, 5)
    assert isinstance(selected, RankAllocationSelection)
    assert selected.ranks == (1, 2, 1)
    assert selected.cost == 4.0
    assert selected.score == 1.0
    assert candidates[0] == ((1, 3, 1), 4.0, 1.0)


def test_budget_includes_exact_cost_and_rejects_an_empty_feasible_set():
    selected = select_best_rank_allocation_under_budget(
        [((1, 2), 5.0, 3.0), ((1, 1), 5.0, 3.0)],
        5,
    )
    assert selected.ranks == (1, 1)
    with pytest.raises(ValueError, match="budget 以下"):
        select_best_rank_allocation_under_budget([((1, 2), 5.1, 0.0)], 5)
