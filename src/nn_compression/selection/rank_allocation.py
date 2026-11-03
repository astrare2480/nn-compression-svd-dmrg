"""TT bond rank の割当候補を列挙し、予算内で score 最小のものを選ぶ。

検証 loss や accuracy は見ない。同点は cost、その次は rank 列そのもので
決めるので、候補を渡す順番では結果が変わらない。
"""

from __future__ import annotations

import itertools
import operator
from collections.abc import Sequence
from dataclasses import dataclass

@dataclass(frozen=True)
class RankAllocationSelection:
    """予算内で選んだ rank 列と、その cost・score。"""

    ranks: tuple[int, ...]
    cost: float
    score: float


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} は bool 以外の正の整数である必要があります: {value!r}")
    try:
        number = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} は正の整数である必要があります: {value!r}") from exc
    if number < 1:
        raise ValueError(f"{name} は正の整数である必要があります: {number}")
    return number


def _as_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} は列である必要があります: {type(value)!r}")
    return value


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} は有限の実数である必要があります: {value!r}")
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} は有限の実数である必要があります: {value!r}")
    return number


def enumerate_bond_rank_allocations(
    base_ranks: Sequence[int],
    bond_indices: Sequence[int],
    rank_choices: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """基準 rank 列の指定 bond だけを、候補の直積で置き換えた列を返す。

    ``bond_indices[i]`` の候補は ``rank_choices[i]`` である。並びは bond の
    指定順に ``itertools.product`` した順で、同じ rank 列は最初の 1 つだけ残す。
    基準値、bond、候補の具体的な数値は引数で渡す。入力は変更しない。
    """
    base = tuple(_positive_int(rank, "base_ranks") for rank in _as_sequence(base_ranks, "base_ranks"))
    if not base:
        raise ValueError("基準 rank 列が空です。")
    bonds = []
    for bond in _as_sequence(bond_indices, "bond_indices"):
        if isinstance(bond, bool):
            raise TypeError(f"bond index は bool 以外の整数である必要があります: {bond!r}")
        try:
            index = operator.index(bond)
        except TypeError as exc:
            raise TypeError(f"bond index は整数である必要があります: {bond!r}") from exc
        if not 0 <= index < len(base):
            raise ValueError(f"bond index が rank 列の範囲外です: {index}")
        bonds.append(index)
    if len(set(bonds)) != len(bonds):
        raise ValueError(f"bond index が重複しています: {tuple(bonds)}")
    choices = _as_sequence(rank_choices, "rank_choices")
    if len(choices) != len(bonds):
        raise ValueError(
            f"rank 候補の数が bond 数と一致しません: choices={len(choices)} bonds={len(bonds)}"
        )
    parsed: list[tuple[int, ...]] = []
    for position, choice in enumerate(choices):
        values = _as_sequence(choice, f"rank_choices[{position}]")
        if len(values) == 0:
            raise ValueError(f"rank_choices[{position}] が空です。")
        parsed.append(tuple(_positive_int(rank, f"rank_choices[{position}]") for rank in values))
    found: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for combo in itertools.product(*parsed):
        ranks = list(base)
        for bond, rank in zip(bonds, combo, strict=True):
            ranks[bond] = rank
        item = tuple(ranks)
        if item not in seen:
            seen.add(item)
            found.append(item)
    return tuple(found)


def select_best_rank_allocation_under_budget(
    candidates: Sequence[tuple[Sequence[int], float, float]],
    budget: float,
) -> RankAllocationSelection:
    """``cost <= budget`` の候補から、score 最小、次に cost 最小、次に rank 列の昇順を選ぶ。

    候補は ``(ranks, cost, score)`` である。同じ key の候補が複数あっても、
    先に並んでいた方を優先しない。検証指標は受け取らない。
    """
    rows = _as_sequence(candidates, "candidates")
    limit = _finite_real(budget, "budget")
    feasible: list[tuple[tuple[float, float, tuple[int, ...]], tuple[int, ...], float, float]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"candidates[{position}] は (ranks, cost, score) である必要があります: {row!r}")
        ranks, cost, score = row
        parsed = tuple(_positive_int(rank, f"candidates[{position}] ranks") for rank in _as_sequence(ranks, "ranks"))
        if not parsed:
            raise ValueError(f"candidates[{position}] の rank 列が空です。")
        parsed_cost = _finite_real(cost, f"candidates[{position}] cost")
        parsed_score = _finite_real(score, f"candidates[{position}] score")
        if parsed_cost <= limit:
            feasible.append(((parsed_score, parsed_cost, parsed), parsed, parsed_cost, parsed_score))
    if not feasible:
        raise ValueError("budget 以下の候補がありません。")
    _key, ranks, cost, score = min(feasible, key=lambda item: item[0])
    return RankAllocationSelection(ranks=ranks, cost=cost, score=score)
