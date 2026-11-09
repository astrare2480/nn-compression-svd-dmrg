"""2 本の長い ALS の事後判定（03 の設計 §9）の検算。"""

from __future__ import annotations

import itertools
import math

import pytest

from nn_compression.metrics.als_convergence import (
    ALS_STOP_REASONS,
    classify_als_initialization_dependence,
    relative_objective_difference,
)

TOLERANCE = 1e-3


def _classify(a, b, qa=100.0, qb=100.0):
    return classify_als_initialization_dependence(a, b, qa, qb, equivalence_tolerance=TOLERANCE)


def test_equivalent_objective():
    result = _classify("converged", "converged", 100.0, 100.05)
    assert result.verdict == "converged_equivalent_objective"
    assert result.delta_q == pytest.approx(0.05 / 100.0)
    assert result.equivalence_tolerance == TOLERANCE


def test_initialization_dependent_including_the_boundary():
    assert _classify("converged", "converged", 100.0, 106.0).verdict == "converged_initialization_dependent"
    # δ_Q = 1e-3 ちょうどは「同等」に入らない（δ_Q < 0.1% が同等）。
    assert _classify("converged", "converged", 1000.0, 1001.0).verdict == "converged_initialization_dependent"


def test_max_sweeps_is_unresolved():
    assert _classify("max_sweeps", "converged").verdict == "max_sweeps_unresolved"
    assert _classify("max_sweeps", "max_sweeps").verdict == "max_sweeps_unresolved"


def test_local_solver_unresolved_takes_precedence():
    assert _classify("local_solver_unresolved", "max_sweeps").verdict == "local_solver_unresolved"
    assert _classify("converged", "local_solver_unresolved").verdict == "local_solver_unresolved"


@pytest.mark.parametrize("a, b", list(itertools.product(ALS_STOP_REASONS, repeat=2)))
def test_swapping_the_two_runs_gives_the_same_verdict(a, b):
    left = _classify(a, b, 100.0, 103.0)
    right = _classify(b, a, 103.0, 100.0)
    assert left == right
    assert relative_objective_difference(100.0, 103.0) == relative_objective_difference(103.0, 100.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_non_positive_or_non_finite_objectives_are_refused(bad):
    with pytest.raises(ValueError):
        _classify("converged", "converged", bad, 100.0)
    with pytest.raises(ValueError):
        _classify("converged", "converged", 100.0, bad)


def test_unknown_reasons_and_bad_tolerances_are_refused():
    with pytest.raises(ValueError):
        _classify("sweep_tolerance", "converged")
    with pytest.raises(ValueError):
        classify_als_initialization_dependence("converged", "converged", 1.0, 1.0, equivalence_tolerance=0.0)
    with pytest.raises(ValueError):
        classify_als_initialization_dependence("converged", "converged", 1.0, 1.0, equivalence_tolerance=math.nan)
    with pytest.raises(TypeError):
        classify_als_initialization_dependence("converged", "converged", True, 1.0, equivalence_tolerance=1e-3)
