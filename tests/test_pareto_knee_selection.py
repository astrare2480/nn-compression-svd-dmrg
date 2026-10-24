"""``selection.pareto_knee.select_pareto_knee`` のテスト。"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from nn_compression.selection import (
    ParetoKneeSelection,
    extract_pareto_frontier,
    select_pareto_knee,
)
from nn_compression.selection import pareto_knee as pareto_knee_module

X, Y = "parameters", "loss"


def _frame(points, index=None, **extra):
    frame = pd.DataFrame({X: [p[0] for p in points], Y: [p[1] for p in points]}, index=index)
    for name, values in extra.items():
        frame[name] = values
    return frame


# x 昇順で y が単調減少し、(2, 4) が明確な内部 knee になる frontier。
CLEAR = [(1, 10.0), (2, 4.0), (3, 2.2), (4, 2.0), (5, 1.9)]
# 他点に支配される点（frontier に入らない）。
DOMINATED = [(2.5, 9.0), (4.5, 5.0), (6, 12.0)]


def _clear_frame():
    return _frame(CLEAR + DOMINATED, index=list("abcdefgh"))


def _knee_coordinates(result: ParetoKneeSelection, frame: pd.DataFrame):
    row = frame.loc[result.knee_index]
    return (float(row[X]), float(row[Y]))


# ---------------------------------------------------------------------------
# 明確な knee
# ---------------------------------------------------------------------------


def test_clear_interior_knee_is_selected_with_original_index_and_row():
    frame = _clear_frame()

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    assert isinstance(result, ParetoKneeSelection)
    assert result.is_defined is True
    assert result.undefined_reason is None
    assert result.knee_index == "b"
    assert result.knee_row[X] == 2 and result.knee_row[Y] == 4.0
    assert result.knee_row.name == "b"
    assert result.knee_row.equals(frame.loc["b"])


def test_knee_distance_is_the_distance_in_unit_square_coordinates():
    frame = _clear_frame()

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    xs = np.array([p[0] for p in CLEAR], dtype=float)
    ys = np.array([p[1] for p in CLEAR], dtype=float)
    nx = (xs - xs.min()) / (xs.max() - xs.min())
    ny = (ys - ys.min()) / (ys.max() - ys.min())
    expected = (1.0 - nx - ny) / math.sqrt(2.0)
    assert result.knee_distance == pytest.approx(float(expected[1]), rel=1e-12)
    assert result.knee_distance == pytest.approx(float(expected[1:-1].max()), rel=1e-12)
    assert type(result.knee_distance) is float


def test_frontier_matches_existing_extract_pareto_frontier_and_keeps_original_dtypes():
    frame = _clear_frame()
    frame[X] = (frame[X] * 2).astype("int64")  # 整数 dtype の列でも frontier で dtype が保たれる

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    expected = extract_pareto_frontier(frame, X, Y)
    pd.testing.assert_frame_equal(result.frontier, expected, check_dtype=True)
    assert result.frontier[X].dtype == np.dtype("int64")
    assert list(result.frontier.index) == list(range(5))
    assert result.frontier[X].is_monotonic_increasing


def test_frontier_source_index_points_to_the_input_rows():
    frame = _clear_frame()

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    assert result.frontier_source_index == ("a", "b", "c", "d", "e")
    assert isinstance(result.frontier_source_index, tuple)
    restored = frame.loc[list(result.frontier_source_index)].reset_index(drop=True)
    pd.testing.assert_frame_equal(result.frontier, restored)


def test_extra_columns_are_kept_in_frontier_and_knee_row():
    frame = _frame(CLEAR + DOMINATED, rank=list(range(8)), label=list("abcdefgh"))

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    assert list(result.frontier.columns) == [X, Y, "rank", "label"]
    assert result.knee_row["rank"] == 1 and result.knee_row["label"] == "b"


def test_integer_default_index_is_returned_as_label():
    frame = _frame(CLEAR)

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    assert result.knee_index == 1


def test_non_default_and_duplicate_index_labels_are_mapped_correctly():
    frame = _frame(CLEAR, index=[7, 7, 7, 7, 7])

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    assert result.is_defined
    assert result.knee_index == 7
    assert (result.knee_row[X], result.knee_row[Y]) == (2, 4.0)
    assert result.frontier_source_index == (7, 7, 7, 7, 7)


def test_selected_knee_is_never_a_frontier_endpoint():
    rng = np.random.default_rng(0)
    defined = 0
    for _ in range(200):
        n = int(rng.integers(3, 14))
        frame = _frame(list(zip(rng.random(n), rng.random(n))))
        result = select_pareto_knee(frame, x_column=X, y_column=Y)
        if result.is_defined:
            defined += 1
            assert result.knee_index != result.frontier_source_index[0]
            assert result.knee_index != result.frontier_source_index[-1]
            assert result.knee_distance > 0.0
    assert defined > 0


# ---------------------------------------------------------------------------
# 未定義
# ---------------------------------------------------------------------------


def _assert_undefined(result, reason, frame=None):
    assert isinstance(result, ParetoKneeSelection)
    assert result.is_defined is False
    assert result.undefined_reason == reason
    assert result.knee_index is None
    assert result.knee_row is None
    assert result.knee_distance is None
    assert isinstance(result.frontier, pd.DataFrame) and len(result.frontier) >= 1
    assert len(result.frontier_source_index) == len(result.frontier)


def test_single_point_is_undefined():
    result = select_pareto_knee(_frame([(3, 1.0)]), x_column=X, y_column=Y)

    _assert_undefined(result, "too_few_points")
    assert len(result.frontier) == 1


def test_two_points_are_undefined():
    result = select_pareto_knee(_frame([(1, 5.0), (2, 1.0)]), x_column=X, y_column=Y)

    _assert_undefined(result, "too_few_points")
    assert len(result.frontier) == 2


def test_frontier_with_two_points_is_undefined_even_when_many_rows_are_dominated():
    frame = _frame([(1, 5.0), (2, 1.0), (3, 6.0), (4, 7.0), (5, 8.0)])

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    _assert_undefined(result, "too_few_points")
    assert result.frontier_source_index == (0, 1)


def test_identical_rows_count_as_one_point():
    result = select_pareto_knee(_frame([(2, 2.0)] * 4), x_column=X, y_column=Y)

    _assert_undefined(result, "too_few_points")


def test_zero_x_width_is_undefined():
    # x がすべて同じだと frontier は最小の y の1点になる。点数を先に見るので too_few_points。
    result = select_pareto_knee(_frame([(5, 3.0), (5, 2.0), (5, 1.0)]), x_column=X, y_column=Y)

    _assert_undefined(result, "too_few_points")
    assert len(result.frontier) == 1


def test_zero_y_width_is_undefined():
    # y がすべて同じだと frontier は最小の x の1点になる。
    result = select_pareto_knee(_frame([(1, 2.0), (2, 2.0), (3, 2.0)]), x_column=X, y_column=Y)

    _assert_undefined(result, "too_few_points")
    assert len(result.frontier) == 1


def test_all_collinear_points_are_undefined():
    result = select_pareto_knee(_frame([(1, 4.0), (2, 3.0), (3, 2.0), (4, 1.0)]), x_column=X, y_column=Y)

    _assert_undefined(result, "nearly_collinear")
    assert len(result.frontier) == 4


def test_collinear_points_with_unequal_axis_scales_are_undefined():
    points = [(10 * k, 0.001 * (9 - k)) for k in range(10)]

    result = select_pareto_knee(_frame(points), x_column=X, y_column=Y)

    _assert_undefined(result, "nearly_collinear")


def test_tolerance_controls_the_nearly_collinear_decision():
    # 正規化後の内部点が直線から約 7e-7 だけ離れている。
    frame = _frame([(0, 10.0), (5, 5.0 - 1e-5), (10, 0.0)])

    assert select_pareto_knee(frame, x_column=X, y_column=Y).is_defined is True
    _assert_undefined(select_pareto_knee(frame, x_column=X, y_column=Y, tolerance=1e-3), "nearly_collinear")
    assert select_pareto_knee(frame, x_column=X, y_column=Y, tolerance=0.0).is_defined is True


def test_tied_maximum_distance_is_undefined():
    # 正規化後 (0,1), (0.2,0.6), (0.6,0.2), (1,0)。内部2点は直線から同じ距離。
    frame = _frame([(0, 10.0), (2, 6.0), (6, 2.0), (10, 0.0)])

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    _assert_undefined(result, "ambiguous_tie")


def test_near_tie_within_tolerance_is_undefined_and_outside_is_defined():
    close = _frame([(0, 10.0), (2, 6.0), (6, 2.0 + 1e-6), (10, 0.0)])
    apart = _frame([(0, 10.0), (2, 6.0), (6, 2.5), (10, 0.0)])

    _assert_undefined(select_pareto_knee(close, x_column=X, y_column=Y, tolerance=1e-3), "ambiguous_tie")
    assert select_pareto_knee(close, x_column=X, y_column=Y).is_defined is True
    assert select_pareto_knee(apart, x_column=X, y_column=Y, tolerance=1e-3).is_defined is True


def test_duplicated_knee_rows_are_not_a_tie_and_first_row_is_returned():
    frame = _frame(CLEAR + [(2, 4.0)], index=list("abcdef"))

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    assert result.is_defined
    assert result.knee_index == "b"
    assert len(result.frontier) == 6  # 同一点は互いに支配しないので frontier には両方残る


def test_concave_frontier_has_no_knee_on_the_undesirable_side():
    # 内部点 (0.5, 0.9) は直線 x + y = 1 より原点から遠い側。
    result = select_pareto_knee(_frame([(0, 10.0), (5, 9.0), (10, 0.0)]), x_column=X, y_column=Y)

    _assert_undefined(result, "undesirable_side")
    assert len(result.frontier) == 3


def test_farthest_point_on_the_undesirable_side_is_not_replaced_by_a_nearer_good_side_point():
    # 正規化後: A=(0.2,0.95) は悪い側で距離 ≈ 0.106、B=(0.6,0.3) は良い側で距離 ≈ 0.071。
    frame = _frame([(0, 1.0), (2, 0.95), (6, 0.3), (10, 0.0)])

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    _assert_undefined(result, "undesirable_side")


def test_dominated_undesirable_points_do_not_affect_the_decision():
    frame = _frame(CLEAR + [(1.5, 11.0), (2, 9.9), (3, 9.0)])

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    assert result.is_defined and result.knee_index == 1


def test_undefined_reason_values_are_the_documented_strings():
    reasons = {
        select_pareto_knee(_frame([(1, 1.0)]), x_column=X, y_column=Y).undefined_reason,
        select_pareto_knee(_frame([(5, 3.0), (5, 1.0), (5, 2.0)]), x_column=X, y_column=Y).undefined_reason,
        select_pareto_knee(_frame([(1, 2.0), (2, 2.0), (3, 2.0)]), x_column=X, y_column=Y).undefined_reason,
        select_pareto_knee(_frame([(1, 3.0), (2, 2.0), (3, 1.0)]), x_column=X, y_column=Y).undefined_reason,
        select_pareto_knee(_frame([(0, 10.0), (5, 9.0), (10, 0.0)]), x_column=X, y_column=Y).undefined_reason,
        select_pareto_knee(_frame([(0, 10.0), (2, 6.0), (6, 2.0), (10, 0.0)]), x_column=X, y_column=Y).undefined_reason,
    }
    # x または y が入力全体で同じケースは frontier が1点になり、too_few_points になる。
    assert reasons == {
        "too_few_points",
        "nearly_collinear",
        "undesirable_side",
        "ambiguous_tie",
    }


# ---------------------------------------------------------------------------
# 入力検証
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("column", [X, Y])
def test_non_finite_values_are_rejected(column, bad):
    frame = _clear_frame()
    frame[column] = frame[column].astype(float)
    frame.loc["g", column] = bad  # frontier に入らない行でも拒否する

    with pytest.raises(ValueError, match="NaN|inf"):
        select_pareto_knee(frame, x_column=X, y_column=Y)


def test_missing_values_and_non_numeric_values_are_rejected():
    frame = _clear_frame()
    frame.loc["a", Y] = None
    with pytest.raises(ValueError, match="NaN"):
        select_pareto_knee(frame, x_column=X, y_column=Y)
    frame = _clear_frame().astype({Y: object})
    frame.loc["a", Y] = "bad"
    with pytest.raises(ValueError, match="NaN"):
        select_pareto_knee(frame, x_column=X, y_column=Y)


def test_empty_frame_is_rejected():
    with pytest.raises(ValueError, match="空"):
        select_pareto_knee(pd.DataFrame({X: [], Y: []}), x_column=X, y_column=Y)


def test_missing_column_is_rejected():
    with pytest.raises(KeyError):
        select_pareto_knee(_clear_frame(), x_column="nope", y_column=Y)
    with pytest.raises(KeyError):
        select_pareto_knee(_clear_frame(), x_column=X, y_column="nope")


def test_same_column_for_both_axes_is_rejected():
    with pytest.raises(ValueError, match="異なる列"):
        select_pareto_knee(_clear_frame(), x_column=X, y_column=X)


@pytest.mark.parametrize("bad", [[(1, 1.0)], {X: [1], Y: [1]}, np.zeros((3, 2)), None])
def test_non_dataframe_input_is_rejected(bad):
    with pytest.raises(TypeError, match="DataFrame"):
        select_pareto_knee(bad, x_column=X, y_column=Y)


@pytest.mark.parametrize("bad", [-1e-9, math.nan, math.inf, -math.inf])
def test_invalid_tolerance_values_are_rejected(bad):
    with pytest.raises(ValueError, match="tolerance"):
        select_pareto_knee(_clear_frame(), x_column=X, y_column=Y, tolerance=bad)


@pytest.mark.parametrize("bad", [True, False, "1e-9", None, [1e-9]])
def test_non_real_or_bool_tolerance_is_rejected(bad):
    with pytest.raises(TypeError, match="tolerance"):
        select_pareto_knee(_clear_frame(), x_column=X, y_column=Y, tolerance=bad)


def test_x_column_and_y_column_are_keyword_only():
    with pytest.raises(TypeError):
        select_pareto_knee(_clear_frame(), X, Y)


# ---------------------------------------------------------------------------
# 非破壊・順序不変・既存関数の再利用
# ---------------------------------------------------------------------------


def test_input_dataframe_is_not_modified():
    frame = _frame(CLEAR + DOMINATED, index=list("abcdefgh"), note=list("ABCDEFGH"))
    frame[X] = frame[X].astype("float32")
    snapshot = frame.copy(deep=True)

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)
    assert list(frame.columns) == [X, Y, "note"]
    assert frame[X].dtype == np.float32
    # 戻り値を書き換えても入力には影響しない。
    result.frontier.loc[0, Y] = -1.0
    result.knee_row[Y] = -1.0
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)


def test_input_dataframe_is_not_modified_when_result_is_undefined():
    frame = _frame([(1, 3.0), (2, 2.0), (3, 1.0)])
    snapshot = frame.copy(deep=True)

    select_pareto_knee(frame, x_column=X, y_column=Y)

    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)


def test_integer_columns_are_accepted_and_stay_integer_in_outputs():
    frame = _frame([(10, 100), (20, 40), (30, 22), (40, 20), (50, 19)])

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    assert result.is_defined and result.knee_index == 1
    assert result.frontier[Y].dtype == np.dtype("int64")


def test_existing_column_named_like_the_internal_position_column_is_safe():
    frame = _frame(CLEAR + DOMINATED, __pareto_source_position__=list(range(100, 108)))
    snapshot = frame.copy(deep=True)

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    assert result.is_defined and result.knee_index == 1
    assert list(result.frontier["__pareto_source_position__"]) == [100, 101, 102, 103, 104]
    pd.testing.assert_frame_equal(frame, snapshot)


@pytest.mark.parametrize(
    "points",
    [
        CLEAR + DOMINATED,
        [(0, 10.0), (2, 6.0), (6, 2.0), (10, 0.0)],  # tie
        [(0, 10.0), (5, 9.0), (10, 0.0)],  # undesirable side
        [(1, 4.0), (2, 3.0), (3, 2.0), (4, 1.0)],  # collinear
        [(1, 1.0), (2, 5.0), (3, 6.0)],  # 1 点 frontier
    ],
)
def test_row_order_does_not_change_frontier_or_knee(points):
    labels = [f"row{i}" for i in range(len(points))]
    frame = _frame(points, index=labels)
    baseline = select_pareto_knee(frame, x_column=X, y_column=Y)
    rng = np.random.default_rng(1)

    for _ in range(25):
        permuted = frame.iloc[rng.permutation(len(frame))]
        result = select_pareto_knee(permuted, x_column=X, y_column=Y)

        pd.testing.assert_frame_equal(result.frontier, baseline.frontier)
        assert result.frontier_source_index == baseline.frontier_source_index
        assert result.is_defined == baseline.is_defined
        assert result.undefined_reason == baseline.undefined_reason
        assert result.knee_index == baseline.knee_index
        assert result.knee_distance == baseline.knee_distance


def test_knee_is_invariant_to_scaling_and_shifting_each_axis():
    base = select_pareto_knee(_frame(CLEAR), x_column=X, y_column=Y)

    transformed = _frame([(x * 1e6 + 5e6, y * 1e-9 + 3.0) for x, y in CLEAR])
    result = select_pareto_knee(transformed, x_column=X, y_column=Y)
    huge = select_pareto_knee(_frame([(x * 1e290, y * 1e-290) for x, y in CLEAR]), x_column=X, y_column=Y)

    assert result.knee_index == base.knee_index
    assert result.knee_distance == pytest.approx(base.knee_distance, rel=1e-6)
    assert huge.knee_index == base.knee_index
    assert huge.knee_distance == pytest.approx(base.knee_distance, rel=1e-9)


def test_existing_selection_functions_are_reused(monkeypatch):
    calls = {"frontier": 0, "endpoints": 0, "line": 0, "knee": 0}
    originals = {
        name: getattr(pareto_knee_module, name)
        for name in ("extract_pareto_frontier", "get_endpoints", "line_equation", "find_knee_point_numpy")
    }

    def wrap(key, function):
        def wrapper(*args, **kwargs):
            calls[key] += 1
            return function(*args, **kwargs)

        return wrapper

    for key, name in zip(calls, originals):
        monkeypatch.setattr(pareto_knee_module, name, wrap(key, originals[name]))

    result = select_pareto_knee(_clear_frame(), x_column=X, y_column=Y)

    assert result.is_defined
    assert calls["frontier"] == 1 and calls["endpoints"] == 1 and calls["line"] == 1
    assert calls["knee"] >= 1


def test_result_is_a_frozen_dataclass_with_expected_fields():
    result = select_pareto_knee(_clear_frame(), x_column=X, y_column=Y)

    assert [field.name for field in dataclasses.fields(result)] == [
        "frontier",
        "frontier_source_index",
        "is_defined",
        "knee_index",
        "knee_row",
        "knee_distance",
        "undefined_reason",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.is_defined = False
    assert result == result  # DataFrame を含んでも比較でエラーにならない（同一性比較）


# ---------------------------------------------------------------------------
# 幅と点数の判定は Pareto frontier 上で行う（P2-5）
# ---------------------------------------------------------------------------


def _frontier_coordinates(result):
    return list(zip(result.frontier[X].tolist(), result.frontier[Y].tolist()))


def test_constant_x_collapses_the_frontier_to_one_point():
    result = select_pareto_knee(_frame([(5, 3.0), (5, 2.0), (5, 1.0), (5, 4.0)]), x_column=X, y_column=Y)

    _assert_undefined(result, "too_few_points")
    assert _frontier_coordinates(result) == [(5, 1.0)]


def test_constant_y_collapses_the_frontier_to_one_point():
    result = select_pareto_knee(_frame([(1, 2.0), (3, 2.0), (2, 2.0)]), x_column=X, y_column=Y)

    _assert_undefined(result, "too_few_points")
    assert _frontier_coordinates(result) == [(1, 2.0)]


def test_dominated_points_that_leave_a_single_frontier_point():
    frame = _frame([(1, 1.0), (2, 5.0), (3, 6.0), (4, 7.0)])

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    _assert_undefined(result, "too_few_points")
    assert _frontier_coordinates(result) == [(1, 1.0)]


def test_dominated_points_that_leave_two_frontier_points():
    frame = _frame([(1, 5.0), (4, 1.0), (3, 6.0), (5, 4.0), (2, 7.0)])

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    _assert_undefined(result, "too_few_points")
    assert _frontier_coordinates(result) == [(1, 5.0), (4, 1.0)]


def test_three_distinct_frontier_points_can_define_a_knee():
    result = select_pareto_knee(_frame(CLEAR), x_column=X, y_column=Y)

    assert result.is_defined is True
    assert result.undefined_reason is None
    assert len({(float(x), float(y)) for x, y in _frontier_coordinates(result)}) >= 3
    assert result.knee_index == 1


def test_duplicate_coordinates_do_not_invent_extra_frontier_points():
    # 同じ座標の重複は、重複除去後の frontier が2点なら点数不足のまま。
    frame = _frame([(1, 5.0), (1, 5.0), (4, 1.0), (4, 1.0), (6, 8.0)], index=list("abcde"))

    result = select_pareto_knee(frame, x_column=X, y_column=Y)

    _assert_undefined(result, "too_few_points")
    assert list(result.frontier[X]) == [1, 1, 4, 4]
    assert result.frontier_source_index == ("a", "b", "c", "d")


def test_frontier_reason_is_invariant_to_row_order_and_input_is_not_modified():
    points = [(5, 3.0), (2, 9.0), (5, 1.0), (8, 7.0), (5, 1.0)]
    frame = _frame(points, index=list("abcde"))
    snapshot = frame.copy(deep=True)
    baseline = select_pareto_knee(frame, x_column=X, y_column=Y)
    rng = np.random.default_rng(2)

    for _ in range(10):
        permuted = frame.iloc[rng.permutation(len(frame))]
        result = select_pareto_knee(permuted, x_column=X, y_column=Y)
        assert result.undefined_reason == baseline.undefined_reason == "too_few_points"
        assert _frontier_coordinates(result) == _frontier_coordinates(baseline)

    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)
