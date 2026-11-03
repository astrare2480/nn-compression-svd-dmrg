"""strict best と early-stopping reference を分けて更新する。

strict best は validation loss が少しでも下がった時点の checkpoint 用。
stopping reference は、以前の reference から ``min_delta`` 以上下がったときだけ動き、
patience はその reference に対する非改善回数である。
"""

from __future__ import annotations

import math
from collections.abc import Sequence


class ValidationSelection:
    """1 回の validation 観測後の選択状態。"""

    def __init__(
        self,
        strict_best_loss: float,
        strict_best_step: int,
        reference_loss: float,
        non_improvement_count: int,
        early_stop_triggered: bool,
    ) -> None:
        self.strict_best_loss = float(strict_best_loss)
        self.strict_best_step = int(strict_best_step)
        self.reference_loss = float(reference_loss)
        self.non_improvement_count = int(non_improvement_count)
        self.early_stop_triggered = bool(early_stop_triggered)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValidationSelection):
            return NotImplemented
        return (
            self.strict_best_loss == other.strict_best_loss
            and self.strict_best_step == other.strict_best_step
            and self.reference_loss == other.reference_loss
            and self.non_improvement_count == other.non_improvement_count
            and self.early_stop_triggered == other.early_stop_triggered
        )

    def __repr__(self) -> str:
        return (
            "ValidationSelection("
            f"strict_best_loss={self.strict_best_loss}, "
            f"strict_best_step={self.strict_best_step}, "
            f"reference_loss={self.reference_loss}, "
            f"non_improvement_count={self.non_improvement_count}, "
            f"early_stop_triggered={self.early_stop_triggered})"
        )


def _require_finite_loss(loss: float) -> float:
    try:
        value = float(loss)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"validation loss が有限値ではありません: {loss!r}") from exc
    if isinstance(loss, bool) or not math.isfinite(value):
        raise ValueError(f"validation loss が有限値ではありません: {loss!r}")
    return value


def _require_patience(patience: int) -> int:
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise ValueError(f"patience は 1 以上の整数である必要があります: {patience!r}")
    return patience


def _require_min_delta(min_delta: float) -> float:
    try:
        value = float(min_delta)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"min_delta は 0 以上の有限値である必要があります: {min_delta!r}") from exc
    if isinstance(min_delta, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"min_delta は 0 以上の有限値である必要があります: {min_delta!r}")
    return value


def initial_validation_selection(loss: float, step: int) -> ValidationSelection:
    """最初の観測を strict best と stopping reference の両方にする。"""
    finite = _require_finite_loss(loss)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step は 0 以上の整数である必要があります: {step!r}")
    return ValidationSelection(
        strict_best_loss=finite,
        strict_best_step=step,
        reference_loss=finite,
        non_improvement_count=0,
        early_stop_triggered=False,
    )


def update_validation_selection(
    state: ValidationSelection,
    loss: float,
    step: int,
    *,
    min_delta: float,
    patience: int,
) -> ValidationSelection:
    """1 観測を反映する。

    loss が strict best より少しでも低ければ best の位置を更新する。
    reference は、loss が reference より厳密に低く、その低下量が ``min_delta``
    以上のときだけ現在の loss へ動かし、非改善回数を 0 に戻す。
    ``min_delta=0`` で loss が reference と同値のときは改善にしない。
    それ以外は非改善として回数を 1 増やす。回数が ``patience`` に達した観測で
    ``early_stop_triggered`` を真にする。
    """
    if not isinstance(state, ValidationSelection):
        raise TypeError("state は ValidationSelection である必要があります。")
    if state.early_stop_triggered:
        raise RuntimeError("early stopping 成立後の観測は受け付けません。")
    finite = _require_finite_loss(loss)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step は 0 以上の整数である必要があります: {step!r}")
    delta = _require_min_delta(min_delta)
    limit = _require_patience(patience)
    if finite < state.strict_best_loss:
        best_loss = finite
        best_step = step
    else:
        best_loss = state.strict_best_loss
        best_step = state.strict_best_step
    decrease = state.reference_loss - finite
    if decrease > 0.0 and decrease >= delta:
        reference = finite
        count = 0
    else:
        reference = state.reference_loss
        count = state.non_improvement_count + 1
    return ValidationSelection(
        strict_best_loss=best_loss,
        strict_best_step=best_step,
        reference_loss=reference,
        non_improvement_count=count,
        early_stop_triggered=count >= limit,
    )


def replay_validation_selection(
    losses: Sequence[float],
    steps: Sequence[int],
    *,
    min_delta: float,
    patience: int,
) -> list[ValidationSelection]:
    """先頭から観測列を再生し、各時点の選択状態を返す。"""
    if len(losses) != len(steps):
        raise ValueError("losses と steps の長さが一致しません。")
    if len(losses) < 1:
        raise ValueError("観測列は 1 点以上必要です。")
    states = [initial_validation_selection(losses[0], steps[0])]
    for loss, step in zip(losses[1:], steps[1:], strict=True):
        states.append(
            update_validation_selection(
                states[-1],
                loss,
                step,
                min_delta=min_delta,
                patience=patience,
            )
        )
        if states[-1].early_stop_triggered:
            break
    return states
