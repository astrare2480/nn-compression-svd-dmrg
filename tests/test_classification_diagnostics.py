"""正解クラス margin と予測遷移。CPU の小さい Tensor だけを使う。"""

from __future__ import annotations

import pytest
import torch

from nn_compression.metrics import (
    MarginChangeSummary,
    PredictionTransitionSummary,
    classification_margins,
    margin_change_summary,
    prediction_transition_summary,
    true_class_margins,
)


def test_true_class_margin_can_be_negative_and_differs_from_top_margin():
    logits = torch.tensor([[0.0, 3.0, 1.0], [4.0, 1.0, 0.0]])
    labels = torch.tensor([0, 0])
    original = logits.clone()
    margins = true_class_margins(logits, labels)
    assert torch.allclose(margins, torch.tensor([-3.0, 3.0]))
    assert torch.equal(logits, original)
    top_margin = classification_margins(logits)
    assert torch.all(top_margin >= 0)
    assert not torch.allclose(margins, top_margin)
    with pytest.raises(ValueError, match="範囲外"):
        true_class_margins(logits, torch.tensor([0, 3]))


def test_margin_change_summary_counts_worsened_rows():
    baseline = torch.tensor([1.0, -0.5, 2.0])
    candidate = torch.tensor([0.0, 0.5, 2.0])
    summary = margin_change_summary(baseline, candidate)
    assert isinstance(summary, MarginChangeSummary)
    assert torch.equal(summary.delta_margin, torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float64))
    assert summary.delta_margin_mean == pytest.approx(0.0)
    assert summary.margin_worsened_count == 1


def test_prediction_transition_does_not_depend_on_a_fixed_sample_count():
    labels = torch.tensor([0, 0, 1])
    baseline = torch.tensor([0, 0, 0])
    candidate = torch.tensor([0, 1, 1])
    summary = prediction_transition_summary(labels, baseline, candidate)
    assert isinstance(summary, PredictionTransitionSummary)
    assert summary.correct_to_correct == 1
    assert summary.correct_to_incorrect == 1
    assert summary.incorrect_to_correct == 1
    assert summary.incorrect_to_incorrect == 0
    assert summary.prediction_changed_count == 2
    assert summary.accuracy_delta == pytest.approx(0.0)

    shifted = prediction_transition_summary(
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
    )
    assert shifted.correct_to_incorrect == 1
    assert shifted.prediction_changed_count == 1
    assert shifted.accuracy_delta == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="同じ長さ"):
        prediction_transition_summary(labels, baseline, candidate[:2])
