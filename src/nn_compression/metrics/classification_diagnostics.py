"""正解クラスを基準にした margin と、予測の正誤遷移。

``classification_margins`` は予測 top-1 と top-2 の差である。
ここでの margin は、正解クラスの logit から最大の誤クラス logit を引いた値で、
負になり得る。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MarginChangeSummary:
    """候補 margin から baseline margin を引いた変化。"""

    delta_margin: torch.Tensor
    delta_margin_mean: float
    delta_margin_median: float
    delta_margin_p10: float
    delta_margin_p90: float
    margin_worsened_count: int


@dataclass(frozen=True)
class PredictionTransitionSummary:
    """baseline 予測から候補予測への正誤の遷移。"""

    correct_to_correct: int
    correct_to_incorrect: int
    incorrect_to_correct: int
    incorrect_to_incorrect: int
    prediction_changed_count: int
    accuracy_delta: float


def _finite_quantile(values: torch.Tensor, quantile: float) -> float:
    result = torch.quantile(values, float(quantile))
    if not torch.isfinite(result):
        raise ValueError("分位点が有限ではありません。")
    return float(result)


def _labels(labels: torch.Tensor, batch: int, class_count: int) -> torch.Tensor:
    if not torch.is_tensor(labels) or labels.ndim != 1 or labels.shape[0] != batch:
        raise ValueError(
            "labels は logits と同じ batch 長の 1 階 Tensor である必要があります: "
            f"shape={tuple(labels.shape) if torch.is_tensor(labels) else type(labels)!r}"
        )
    if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
        raise TypeError(f"labels は整数 dtype である必要があります: {labels.dtype}")
    values = labels.to(dtype=torch.int64)
    if batch > 0 and (int(values.min()) < 0 or int(values.max()) >= class_count):
        raise ValueError(
            f"正解ラベルがクラス番号の範囲外です: min={int(values.min())} max={int(values.max())} classes={class_count}"
        )
    return values


def true_class_margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """正解クラスの logit から、それ以外の最大 logit を引いた margin を返す。

    正なら正解クラスが最大、負なら別クラスの方が大きい。
    ``classification_margins`` のような非負の top-1 margin とは別である。
    入力は変更しない。戻り値の dtype と device は ``logits`` に合わせる。
    """
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError(
            "logits は 2 階 (batch, classes) である必要があります: "
            f"shape={tuple(logits.shape) if torch.is_tensor(logits) else type(logits)!r}"
        )
    if logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError(f"logits は 1 件以上、2 クラス以上が必要です: shape={tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("logits に非有限値があるため、margin は計算しません。")
    label_index = _labels(labels, int(logits.shape[0]), int(logits.shape[1])).to(device=logits.device)
    correct = logits.gather(1, label_index.view(-1, 1)).squeeze(1)
    excluded = torch.nn.functional.one_hot(label_index, int(logits.shape[1])).to(dtype=torch.bool)
    other = logits.masked_fill(excluded, torch.finfo(logits.dtype).min).max(dim=1).values
    if not torch.isfinite(other).all():
        raise ValueError("正解以外の最大 logit が有限ではありません。")
    margin = correct - other
    if not torch.isfinite(margin).all():
        raise ValueError("margin が有限ではありません。")
    return margin


def margin_change_summary(
    baseline_margin: torch.Tensor,
    candidate_margin: torch.Tensor,
) -> MarginChangeSummary:
    """``candidate - baseline`` の分布と、margin が下がった件数を返す。"""
    if (
        not torch.is_tensor(baseline_margin)
        or not torch.is_tensor(candidate_margin)
        or baseline_margin.shape != candidate_margin.shape
        or baseline_margin.ndim != 1
        or baseline_margin.shape[0] < 1
    ):
        raise ValueError("baseline と candidate の margin は、空でない同じ形の 1 階 Tensor である必要があります。")
    if not torch.isfinite(baseline_margin).all() or not torch.isfinite(candidate_margin).all():
        raise ValueError("margin に非有限値があります。0 へは置き換えません。")
    delta = candidate_margin.detach().to(dtype=torch.float64) - baseline_margin.detach().to(dtype=torch.float64)
    if not torch.isfinite(delta).all():
        raise ValueError("delta_margin が有限ではありません。")
    mean = delta.mean()
    if not torch.isfinite(mean):
        raise ValueError("delta_margin の平均が有限ではありません。")
    return MarginChangeSummary(
        delta_margin=delta,
        delta_margin_mean=float(mean),
        delta_margin_median=_finite_quantile(delta, 0.5),
        delta_margin_p10=_finite_quantile(delta, 0.1),
        delta_margin_p90=_finite_quantile(delta, 0.9),
        margin_worsened_count=int((delta < 0).sum()),
    )


def _prediction(prediction: torch.Tensor, count: int, name: str) -> torch.Tensor:
    if not torch.is_tensor(prediction) or prediction.ndim != 1 or prediction.shape[0] != count:
        raise ValueError(f"{name} は labels と同じ長さの 1 階 Tensor である必要があります。")
    if prediction.dtype == torch.bool or prediction.is_floating_point() or prediction.is_complex():
        raise TypeError(f"{name} は整数 dtype である必要があります: {prediction.dtype}")
    return prediction.to(dtype=torch.int64)


def prediction_transition_summary(
    labels: torch.Tensor,
    baseline_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
) -> PredictionTransitionSummary:
    """正誤の 4 遷移、予測が変わった件数、accuracy の差を返す。

    accuracy の差は ``(incorrect→correct - correct→incorrect) / サンプル数`` である。
    サンプル数は引数の長さから取り、固定の件数には依存しない。
    """
    if not torch.is_tensor(labels) or labels.ndim != 1 or labels.shape[0] < 1:
        raise ValueError("labels は 1 件以上の 1 階 Tensor である必要があります。")
    if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
        raise TypeError(f"labels は整数 dtype である必要があります: {labels.dtype}")
    count = int(labels.shape[0])
    label_values = labels.to(dtype=torch.int64).view(-1)
    baseline = _prediction(baseline_prediction, count, "baseline_prediction")
    candidate = _prediction(candidate_prediction, count, "candidate_prediction")
    baseline_hit = baseline.eq(label_values)
    candidate_hit = candidate.eq(label_values)
    correct_to_correct = int((baseline_hit & candidate_hit).sum())
    correct_to_incorrect = int((baseline_hit & ~candidate_hit).sum())
    incorrect_to_correct = int((~baseline_hit & candidate_hit).sum())
    incorrect_to_incorrect = int((~baseline_hit & ~candidate_hit).sum())
    if (
        correct_to_correct + correct_to_incorrect + incorrect_to_correct + incorrect_to_incorrect
        != count
    ):
        raise RuntimeError("正誤遷移の合計がサンプル数と一致しません。")
    changed = baseline.ne(candidate)
    changed_count = int(changed.sum())
    if int((baseline_hit & candidate_hit & changed).sum()) != 0:
        raise ValueError("correct→correct で予測クラスが変化しています。")
    incorrect_changed_class = int((~baseline_hit & ~candidate_hit & changed).sum())
    if changed_count != correct_to_incorrect + incorrect_to_correct + incorrect_changed_class:
        raise RuntimeError("予測変更数が正誤遷移の内訳と一致しません。")
    return PredictionTransitionSummary(
        correct_to_correct=correct_to_correct,
        correct_to_incorrect=correct_to_incorrect,
        incorrect_to_correct=incorrect_to_correct,
        incorrect_to_incorrect=incorrect_to_incorrect,
        prediction_changed_count=changed_count,
        accuracy_delta=(incorrect_to_correct - correct_to_incorrect) / count,
    )
