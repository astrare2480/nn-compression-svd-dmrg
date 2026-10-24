# `classification_metrics_from_logits`

**Stability:** A
**定義:** `src/nn_compression/metrics/classification_outputs.py`

## 責務

計算済みlogitsと整数labelsから、全sample平均cross entropyとaccuracyを再forwardなしで計算する。

## Signature
```python
classification_metrics_from_logits(logits, labels)
```

## 引数

- `logits`: shape `(N,C)`の有限float32/float64 Tensor。
- `labels`: shape `(N,)`のboolでない整数Tensor。値域は`[0,C)`、deviceはlogitsと同じ。

## 戻り値

`ClassificationMetrics`。lossはfloat64で全sample平均、accuracyはPython float、sample数はPython int。

## 使用場面

保存済みlogitsからrank候補の分類性能を計算するとき。

## 処理概要

`torch.no_grad()`でlogitsをfloat64へ昇格して`F.cross_entropy`を計算し、元logitsの`argmax`とint64 labelsを比較する。

## 主なcontract / 注意事項

- 評価専用でautogradを契約に含めない。入力を変更しない。
- 空、NaN/inf、shape/device不一致、範囲外labelを拒否する。

## 関連API

`ClassificationMetrics`, `collect_model_logits`
