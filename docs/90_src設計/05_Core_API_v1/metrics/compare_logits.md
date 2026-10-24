# `compare_logits`

**Stability:** A
**定義:** `src/nn_compression/metrics/classification_outputs.py`

## 責務

同じsample順の2組のlogitsについて、予測class一致率と数値差をTensor単位で計算する。

## Signature
```python
compare_logits(reference_logits, candidate_logits)
```

## 引数

- `reference_logits`: 基準となるshape `(N,C)` logits。
- `candidate_logits`: 比較対象。基準とshape/dtype/deviceを一致させる。

## 戻り値

`LogitsComparison`。agreement、RMSE、最大絶対誤差をPython floatで返す。

## 使用場面

baselineと圧縮modelの収集済みlogitsを、modelの再forwardなしで比較するとき。

## 処理概要

各行の`argmax`一致率を求め、2のべき乗による共通scale上で差のRMSEと最大値を計算して元の単位へ戻す。

## 主なcontract / 注意事項

- 有限float32/float64、非空、同一shape/dtype/deviceを要求する。
- referenceが零でも使用できる。tie時のclassは`torch.argmax`と同じく最初の最大要素。
- 評価用でautogradを契約に含めない。

## 関連API

`LogitsComparison`, `collect_model_logits`, `tensor_approximation_metrics`
