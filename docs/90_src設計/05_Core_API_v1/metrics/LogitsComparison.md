# `LogitsComparison`

**Stability:** A
**定義:** `src/nn_compression/metrics/classification_outputs.py`

## 責務

2組のlogitsの予測一致率、RMSE、最大絶対誤差を保持する不変dataclass。

## Signature
```python
LogitsComparison(prediction_agreement, logits_rmse, logits_max_abs_error)
```

## 引数

- `prediction_agreement`: 行ごとの`argmax`一致率。
- `logits_rmse`: 全logit要素のRMSE。
- `logits_max_abs_error`: 最大絶対差。

## 戻り値

3つのPython `float` fieldを持つfrozen instance。

## 使用場面

baselineと圧縮候補の判断一致とlogit値の差を分けて記録するとき。

## 処理概要

値の計算は`compare_logits`が担当する。

## 主なcontract / 注意事項

- agreementとaccuracyは異なる指標である。
- RMSEと最大誤差はclass予測が同じでも非零になり得る。

## 関連API

`compare_logits`, `agreement`, `logits_rmse`
