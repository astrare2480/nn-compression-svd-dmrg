# `agreement`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

baseline/compressed modelが同じclassを予測する割合をloader全体で計算する。

## Signature

```python
agreement(
    baseline_model,
    compressed_model,
    loader,
    device,
) -> float
```

## 引数

2モデル、評価loader、device。

## 戻り値

予測class一致率 `[0, 1]`。

## 使用場面

accuracyとは別に「baselineの判断をどれだけ保持したか」を測るとき。

## ざっくりした処理

```text
両modelの状態保存
→ eval
→ inference_mode
→ 各batchでargmax比較
→ 一致数/総sample数
→ training状態復元
```

## 主なcontract / 注意事項

- empty loaderは`ValueError`。
- root + 全submodule training状態を復元。

## 関連API

`logits_rmse`, `collect_compression_metrics`
