# `accuracy_drop`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

baseline accuracyからcompressed accuracyがどれだけ低下したかを差分で返す。

## Signature

```python
accuracy_drop(
    baseline_accuracy,
    compressed_accuracy,
    verbose=True,
) -> float
```

## 引数

- `baseline_accuracy`: 圧縮前accuracy。
- `compressed_accuracy`: 圧縮後accuracy。
- `verbose`: 2つのaccuracyと差分を表示するか。

## 戻り値

`baseline_accuracy - compressed_accuracy`。

## 使用場面

圧縮によるtask性能差を、絶対accuracyと別の1列として比較表へ入れるとき。

## 処理の流れ（日本語）

1. **baseline accuracyからcompressed accuracyを引く。**
2. **`verbose=True`なら比較元・比較先・差分を表示する。**
3. **差分をfloat相当の値として返す。**

### 値の読み方

```text
正の値 → compressed側のaccuracyが低い
0       → 同じaccuracy
負の値 → compressed側のaccuracyが高い
```

## 主なcontract / 注意事項

この関数は統計的有意性を判定しない。single seedの小差を「改善」と強く解釈するかどうかは実験設計側の責務。

## 関連API

`collect_compression_metrics`
