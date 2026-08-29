# `parameters_reduction`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

baselineに対してcompressed modelのParameter総要素数がどれだけ減ったか、削減率として計算する。

## Signature

```python
parameters_reduction(
    baseline_model,
    compressed_model,
) -> float
```

## 引数

baseline modelとcompressed model。

## 戻り値

```text
1 - compressed_parameters / baseline_parameters
```

0.9なら90%削減、0なら削減なし。負値ならcompressed側のParameter数が増えている。

## 使用場面

model-levelの圧縮効果をparameter数で比較するとき。

## 処理概要

1. **baselineへ`count_parameters()`を適用する。**
2. **compressed modelにも同じ関数を適用する。**
3. **compressed / baselineの比率を計算する。**
4. **1からその比率を引いて削減率へ変換する。**
5. **floatを返す。**

## 主なcontract / 注意事項

accuracyやMACsの削減率とは別指標。Parameterが減ってもlatency短縮を保証しない。

## 関連API

`count_parameters`, `collect_compression_metrics`
