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

- `baseline_model`: 圧縮前の比較基準model。総Parameter要素数を削減率の分母として使う。
- `compressed_model`: 圧縮後の比較対象model。総Parameter要素数をbaselineと比較する。

## 戻り値

baselineに対してParameter要素数がどれだけ減ったかを表す削減率`float`。

```text
1 - compressed_parameters / baseline_parameters
```

`0.9`なら90%削減、`0`なら削減なし、負値ならcompressed側のParameter数が増えている。

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
