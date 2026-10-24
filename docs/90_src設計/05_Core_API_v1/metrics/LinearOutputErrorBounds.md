# `LinearOutputErrorBounds`

**Stability:** A
**定義:** `src/nn_compression/metrics/error_bounds.py`

## 責務

Linear出力誤差に対する2種類のノルム上界を、名前付きで保持する不変dataclass。

## Signature
```python
LinearOutputErrorBounds(frobenius_x_spectral_dw, spectral_x_frobenius_dw)
```

## 引数

- `frobenius_x_spectral_dw`: $\lVert X\rVert_F\lVert\Delta W\rVert_2$を表す0階Tensor。
- `spectral_x_frobenius_dw`: $\lVert X\rVert_2\lVert\Delta W\rVert_F$を表す0階Tensor。

## 戻り値

2つの上界をfieldとして保持するfrozen instance。

## 使用場面

`linear_output_error_bounds`の結果を、上界の導出を区別して参照するとき。

## 処理概要

値の計算は行わず、渡されたTensorをfieldへ保存する。

## 主なcontract / 注意事項

- 両fieldはbatch全体のFrobenius誤差に対する上界であり、sampleごとの値ではない。
- dtype/device/autogradは生成元Tensorに従う。

## 関連API

`linear_output_error_bounds`
