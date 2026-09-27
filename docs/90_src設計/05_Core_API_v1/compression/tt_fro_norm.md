# `tt_fro_norm`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt_contraction.py`  
**参照Notebook:** `notebooks/30_tt_mps/00_fundamentals/13_tt_inner_frobenius_norm_and_distance.ipynb`

## 責務

実数値TTのFrobeniusノルムを、dense化せず自己内積から計算する。

## Signature

```python
tt_fro_norm(
    cores: list[torch.Tensor],
) -> torch.Tensor
```

## 引数

- `cores`: `float32`または`float64`のTT core列。TT構造、dtype、deviceの整合を満たす必要がある。

## 戻り値

shape `()` のTensorとして

$$
\|\mathcal A\|_F
=
\sqrt{\langle\mathcal A,\mathcal A\rangle}
$$

を返す。零TTではforward値を厳密に0とし、backward時の勾配も有限な0とする。

## 使用場面

- TTの大きさをdense化せず評価するとき。
- roundingの誤差・相対誤差を外部検証するとき。
- contraction実装のdense基準との一致確認。

## 処理概要

1. `tt_inner(cores, cores)`で自己内積を求める。
2. internal helper `_safe_sqrt`で平方根を計算する。
3. 正値では通常の`torch.sqrt`と同じ値・勾配を保ち、零点ではNaN勾配を避ける分岐を使う。

## 主なcontract / 注意事項

- 正の自己内積では`torch.sqrt`と一致する。
- 自己内積が`0`または`-0.0`なら0を返し、勾配も0とする。
- 有限の負値または`-inf`は、数値失敗を隠さず`ValueError`とする。
- `NaN`は0へ丸めず、そのまま伝播する。
- `_safe_sqrt`の負値判定は`.item()`を使うため、CUDAでは同期が発生する。
- `tt_inner`と同じく極端なcore scaleではoverflow / underflowし得る。
- TT間距離は公開責務に含めない。

## 関連API

[[90_src設計/05_Core_API_v1/compression/tt_inner]]、[[90_src設計/05_Core_API_v1/compression/tt_round]]、[[90_src設計/05_Core_API_v1/Internal_API]]、[[90_src設計/07_既知の制約と拡張方針]]
