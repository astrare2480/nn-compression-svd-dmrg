# `compressed_linear_macs`

**Stability:** A  
**定義:** `src/nn_compression/metrics/macs.py`

## 責務

SVDで`Linear(in → out)`を`Linear(in → rank) → Linear(rank → out)`へ2層化した場合の理論MACsを計算する。

## Signature

```python
compressed_linear_macs(
    in_features: int,
    out_features: int,
    rank: int,
) -> int
```

## 引数

- `in_features`: 圧縮前Linearの入力feature数。
- `out_features`: 圧縮前Linearの出力feature数。
- `rank`: SVD 2層化後の中間dimension。第1層の出力数かつ第2層の入力数になる。

## 戻り値

2層Linearの理論MAC総数を表す整数。

```text
in_features * rank + rank * out_features
```

第1項が`in_features → rank`、第2項が`rank → out_features`の計算量。

## 使用場面

Linear SVD rank候補を理論計算量で比較するとき。

## 処理概要

1. **元行列で可能な最大rankを求める。**  
   `min(in_features, out_features)`。
2. **rankを整数・範囲contractで検証する。**
3. **入力側LinearのMACsを計算する。**  
   `in_features * rank`。
4. **出力側LinearのMACsを計算する。**  
   `rank * out_features`。
5. **2層分を合計して返す。**

## 主なcontract / 注意事項

rankはbool/Tensor scalar不可、`1 <= rank <= min(in_features, out_features)`。

## 関連API

`linear_macs`, `factorize_linear_layer`
