# `relative_frobenius_error`

**Stability:** A  
**定義:** `src/nn_compression/metrics/tensor_approximation.py`

## 責務

元Tensorと近似Tensorの差をFrobenius normで測り、元Tensorのnormで正規化した相対誤差を返す。

## Signature

```python
relative_frobenius_error(
    X: torch.Tensor,
    X_hat: torch.Tensor,
) -> torch.Tensor
```

## 引数

同shapeの元Tensor`X`と近似Tensor`X_hat`。

## 戻り値

```text
||X - X_hat||_F / ||X||_F
```

のscalar Tensor。

## 使用場面

SVD/Tucker/HOOIのweight再構成誤差比較、自作分解とTensorLy等の数値照合。

## 処理概要

1. **`X`と`X_hat`のshapeが一致するか確認する。**
2. **元TensorのFrobenius normを計算する。**  
   `torch.linalg.vector_norm(X)`を分母にする。
3. **分母が0でないか確認する。**  
   zero tensorでは相対誤差を定義できないため`ValueError`にする。
4. **差Tensor`X - X_hat`のFrobenius normを計算する。**
5. **差のnormを元Tensorのnormで割る。**
6. **scalar Tensorのまま返す。**  
   関数内で`detach()`やPython float化をしないため、必要なら呼び出し側がautograd graphを利用できる。

## 主なcontract / 注意事項

- shape不一致は`ValueError`。
- zero tensorは`ValueError`。
- weight errorの小ささはtask accuracyの高さを保証しない。

## 関連API

`hooi`, `reconstruct_tucker`, `tucker2_effective_weight`
