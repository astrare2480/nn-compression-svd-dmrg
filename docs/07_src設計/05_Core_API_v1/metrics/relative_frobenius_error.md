# `relative_frobenius_error`

**Stability:** A  
**定義:** `src/nn_compression/metrics/tensor_approximation.py`

## 責務

元Tensorと近似Tensorの相対Frobenius誤差を計算する。

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

SVD/Tucker/HOOIのweight再構成誤差比較、自作実装とTensorLy照合。

## ざっくりした処理

shape確認 → `||X||`計算 → zero確認 → 差のnormを分母で割る。

## 主なcontract / 注意事項

- shape不一致は`ValueError`。
- zero tensorでは相対値が定義できないため`ValueError`。
- 関数内でdetach/float化せずTensorを返す。

## 関連API

`hooi`, `reconstruct_tucker`, `tucker2_effective_weight`
