# `tt_unfold`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt.py`  
**参照Notebook:** `notebooks/30_tt_mps/00_fundamentals/01_tt_rank_unfolding.ipynb`

## 責務

$d$ 階テンソル $X$ を、TTのcut

$$
1{:}k \mid k+1{:}d
$$

に対応する2次元行列へ変形する。

Tuckerのmode-$n$ unfoldingとは異なり、**先頭から連続する $k$ 個のmodeを行側へまとめ、残りを列側へまとめる**ことをCore API v1の意味として固定する。

## Signature

```python
tt_unfold(
    X: torch.Tensor,
    k: int,
) -> torch.Tensor
```

## 引数

- `X`: TT cut unfoldingの対象Tensor。`ndim >= 2` かつ全dimensionが正であることを要求する。`tt_unfold` 自体はSVDを行わないため、dtypeを`float32` / `float64`へ限定しない。
- `k`: cut位置。先頭から何modeを行側へ含めるかを表す。`bool`以外の整数で、

  $$
  1 \le k < X.ndim
  $$

  を満たす必要がある。

## 戻り値

`X.shape == (n_1,\ldots,n_d)` のとき、shape

$$
\left(
\prod_{j=1}^{k} n_j,
\prod_{j=k+1}^{d} n_j
\right)
$$

を持つ2次元Tensorを返す。

例えば4階Tensorでは、

```text
tt_unfold(X, 1) -> (n1,       n2*n3*n4)
tt_unfold(X, 2) -> (n1*n2,    n3*n4)
tt_unfold(X, 3) -> (n1*n2*n3, n4)
```

となる。

## 使用場面

- 元Tensorの各cutにおけるTT-rankの確認。
- `tt_svd_exact` が生成したbond dimensionとcut unfolding rankの照合。
- TT/MPS理論の

  $$
  r_k
  =
  \operatorname{rank}\left(X^{\langle k\rangle}\right)
  $$

  を数値的に検証するとき。

## 処理概要

1. `validate_tensor_shape(X)` で、`ndim >= 2` と各dimensionが正であることを確認する。
2. `validate_tt_cut_index(k, X.ndim)` でTT cutの有効範囲を確認する。
3. `left_dim = math.prod(X.shape[:k])` として行側dimensionを計算する。
4. `right_dim = math.prod(X.shape[k:])` として列側dimensionを計算する。
5. `X.reshape(left_dim, right_dim)` を返す。

## 主なcontract / 注意事項

- TT cutは常に**左から連続して切る**。
- `k=0` と `k=X.ndim` は、左右どちらかが空になるため許可しない。
- `bool` はPythonでは`int`のsubclassだが、cut indexとしては拒否する。
- 軸順は変更しない。`permute`は行わず、現在のaxis順を保ったまま連続軸を`reshape`する。
- 3階Tensorで

  ```python
  X.reshape(n1 * n2, n3)
  ```

  はTTの第2 cut

  $$
  (i_1,i_2)\mid i_3
  $$

  であり、mode-2 unfolding

  $$
  i_2\mid(i_1,i_3)
  $$

  ではない。後者には`permute(1, 0, 2)`が必要になる。
- `tt_unfold` は**元Tensorのcut-rankを調べるAPI**であり、`tt_svd_exact` / `tt_svd` の逐次SVD内部で`remainder.reshape(...)`の代わりには使わない。
- `reshape`の戻り値がviewになるかcopyになるかは入力layoutに依存するため、API contractでは固定しない。

## 関連API

`tt_svd_exact`, `tt_svd`, `unfold`, `validate_tt_cut_index`
