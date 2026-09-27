# `tt_svd_ranks`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt.py`  
**参照Notebook:** `notebooks/30_tt_mps/00_fundamentals/02_truncation_error_tradeoff.ipynb`

## 責務

Tensorを逐次SVDし、bondごとに異なるrank上限を指定したTT core列へ分解する。

## Signature

```python
tt_svd_ranks(
    X: torch.Tensor,
    ranks: Sequence[int],
) -> list[torch.Tensor]
```

## 引数

- `X`: 2階以上の実数Tensor。各dimensionは正で、dtypeは`torch.float32`または`torch.float64`とする。
- `ranks`: $d-1$ 個のbond rank上限。`ranks[k]`は第$k$回の逐次SVDで残す右bond dimensionの上限を表す。文字列は受理せず、各要素はbool以外の1以上の整数とする。

## 戻り値

$d$ 個のTT coreからなるlistを返す。各coreのshapeは

$$
G^{(k)} \in \mathbb R^{r_{k-1}\times n_k\times r_k}
$$

で、実際の$r_k$は指定上限とその段階の数値rankの小さい方になる。

## 使用場面

- bondごとのrank配分を比較する実験。
- parameter budgetに合わせて内部bondを個別設計するとき。
- 一様上限の`tt_svd`では表せない非対称なrank構成を作るとき。

## 処理概要

1. `ranks`が文字列でない`Sequence`であることを確認する。
2. 各要素を`coerce_integer_scalar`で整数化し、1以上であることを確認する。
3. 共通実装`_tt_svd_sweep(X, bond_ranks=...)`へ委譲する。
4. 各cutで行列化、数値rank計算、truncated SVD、core化、残差の次siteへの受け渡しを行う。

## 主なcontract / 注意事項

- `len(ranks) == X.ndim - 1`を要求する。
- 指定rankが数値rankを超える場合は数値rankで頭打ちにする。
- 数値rankが0の零Tensorでは、TT bondを正のdimensionに保つためrank 1を使う。
- `ranks`は厳密な出力rankではなく上限である。
- `X`を変更しない。

## 関連API

[[90_src設計/05_Core_API_v1/compression/tt_svd]]、[[90_src設計/05_Core_API_v1/compression/tt_svd_exact]]、[[90_src設計/05_Core_API_v1/compression/tt_ranks]]、[[90_src設計/05_Core_API_v1/compression/tt_reconstruct]]
