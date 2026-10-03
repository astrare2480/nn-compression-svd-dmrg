# `tt_matrix_num_parameters`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt_matrix.py`  
**参照Notebook:** `notebooks/30_tt_mps/00_fundamentals/14_tt_matrix_dense_reconstruction_and_tt_linear_forward.ipynb`

## 責務

TT-matrix core列が保持するscalar parameterの総数を計算する。

## Signature

```python
tt_matrix_num_parameters(
    cores: Sequence[torch.Tensor],
) -> int
```

## 引数

- `cores`: 各要素がshape `(r_{k-1}, m_k, n_k, r_k)`のTT-matrix core列。計数前に構造、dtype、deviceを検証する。

## 戻り値

次の総要素数をPython `int`で返す。

$$
P_{\mathrm{TTM}}
=
\sum_{k=1}^{d}r_{k-1}m_kn_kr_k
$$

値が小さいほどTT-matrix表現が保持するscalar数は少ない。ただしdense weightの$mn$要素より小さいことは自動的には保証されない。

## 使用場面

- dense weightの$mn$要素との圧縮率比較。
- rank設計やparameter budgetの確認。
- 後続のTT-Linear実験でモデルサイズを記録するとき。

## 処理概要

1. `validate_tt_matrix_cores`でcore列を検証する。
2. 各coreの`numel()`を合計する。
3. 合計値をPython `int`として返す。

## 主なcontract / 注意事項

- bias parameterは数えない。対象は引数で渡されたTT-matrix coreだけである。
- dense換算parameter数やMACsは計算しない。
- 入力coreを変更せず、autograd graphにも演算を追加しない。

## 関連API

[[90_src設計/05_Core_API_v1/compression/tt_matrix_ranks]]、[[90_src設計/05_Core_API_v1/compression/tt_num_parameters]]、[[90_src設計/05_Core_API_v1/metrics/compressed_linear_macs]]
