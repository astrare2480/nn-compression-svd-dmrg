# `TTMatrixPairSplit`

**Stability:** A  
**定義:** `src/nn_compression/tt_mps/matrix_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb`

## 責務

結合テンソルを方向付きSVDで分割した結果を、再正規化しないcoreと切り捨て診断として保持する。右向きでは $\Sigma$ を右core、左向きでは $\Sigma$ を左coreに置く。

## Signature

```python
TTMatrixPairSplit(
    left_core: torch.Tensor,
    right_core: torch.Tensor,
    retained_rank: int,
    singular_values: torch.Tensor,
    discarded_squared_norm: torch.Tensor,
)
```

## 引数

- `left_core`: 右向きでは $G_k'=\mathrm{reshape}(U_r)$、左向きでは $G_k'=\mathrm{reshape}(U_r\Sigma_r)$。shape `(r_{k-1}, m_k, n_k, r_{\mathrm{keep}})`。
- `right_core`: 右向きでは $G_{k+1}'=\mathrm{reshape}(\Sigma_r V_r^{\mathsf T})$、左向きでは $G_{k+1}'=\mathrm{reshape}(V_r^{\mathsf T})$。shape `(r_{\mathrm{keep}}, m_{k+1}, n_{k+1}, r_{k+1})`。
- `retained_rank`: 保持した特異値の本数。
- `singular_values`: 切り捨て前の還元SVDが返した特異値全部。降順。
- `discarded_squared_norm`: 捨てた特異値の二乗和 $\sum_{j>r_{\mathrm{keep}}}\sigma_j^2$。0階Tensor。

## 戻り値

上記のフィールドを持つfrozen dataclass。インスタンス生成が戻り値そのものである。

## 使用場面

- `split_merged_tt_matrix`の戻り値として、再結合・切り捨て誤差・bond次元をあとから読むとき。

## 処理概要

1. 呼び出し側が還元SVDの因子から各フィールドを作る。
2. このclass自体は再正規化も数値rankの推定も行わない。

## 主なcontract / 注意事項

- `singular_values`は数値rankで小さい成分を落としたあとの列ではない。
- `discarded_squared_norm`は空のtailでは0であり、入力と同じdtypeとdeviceを持つ。
- 特異値をノルムで割ったcoreは入れない。

## 関連API

[[90_src設計/05_Core_API_v1/tt_mps/split_merged_tt_matrix]]、[[90_src設計/05_Core_API_v1/tt_mps/merge_tt_matrix_cores]]、[[90_src設計/05_Core_API_v1/Internal_API]]
