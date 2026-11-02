# `split_merged_tt_matrix`

**Stability:** A  
**定義:** `src/nn_compression/tt_mps/matrix_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb`

## 責務

結合テンソルを行列化して還元SVDし、左coreを $U$、右coreを $\Sigma V^{\mathsf T}$ として4階coreへ戻す。ノルムを1へ戻さない。

## Signature

```python
split_merged_tt_matrix(
    merged: torch.Tensor,
    *,
    max_bond: int | None = None,
    direction: str = "right",
) -> TTMatrixPairSplit
```

## 引数

- `merged`: shape `(r_{k-1}, m_k, n_k, m_{k+1}, n_{k+1}, r_{k+1})`の結合テンソル。行列化は左側 `(r_{k-1}, m_k, n_k)`、右側 `(m_{k+1}, n_{k+1}, r_{k+1})`。
- `max_bond`: 保持するbond次元の上限。`None`なら還元SVDの全成分を保持する。1以上の整数なら、その本数と還元SVDの本数の小さい方を保持する。
- `direction`: `"right"`（既定）または `"left"`。`"right"` は `G_k'=reshape(U_r)`、`G_{k+1}'=reshape(Σ_r V_r^T)`。`"left"` は `G_k'=reshape(U_r Σ_r)`、`G_{k+1}'=reshape(V_r^T)`。既定値があるので、向きを省略した既存の呼び出しは右向きのままである。

## 戻り値

`TTMatrixPairSplit`。`max_bond=None`のとき、再結合は丸め誤差を除いて`merged`を再現する。

## 使用場面

- 勾配更新後の結合テンソルを、切り捨てなしで2 coreへ戻して再構成を確認するとき。
- 同じ結合テンソルから、別のrank上限でFrobenius最良近似を作るとき。

## 処理概要

1. `merged`を `(r_{k-1} m_k n_k,\ m_{k+1} n_{k+1} r_{k+1})` の行列 $M$ にする。
2. `torch.linalg.svd(M, full_matrices=False)`で $M=U\Sigma V^{\mathsf T}$ を求める。返った特異値は全部保持し、`matrix_rank`では削らない。
3. `max_bond`が指定されていれば先頭だけを残す。
4. `direction="right"` なら $G_k'=\mathrm{reshape}(U_r)$、$G_{k+1}'=\mathrm{reshape}(\Sigma_r V_r^{\mathsf T})$。`direction="left"` なら $G_k'=\mathrm{reshape}(U_r \Sigma_r)$、$G_{k+1}'=\mathrm{reshape}(V_r^{\mathsf T})$。
5. 捨てた特異値の二乗和を返す。

## 主なcontract / 注意事項

- 既定の `direction="right"` は、従来の左から右への分割である。`"left"` は特異値を左 core に移す。
- 物理DMRGの`split_two_site_tensor`は分割後のテンソルを単位ノルムへ再正規化する。この関数はそれをしない。
- `max_bond`による切り捨ては、この行列のFrobeniusノルムに関する最良低rank近似であり、分類損失の最小化ではない。
- 入力は変更しない。`detach`も`torch.no_grad`も使わない。
- `max_bond=None`は、非ゼロ特異値の本数を推定して削る処理ではない。

## 関連API

[[90_src設計/05_Core_API_v1/tt_mps/TTMatrixPairSplit]]、[[90_src設計/05_Core_API_v1/tt_mps/merged_tt_matrix_to_matrix]]、[[90_src設計/05_Core_API_v1/tt_mps/merge_tt_matrix_cores]]、[[90_src設計/05_Core_API_v1/Internal_API]]
