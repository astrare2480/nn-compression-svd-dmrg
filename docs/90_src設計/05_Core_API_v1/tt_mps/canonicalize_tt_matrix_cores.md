# `canonicalize_tt_matrix_cores`

**Stability:** A  
**定義:** `src/nn_compression/tt_mps/matrix_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb`

## 責務

TT-matrix core列を、指定siteを直交中心とする混合正準形へ変換する。3階TTへまとめてから既存の`tt_canonicalize`を使う。

## Signature

```python
canonicalize_tt_matrix_cores(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    center: int,
) -> list[torch.Tensor]
```

## 引数

- `cores`: 4階TT-matrix core列。変更しない。
- `out_modes`: 出力mode $(m_1,\ldots,m_d)$。
- `in_modes`: 入力mode $(n_1,\ldots,n_d)$。
- `center`: 直交中心のsite index。`0 <= center < d`。

## 戻り値

混合正準形の新しい4階core列。dense重みは、QRの丸めを除いて変わらない。

## 使用場面

- 隣接pairの局所更新の前に、pairの外側を等長写像にしておくとき。
- pairの左siteを`center`にし、左を左直交、右を右直交にするとき。

## 処理概要

1. 各coreを `(r_{k-1}, m_k n_k, r_k)` にまとめる。
2. `tt_canonicalize(cores_3d, center)`を呼ぶ。
3. 結果を4階TT-matrix coreへ戻す。

## 主なcontract / 注意事項

- `center`より左は左直交、`center`より右は右直交である。`center`自身には直交条件を課さない。
- pair $(k,k+1)$に対して`center=k`とすると、pairの右coreも右直交に含まれる。pairの左coreは直交中心である。
- このgaugeでは、pairを結合したテンソルのFrobeniusノルムは重み $W$ のFrobeniusノルムと一致する。正準化していない列では、その一致を仮定しない。
- reduced QRはbond次元を`min(行数, 列数)`まで変えることがある。数値rankによる切り捨てはしない。
- 入力は変更しない。

## 関連API

[[90_src設計/05_Core_API_v1/tt_mps/tt_canonicalize]]、[[90_src設計/05_Core_API_v1/tt_mps/tt_matrix_cores_to_tt_cores]]、[[90_src設計/05_Core_API_v1/tt_mps/tt_matrix_canonicality_errors]]、[[90_src設計/05_Core_API_v1/Internal_API]]
