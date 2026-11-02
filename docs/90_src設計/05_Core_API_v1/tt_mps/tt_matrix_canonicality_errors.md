# `tt_matrix_canonicality_errors`

**Stability:** A  
**定義:** `src/nn_compression/tt_mps/matrix_two_site.py`
**参照Notebook:** `notebooks/40_dmrg/00_fundamentals/04_tt_mlp_two_core_local_update.ipynb`

## 責務

TT-matrix core列の左右直交誤差を、物理軸を1本にまとめたうえで既存の`tt_canonicality_errors`と同じ定義で返す。

## Signature

```python
tt_matrix_canonicality_errors(
    cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    center: int,
) -> dict[str, list[torch.Tensor]]
```

## 引数

- `cores`: 4階TT-matrix core列。変更しない。
- `out_modes`: 出力mode $(m_1,\ldots,m_d)$。
- `in_modes`: 入力mode $(n_1,\ldots,n_d)$。
- `center`: 直交中心のsite index。このsite自身は評価しない。

## 戻り値

`{"left": [...], "right": [...]}`。`left`はsite `0 .. center-1`の $\lVert M^{\mathsf T}M-I\rVert_F$、`right`はsite `center+1 .. d-1`の $\lVert NN^{\mathsf T}-I\rVert_F$。各要素は入力と同じdtypeとdeviceのスカラーTensor。

## 使用場面

- `canonicalize_tt_matrix_cores`のあと、指定した中心に対する左右直交が数値的に成り立つかを確認するとき。

## 処理概要

1. 各coreを `(r_{k-1}, m_k n_k, r_k)` にまとめる。
2. `tt_canonicality_errors`へ渡す。
3. 返った辞書をそのまま返す。

## 主なcontract / 注意事項

- 左展開は `(r_{k-1} m_k n_k, r_k)`、右展開は `(r_{k-1}, m_k n_k r_k)` である。
- `center`自身のcoreは一般の中心係数なので、誤差リストに入れない。
- 誤差が小さいことは、dense重みそのものが変わっていないこととは別の確認である。

## 関連API

[[90_src設計/05_Core_API_v1/tt_mps/tt_canonicality_errors]]、[[90_src設計/05_Core_API_v1/tt_mps/canonicalize_tt_matrix_cores]]、[[90_src設計/05_Core_API_v1/tt_mps/tt_matrix_cores_to_tt_cores]]、[[90_src設計/05_Core_API_v1/Internal_API]]
