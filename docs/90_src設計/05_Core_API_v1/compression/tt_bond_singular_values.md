# `tt_bond_singular_values`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt_rounding.py`  
**参照Notebook:** `notebooks/30_tt_mps/00_fundamentals/08_svd_center_move_and_schmidt_form.ipynb`

## 責務

TTをdense化せず、指定bondにおけるSchmidt特異値をmixed-canonical環境から求める。

## Signature

```python
tt_bond_singular_values(
    cores: list[torch.Tensor],
    bond: int,
) -> torch.Tensor
```

## 引数

- `cores`: 特異値を調べるTT core列。入力listと各Tensorは変更しない。
- `bond`: `cores[bond]`と`cores[bond + 1]`の間を表す0始まりのbond index。`0 <= bond < len(cores) - 1`を満たすbool以外の整数とする。

## 戻り値

指定bondの特異値を降順に並べた1階Tensorを返す。dtype/deviceとautograd graphは入力から維持する。

## 使用場面

- bondごとのentanglement spectrumまたはSchmidt spectrumを観察するとき。
- `tt_truncate_bond`のrank候補と捨てるtailを事前評価するとき。
- DMRGへ進む前に、canonicalizationとbond SVDの関係を確認するとき。

## 処理概要

1. TT core列、dtype、`bond`を検証する。
2. `tt_canonicalize(cores, center=bond)`で、指定cutの左右環境を等長写像にする。
3. center coreを`(r_{bond-1} n_bond, r_bond)`へ左展開する。
4. reduced SVDを行い、特異値`S`だけを返す。

## 主なcontract / 注意事項

- 返す値はcore単体の任意gaugeに依存する特異値ではなく、mixed-canonical化した環境でのbond特異値である。
- 内部でcanonicalizationとSVDを行うため、単純なmetadata取得ではない。
- single-core TTには内部bondがないため、有効な`bond`は存在しない。
- 極端に不均衡なgauge scaleでは、`tt_canonicalize`のQRと余り吸収がoverflow / underflowする既知制約を引き継ぐ。

## 関連API

[[90_src設計/05_Core_API_v1/compression/tt_canonicalize]]、[[90_src設計/05_Core_API_v1/compression/tt_canonicality_errors]]、[[90_src設計/05_Core_API_v1/compression/tt_truncate_bond]]、[[90_src設計/05_Core_API_v1/compression/tt_round]]
