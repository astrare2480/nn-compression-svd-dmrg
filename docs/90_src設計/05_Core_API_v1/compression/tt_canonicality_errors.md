# `tt_canonicality_errors`

**Stability:** A  
**定義:** `src/nn_compression/compression/tt_canonical.py`  
**参照Notebook:** `notebooks/30_tt_mps/00_fundamentals/06_mixed_canonical_form_fixed.ipynb`

## 責務

指定したorthogonality centerを基準に、左側coreの列直交性と右側coreの行直交性がどれだけ崩れているかを、site別Frobenius誤差として測定する。

## Signature

```python
tt_canonicality_errors(
    cores: list[torch.Tensor],
    center: int,
) -> dict[str, list[torch.Tensor]]
```

## 引数

- `cores`: canonicalityを診断するTT core列。各coreは`(r_{k-1}, n_k, r_k)`の3階Tensorで、入力listと各Tensorは変更しない。
- `center`: orthogonality centerとみなすsite index。`0 <= center < len(cores)`を満たすbool以外の整数とする。

## 戻り値

`{"left": left_errors, "right": right_errors}`を返す。

- `left`: site`0, ..., center - 1`の順に、左展開行列$M_k$の$\|M_k^\mathsf{T}M_k-I\|_F$を格納する。
- `right`: site`center + 1, ..., d - 1`の順に、右展開行列$N_k$の$\|N_kN_k^\mathsf{T}-I\|_F$を格納する。

center coreは一般の中心係数Tensorであり、どちらのlistにも含めない。

## 使用場面

- `tt_canonicalize`または`tt_move_center`の結果を数値的に検証するとき。
- roundingやSchmidt分解の前提であるmixed-canonical formを診断するとき。
- 学習・最適化後にcanonicalityの崩れをsite別に調べるとき。

## 処理概要

1. TT core列、dtype、`center`を検証する。
2. center左側ではcoreを`(r_{k-1} n_k, r_k)`へreshapeし、列Gram行列と単位行列との差を作る。
3. center右側ではcoreを`(r_{k-1}, n_k r_k)`へreshapeし、行Gram行列と単位行列との差を作る。
4. 各Gram差を最大絶対値で正規化してからFrobenius normを計算する。
5. 左右の誤差列をsite順に返す。

## 主なcontract / 注意事項

- このAPIは診断のみを行い、入力をcanonicalizeしない。
- 誤差の許容値を判定しない。float32/float64、rank、用途に応じた閾値は呼び出し側が決める。
- 各誤差は入力と同じdtype/deviceのスカラーTensorで、autograd graphを維持する。
- norm計算はscale-safeに行い、float32の`1e-30`やfloat64の`1e-200`程度の表現可能なGram差を、二乗underflowによって0と誤報しない。
- `d == 1`かつ`center == 0`では左右とも空listを返す。
- 値が0に近いほど指定centerに対するcanonicalityが良いことを表す。

## 関連API

[[90_src設計/05_Core_API_v1/compression/tt_canonicalize]]、[[90_src設計/05_Core_API_v1/compression/tt_move_center]]、[[90_src設計/05_Core_API_v1/compression/tt_bond_singular_values]]
