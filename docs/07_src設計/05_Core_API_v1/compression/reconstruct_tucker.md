# `reconstruct_tucker`

**Stability:** A  
**定義:** `src/nn_compression/compression/tucker.py`

## 責務

Tucker coreへfactor行列を掛け戻し、元空間の近似Tensorを再構成する。full Tuckerとpartial Tuckerの両方を同じ処理で扱う。

## Signature

```python
reconstruct_tucker(
    core: torch.Tensor,
    factors: dict[int, torch.Tensor],
) -> torch.Tensor
```

## 引数

- `core`: 2階以上のTucker core。
- `factors`: `{mode: U}`。factorがないmodeはそのまま維持される。

## 戻り値

factorを掛け戻した再構成Tensor。

## 使用場面

HOSVD/HOOIのrelative error評価、factor/coreの数値照合、Tucker-2 effective weightの再構成。

## 処理の流れ（日本語）

1. **coreが2階以上か確認する。**  
   Tucker系Public APIの最低階数を単独利用時にも維持する。
2. **再構成結果の初期値をcoreにする。**
3. **`factors`に含まれる各`(mode, U)`を順に処理する。**  
   factor未指定modeには何も行わないため、partial Tuckerも自然に扱える。
4. **各factorを`mode_dot(original, U, mode)`で掛け戻す。**  
   coreのrank dimensionを元Tensor側のdimensionへ戻す。
5. **すべてのfactorを掛けたTensorを返す。**

### 処理フロー（短縮版）

```text
core
→ ndim確認
→ mode 0 factorをmode_dot
→ mode 1 factorをmode_dot
→ ...指定factorだけ繰り返す
→ reconstructed Tensor
```

## 主なcontract / 注意事項

- 空`factors={}`ならidentityとしてcoreをそのまま返す。
- 分解入口ではないため、この関数単独ではcomplex dtypeを拒否しない。
- factorのshape不整合は`mode_dot()`側のdimension validationで検出される。

## 関連API

`hosvd`, `core_from_factors`, `relative_frobenius_error`, `tucker2_effective_weight`
