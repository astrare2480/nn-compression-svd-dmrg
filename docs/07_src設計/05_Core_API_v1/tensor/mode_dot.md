# `mode_dot`

**Stability:** A  
**定義:** `src/nn_compression/tensor/operations.py`

## 責務

Tensorの指定modeへ2次元行列を作用させるn-mode product（mode-n積）を計算する。

## Signature

```python
mode_dot(
    X: torch.Tensor,
    matrix: torch.Tensor,
    mode: int,
) -> torch.Tensor
```

## 引数

- `X`: 2階以上のTensor。
- `matrix`: `(new_dim, X.shape[mode])` の2次元行列。
- `mode`: 行列を作用させるaxis。

## 戻り値

`X` の対象modeだけが `matrix.shape[0]` に置き換わったTensor。他のmodeのdimensionとaxis順序は維持する。

## 使用場面

- HOSVDでfactor転置を掛けてcoreを作るとき。
- HOOIで他modeのfactorを使ってTensorを射影するとき。
- Tucker coreへfactorを掛け戻して再構成するとき。

## 処理の流れ（日本語）

1. **入力Tensorとmodeを検証する。**  
   `X`のshapeと`mode`がTensor演算の共通contractを満たすことを確認する。
2. **作用させる`matrix`が2次元か確認する。**  
   mode-n積は行列を1つのmodeへ作用させる処理なので、2次元以外は拒否する。
3. **行列の入力dimensionを確認する。**  
   `matrix.shape[1]` が `X.shape[mode]` と一致することを要求する。ここが一致しないと対象modeと行列積を作れない。
4. **`X`を対象modeでunfoldする。**  
   `unfold(X, mode)` により、対象modeを行方向へ出した2次元行列を作る。
5. **行列を左から掛ける。**  
   `matrix @ unfolded` を計算し、対象modeのdimensionを `matrix.shape[0]` へ写像する。
6. **出力Tensorのshapeを作る。**  
   元のshapeを基準に、対象modeだけを新しいdimensionへ置き換える。
7. **`fold()`でTensorへ戻す。**  
   行列積結果を出力shapeへfoldし、元と同じaxis順序のTensorとして返す。

### 処理フロー（短縮版）

```text
X / matrix / mode
→ 入力整合を検証
→ unfold(X, mode)
→ matrix @ unfolded
→ 対象modeだけ変更した出力shapeを作成
→ fold
→ mode-n積結果
```

## 主なcontract / 注意事項

- `matrix.shape[1] == X.shape[mode]` が必要。
- `unfold()`と`fold()`のaxis規約を共通利用するため、この3関数は一体として扱う。

## 関連API

`unfold`, `fold`, `hosvd`, `reconstruct_tucker`, `core_from_factors`
