# `fold`

**Stability:** A  
**定義:** `src/nn_compression/tensor/operations.py`

## 責務

`unfold()` の規約で展開された2次元行列を、指定shape・axis配置のTensorへ戻す。

## Signature

```python
fold(
    unfolded: torch.Tensor,
    mode: int,
    shape: tuple[int, ...],
) -> torch.Tensor
```

## 引数

- `unfolded`: mode-n unfolding済みの2次元Tensor。
- `mode`: unfolding時に行方向へ置いたmode。
- `shape`: 復元後Tensorのshape。`tuple`だけでなく`torch.Size`も利用可能。

## 戻り値

`shape` と同じshape・元のaxis配置を持つTensor。

## 使用場面

- `unfold()` の逆変換を行うとき。
- `mode_dot()`で行列積を行った結果をTensorへ戻すとき。
- Tensor演算のshape規約が正しいか検証するとき。

## 処理概要

1. **復元先shapeを通常のtupleへ変換する。**  
   `torch.Size`などを含めて同じ扱いにし、以後のshape計算を単純化する。
2. **復元先shapeとmodeを検証する。**  
   shapeが2次元以上・各dimension正であること、`mode`が有効範囲内であることを確認する。
3. **`unfolded`が本当に2次元行列か確認する。**  
   mode-n unfoldingの逆変換なので、1次元や3次元以上の入力は受け付けない。
4. **行数が対象modeのdimensionと一致するか確認する。**  
   `unfolded.shape[0] == shape[mode]` を要求する。
5. **総要素数と列数を確認する。**  
   指定shapeの総要素数を計算し、`unfolded.numel()`と一致するか、さらに列数が「総要素数 / 対象modeサイズ」と一致するかを確認する。
6. **modeを先頭に置いた一時shapeへreshapeする。**  
   `unfold()`直後と同じaxis配置を再現してからTensorへ戻す。
7. **先頭axisを元のmode位置へ戻す。**  
   `movedim(0, mode)` を使い、最終的な元axis配置を復元して返す。

## 主なcontract / 注意事項

- `unfolded.ndim == 2` が必要。
- 転置された行列を「要素数が同じだから」と黙ってreshapeしない。
- `shape` は2次元以上・各dimension正。

## 関連API

`unfold`, `mode_dot`
