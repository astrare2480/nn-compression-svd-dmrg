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
- `shape`: 復元後Tensorのshape。`torch.Size` も利用可能。

## 戻り値

`shape` と同じshapeを持つTensor。

## 使用場面

mode-n積の結果をTensorへ戻すとき、unfold/foldの可逆性を確認するとき。

## ざっくりした処理

```text
unfolded
→ shape / row / column / numel整合確認
→ modeを先頭にした一時shapeへreshape
→ axisを元位置へmovedim
```

## 主なcontract / 注意事項

- `unfolded.ndim == 2`。
- 転置された行列を「要素数が同じだから」と黙ってreshapeしない。
- `shape` は2次元以上・各dimension正。

## 関連API

`unfold`, `mode_dot`
