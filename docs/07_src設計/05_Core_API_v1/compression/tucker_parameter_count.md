# `tucker_parameter_count`

**Stability:** A  
**定義:** `src/nn_compression/compression/tucker.py`

## 責務

指定rankのTucker表現を保存するために必要なcore + factorの総要素数を理論計算する。

## Signature

```python
tucker_parameter_count(
    shape: tuple[int, ...],
    ranks: dict[int, int],
) -> int
```

## 引数

- `shape`: 元Tensorのshape。
- `ranks`: `{mode: rank}`。未指定modeはcoreに元dimensionを残し、factorを持たない。

## 戻り値

```text
core要素数 + Σ(I_mode * rank_mode)
```

の整数。

## 使用場面

Tucker rank候補の理論parameter量・保存量を比較するとき。

## 処理の流れ（日本語）

1. **shapeとranksを共通Tucker contractで検証する。**
2. **coreの要素数を1から計算する。**  
   各modeについて、圧縮対象なら`rank`、未圧縮なら元dimensionを掛ける。
3. **factorの要素数を加算する。**  
   圧縮対象modeごとにfactor shape `(I_mode, rank_mode)`の要素数`I_mode * rank_mode`を足す。
4. **core要素数と全factor要素数を合計して返す。**

### 処理フロー（短縮版）

```text
shape / ranks検証
→ core shapeの各dimensionを決定
→ core要素数を積算
→ 圧縮modeごとのfactor要素数を加算
→ total
```

## 主なcontract / 注意事項

モデル全体のParameter数ではなく、**1 TensorのTucker表現**の要素数。biasや周辺layerは含まない。

## 関連API

`parameter_ratio`, `compression_factor`
