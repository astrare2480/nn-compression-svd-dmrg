# `tucker2_hooi`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_tucker.py`

## 責務

Conv2d weightのchannel mode 0/1へpartial HOOIを適用する。

## Signature

```python
tucker2_hooi(
    weight: torch.Tensor,
    rank_out: int,
    rank_in: int,
    max_iter: int = 30,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-5,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], list[float]]
```

## 引数

- `weight`: `(C_out,C_in,kH,kW)`の実数浮動小数点Tensor。
- `rank_out`, `rank_in`: channel ranks。
- iteration/tolerance引数はgeneric `hooi()` と同じ意味。

## 戻り値

`(core, factors, history)`。

## 使用場面

同rankのHOSVD Tucker-2よりweight再構成誤差を反復改善したいとき。

## ざっくりした処理

```text
rank validation
→ ranks={0: rank_out, 1: rank_in}
→ generic hooiへ委譲
```

## 主なcontract / 注意事項

- **Tensor-level APIなので入力weightをdetachしない。**
- Moduleへ変換するときのdetachは`build_tucker2_conv()`側の責務。
- `history[0]`はHOSVD初期誤差。

## 関連API

`hooi`, `tucker2_hooi_sweep`, `build_tucker2_conv_from_components`
