# `tucker2_effective_weight`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_tucker.py`

## 責務

3層Tucker-2 Convから、等価な1つの4階Conv weightを再構成する評価utility。

## Signature

```python
tucker2_effective_weight(seq: nn.Sequential) -> torch.Tensor
```

## 引数

Tucker-2構造の3層`nn.Sequential`。

## 戻り値

`(C_out, C_in, kH, kW)` のeffective weight Tensor。

## 使用場面

Fine-tuning前後のTucker-2 Moduleを元Conv weightとrelative Frobenius errorで比較するとき。

## ざっくりした処理

```text
3層構造をvalidation
→ input/core/output weightをdetach
→ U_in/core/U_outとして取り出す
→ reconstruct_tucker
```

## 主なcontract / 注意事項

- 1x1 → core → 1x1の構造・channel接続・projection設定を検証。
- 評価utilityなのでeffective weight経由のgradient計算は目的にしない。

## 関連API

`build_tucker2_conv_from_components`, `reconstruct_tucker`, `relative_frobenius_error`
