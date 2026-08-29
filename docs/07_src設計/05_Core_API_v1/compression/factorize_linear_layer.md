# `factorize_linear_layer`

**Stability:** A  
**定義:** `src/nn_compression/compression/linear_svd.py`

## 責務

1つのLinearをrank次元の2層LinearへSVD分解する。

## Signature

```python
factorize_linear_layer(
    layer: nn.Linear,
    rank: int,
) -> nn.Sequential
```

## 引数

- `layer`: 分解対象`nn.Linear`。
- `rank`: 中間dimension。

## 戻り値

```text
Linear(in_features → rank, bias=False)
→ Linear(rank → out_features, bias=original)
```

の`nn.Sequential`。

## 使用場面

Linearのparameter/MACsを実際に削減するSVD圧縮。

## ざっくりした処理

```text
layer.weight.detach()
→ truncated_svd
→ 1層目へ Vh_r
→ 2層目へ U_r @ diag(S_r)
→ biasを2層目へcopy
```

## 主なcontract / 注意事項

- 元layerは非破壊。
- device/dtype/requires_grad維持。
- 元weightとのautograd graphは初期化時に切る。

## 関連API

`truncated_svd`, `factorize_named_linear`, `factorize_named_layers`
