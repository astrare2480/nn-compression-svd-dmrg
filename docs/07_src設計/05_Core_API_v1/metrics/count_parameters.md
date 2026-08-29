# `count_parameters`

**Stability:** A  
**定義:** `src/nn_compression/metrics/model_comparison.py`

## 責務

modelが保持するParameterの総要素数を数える。

## Signature

```python
count_parameters(model) -> int
```

## 引数

`model`: `nn.Module`。

## 戻り値

全`model.parameters()`の`numel()`合計。

## 使用場面

baseline/compressed modelのサイズ比較。

## ざっくりした処理

全Parameterを走査して要素数を加算する。

## 主なcontract / 注意事項

`requires_grad=False`のParameterも数える。

## 関連API

`parameters_reduction`, `collect_compression_metrics`
