# `make_two_layer_svd_model`

**Stability:** B  
**定義:** `src/nn_compression/compression/mlp_svd.py`

## 責務

現行MLPの`fc1`/`fc2`を、それぞれrank次元の2層Linearへ置換する。

## Signature

```python
make_two_layer_svd_model(
    model,
    fc1_rank=None,
    fc2_rank=None,
    *,
    r1=None,
    r2=None,
)
```

## 引数

- `model`: `fc1`, `fc2`を持つ現行MLP。
- `fc1_rank`, `fc2_rank`: primary rank名。
- `r1`, `r2`: legacy keyword。

## 戻り値

`fc1`/`fc2`をfactorized Linearへ置換したdeepcopy model。

## 使用場面

MNIST/Fashion-MNIST MLPの実圧縮、rank選択、Fine-tuning。

## ざっくりした処理

```text
rank名解決
→ model deepcopy
→ baseline fc1/fc2をfactorize_linear_layer
→ copyへ代入
```

## 主なcontract / 注意事項

- `fc3`を含めbaselineとParameterを共有しない。
- rankのprimary名とlegacy名は同時指定不可。

## 関連API

`factorize_linear_layer`, `make_one_layer_svd_model`, `estimate_mlp_macs`
