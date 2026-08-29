# `make_one_layer_svd_model`

**Stability:** B  
**定義:** `src/nn_compression/compression/mlp_svd.py`

## 責務

現行MLPの`fc1`/`fc2`を、shapeを変えないtruncated-SVD再構築Linearへ差し替える。

## Signature

```python
make_one_layer_svd_model(
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
- `r1`, `r2`: historical Notebook互換keyword。

## 戻り値

`fc1`/`fc2`を低rank近似weightへ置換したdeepcopy model。

## 使用場面

2層化によるparameter削減を行う前に、weight近似だけがaccuracyへ与える影響を確認するとき。

## ざっくりした処理

```text
rank名解決
→ model deepcopy
→ fc1/fc2 weightをSVD
→ rebuild_linear_from_svdで同shape Linear作成
```

## 主なcontract / 注意事項

- `fc1_rank`と`r1`等の同時指定は不可。
- `fc3`を含めbaselineとのParameter共有なし。
- generic MLP APIではなく実験支援API。

## 関連API

`make_two_layer_svd_model`, `rebuild_linear_from_svd`
