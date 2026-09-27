# `make_one_layer_svd_model`

**Stability:** B  
**定義:** `src/nn_compression/compression/mlp_svd.py`

## 責務

現行MLPの`fc1`・`fc2`を、shapeを変えないtruncated-SVD近似Linearへ差し替えたmodel copyを作る。parameter削減ではなくweight近似だけの影響を確認するhistorical実験支援API。

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

- `model`: `fc1`, `fc2`, `fc3`を持つ現行MLP。
- `fc1_rank`, `fc2_rank`: primaryなrank名。
- `r1`, `r2`: historical Notebook互換のlegacy名。

## 戻り値

`fc1`・`fc2`を低rank近似weightの単一Linearへ差し替えたmodel copy。

## 使用場面

2層分解によるparameter削減を入れる前に、「weightをrank制限したこと自体」がaccuracyへ与える影響を確認するとき。

## 処理概要

1. **primary rank名とlegacy名を解決する。**  
   `fc1_rank`と`r1`などを同時指定した場合は曖昧なのでエラーにする。両層のrankが最終的に必須。
2. **model全体を`deepcopy`する。**  
   `fc3`を含めbaselineとParameterを共有しないcopyを作る。
3. **元modelの`fc1.weight`をdetachしてtruncated SVDする。**
4. **SVD成分から元shapeの単一Linearを再構築する。**  
   `rebuild_linear_from_svd()`を使い、copy側`fc1`へ代入する。
5. **`fc2`も同じ手順で再構築する。**
6. **shapeは元と同じままの近似modelを返す。**  
   層数・Parameter shapeを変えないため、これは実圧縮ではない。

## 主なcontract / 注意事項

- generic MLP APIではなく、現行`fc1/fc2`構造を前提とする。
- `r1/r2`は互換用。新規コードでは`fc1_rank/fc2_rank`を使う。

## 関連API

`truncated_svd`, `rebuild_linear_from_svd`, `make_two_layer_svd_model`
