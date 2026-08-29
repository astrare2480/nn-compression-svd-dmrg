# `truncated_svd`

**Stability:** A  
**定義:** `src/nn_compression/compression/svd.py`

## 責務

2次元行列のSVDを計算し、上位`rank`成分だけを返す。

## Signature

```python
truncated_svd(matrix: torch.Tensor, rank: int)
```

## 引数

- `matrix`: 2次元Tensor `(m, n)`。
- `rank`: 残す特異値数。`1 <= rank <= min(m, n)`。

## 戻り値

```text
U_r  : (m, rank)
S_r  : (rank,)
Vh_r : (rank, n)
```

## 使用場面

Linear/Conv SVD、HOSVD、HOOIのfactor更新など低rank分解の数学コア。

## ざっくりした処理

```text
matrix
→ rank validation
→ torch.linalg.svd(full_matrices=False)
→ U/S/Vhをrankまでslice
```

## 主なcontract / 注意事項

- 不正rankをclipしない。
- Python boolとTensor scalar rankを拒否。
- `operator.index()` 対応のNumPy整数scalarは受理。
- dtype/deviceはPyTorch SVDの結果として入力を引き継ぐ。

## 関連API

`retained_energy_from_matrix`, `factorize_linear_layer`, `factorize_conv2d_layer`, `hosvd`, `hooi_sweep`
