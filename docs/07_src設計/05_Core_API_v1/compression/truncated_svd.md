# `truncated_svd`

**Stability:** A  
**定義:** `src/nn_compression/compression/svd.py`

## 責務

2次元行列に対して特異値分解（SVD）を行い、上位`rank`成分だけを残した低rank表現を返す。Linear SVD、Conv SVD、HOSVD、HOOIから共通利用されるSVDの数学コア。

## Signature

```python
truncated_svd(matrix: torch.Tensor, rank: int)
```

## 引数

- `matrix`: 分解対象の2次元Tensor `(m, n)`。
- `rank`: 残す特異値・特異ベクトルの数。`1 <= rank <= min(m, n)`。

## 戻り値

```text
U_r  : (m, rank)
S_r  : (rank,)
Vh_r : (rank, n)
```

近似は `matrix ≈ U_r @ diag(S_r) @ Vh_r`。

## 使用場面

- Linear/Conv2d weightを低rank分解するとき。
- HOSVDでmode-n unfoldingからfactorを求めるとき。
- HOOIで1つのfactorを更新するとき。

## 処理概要

1. **入力が2次元行列か確認する。**  
   SVDの最大rankを求める前に`matrix.ndim == 2`を要求する。Tensor全体を直接扱う関数ではなく、行列に対する基本処理として境界を固定する。
2. **指定rankを整数として検証する。**  
   PythonのboolやTensor scalarを整数rankとして受け入れず、`operator.index()`で整数scalarとして扱える値だけを受理する。
3. **数学的に可能なrank上限を確認する。**  
   `min(m, n)`を最大rankとし、0・負数・上限超過を黙ってclipせずエラーにする。
4. **economy-size SVDを計算する。**  
   `torch.linalg.svd(matrix, full_matrices=False)`で`U, S, Vh`を求める。不要な完全直交基底を作らない。
5. **上位rank成分だけを切り出す。**  
   `U[:, :rank]`, `S[:rank]`, `Vh[:rank, :]`を取得する。
6. **低rank成分をそのまま返す。**  
   この関数では再構成やNN Module化を行わず、後続APIが用途に応じてfactorを利用する。

## 主なcontract / 注意事項

- 不正rankを自動補正しない。
- Python boolとTensor scalar rankを拒否する。
- `operator.index()`対応のNumPy整数scalarは受理する。
- dtype/deviceはPyTorch SVDの演算結果として入力から引き継がれる。

## 関連API

`retained_energy_from_matrix`, `factorize_linear_layer`, `factorize_conv2d_layer`, `hosvd`, `hooi_sweep`
