---
title: SVD実装の確認結果
tags:
  - SVD
  - 実装検証
  - 実験記録
---

# SVD実装の確認結果

> [!info] このノートは、旧 `10_SVD実装/01_Linear層のSVD実装.md` に混在していた実測・実行結果を分離したもの。
> 一般理論と実装方針は [[00_基礎理論/08_Linear層のSVD実装]] を参照する。

## 実コードで確認したshapeと誤差

### Reduced SVDのshape

`00_pytorch_linear_svd.ipynb` では、

```text
Linear(784, 512)
weight.shape = (512, 784)
```

に対して、

```text
U.shape  = (512, 512)
S.shape  = (512,)
Vh.shape = (512, 784)
```

となることを確認した。

これは、

$$
k
=
\min(512,784)
=
512
$$

としてReduced SVDを行った結果である。

### rank 64の重み近似

`01_linear_low_rank_approximation.ipynb` のランダム初期化Linear層では、rank 64の再構成誤差は次のとおりだった。

| 指標 | 実測値 |
|---|---:|
| 重みMSE | 0.2295968533 |
| 最大絶対誤差 | 1.9617638588 |

この値はMNIST学習済み重みの結果ではなく、ランダム初期化された `Linear(784,512)` を使った基礎確認である。

### rank sweep

`02_rank_error_compression_tradeoff.ipynb` の出力は次のとおりである。

| rank | 重みMSE | 2因子のパラメータ数 | 元重みとの圧縮倍率 |
|---:|---:|---:|---:|
| 512 | 8.869e-13 | 664,064 | 0.61× |
| 256 | 0.056970 | 332,288 | 1.21× |
| 128 | 0.152667 | 166,400 | 2.42× |
| 64 | 0.229597 | 83,456 | 4.82× |
| 32 | 0.274677 | 41,984 | 9.57× |
| 16 | 0.303261 | 21,248 | 18.92× |
| 8 | 0.319436 | 10,880 | 36.94× |

rank 512では近似誤差はほぼ0だが、2因子の総要素数は元の重み行列より多くなる。  
したがって、**数学的な最大rankと、圧縮が成立するrankは別**である。

### `detach()` と `no_grad()` の使い分け

最終版PDFで整理した結論を、実装へ対応させると次のようになる。

```python
U_r, S_r, Vh_r = SVD(layer.weight.detach(), r)
```

ここでは重みを読み取り、SVDへ渡しているだけなので `detach()` を使う。

```python
with torch.no_grad():
    first_layer.weight.copy_(Vh_r)
```

ここではParameterを書き換えるため、`torch.no_grad()` と `copy_()` を使う。

```text
読み取り・分解       → detach()
Parameterへの書込み → no_grad() + copy_()
推奨しない           → weight.data = ...
```

初期Notebookの `.data` は学習過程の記録として残っているが、最終実装としては `detach()` と `no_grad() + copy_()` を採用する。

