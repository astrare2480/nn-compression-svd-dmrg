---
title: Linear層2層置換の確認結果
tags:
  - SVD
  - 実装検証
  - 実験記録
---

# Linear層2層置換の確認結果

> [!info] このノートは、旧 `10_SVD実装/02_Linear層の2層置換.md` に混在していた実測・実行結果を分離したもの。
> 一般理論と実装方針は [[00_基礎理論/09_Linear層の2層置換_実装]] を参照する。

## 実コードで確認した2層置換

`03_linear_svd_two_layer_replacement.ipynb` では、rank 64に対して次のshapeを確認した。

```text
U_r.shape             = (512, 64)
S_r.shape             = (64,)
Vh_r.shape            = (64, 784)
first_layer.weight    = (64, 784)
second_layer.weight   = (512, 64)
```

元の計算は、

$$
y
=
Wx+b
$$

であり、SVD後は、

$$
x
\rightarrow
V_r^{\mathsf{T}}x
\rightarrow
U_r\Sigma_rV_r^{\mathsf{T}}x+b
$$

となる。

そのため、

```text
first_layer.weight  ← Vh_r
second_layer.weight ← U_r @ diag(S_r)
second_layer.bias   ← 元のbias
```

とする。

### rank 64の最小実験結果

| 指標 | 実測値 |
|---|---:|
| 出力MAE | 0.38189417 |
| 出力MSE | 0.22959685 |
| 最大絶対誤差 | 1.96176362 |
| 元Linearのパラメータ数 | 401,920 |
| 2層化後のパラメータ数 | 83,456 |
| 圧縮倍率 | 4.82× |

ここでの入力とLinear重みはランダムであり、MNISTの分類accuracyではない。  
目的は「denseな低ランク再構成」と「2層Linear」が同じ低ランク変換を表すことの確認である。

### 1層再構成版と2層圧縮版を分けた理由

`05_mnist_mlp_svd_compression.ipynb` では2種類のモデルを作っている。

```text
one-layer SVD model
元と同じshapeのdense Linearへ低ランク重みを再構成
→ 近似誤差だけを確認
→ パラメータ数は減らない
```

```text
two-layer SVD model
Vh_r と U_rΣ_r を2つのLinearとして保持
→ 近似誤差は同じ
→ パラメータ数と理論MACsが減る
```

この比較により、推論時間の差が「SVD近似そのもの」ではなく、「小さいLinearを2回呼ぶ実装形態」に由来するかを確認できる。

### 実装の安全性

初期のNotebookでは `.data` を使う箇所がある。  
最終版として推奨する形は次である。

```python
weight = layer.weight.detach()
U_r, S_r, Vh_r = SVD(weight, r)

with torch.no_grad():
    first_layer.weight.copy_(Vh_r)
    second_layer.weight.copy_(U_r @ torch.diag(S_r))
    if layer.bias is not None:
        second_layer.bias.copy_(layer.bias.detach())
```

- SVD用に読む：`detach()`
- 新しい層へ書く：`no_grad()` + `copy_()`
- biasは後段だけ
- 2層間にReLUを入れない

