---
title: Conv2d重みの行列化とSVD
aliases:
  - Conv重みのSVD
  - Conv2dの4次元重みを2次元化
  - Conv kernelの低ランク近似
  - Conv retained energy
tags:
  - CNN
  - Conv2d
  - SVD
  - 低ランク近似
  - PyTorch
  - NN圧縮
---

# Conv2d重みの行列化とSVD

## サマリー

Linear層のweightは最初から2次元行列なので、そのままSVDできる。

```text
Linear
weight.shape
=
(out_features, in_features)
```

一方、Conv2dのweightは通常4次元Tensorである。

```text
Conv2d
weight.shape
=
(out_channels, in_channels, kernel_height, kernel_width)
```

今回の `conv2` では、

```text
(64, 32, 3, 3)
```

である。

そこで、**出力filter軸を残し、残りの軸をflattenする**。

```text
(64, 32, 3, 3)
        ↓
(64, 32×3×3)
        ↓
(64, 288)
```

この `64×288` 行列へSVDを行う。

重要なのは、ここまでの `reshape / flatten` は、

```text
圧縮ではない
近似でもない
値を捨ててもいない
```

ということである。

単に、同じ18,432個のweightをSVDできる2次元行列として見直しているだけである。

---

## 対応コード

現在の汎用実装：

```text
src/nn_compression/compression/svd.py
src/nn_compression/compression/conv_svd.py
```

主要関数：

```python
from nn_compression.compression import (
    conv2d_weight_matrix,
    retained_energy,
    retained_energy_from_matrix,
    truncated_svd,
)
```

役割：

| 関数 | 役割 |
|---|---|
| `conv2d_weight_matrix` | Conv2d weightを `(out, -1)` へ行列化 |
| `truncated_svd` | 2次元行列の上位rank SVD |
| `retained_energy_from_matrix` | 2次元行列の特異値エネルギー保持率 |
| `retained_energy` | Linear / Conv2dのweightを適切に行列化してenergy計算 |

---

## 1. SVDするものは何か

最も重要な区別である。

入力特徴マップは、

```text
X
=
(N, 32, H, W)
```

である。

一方、`conv2` の学習済みweightは、

```text
W
=
(64, 32, 3, 3)
```

である。

今回SVDするのは、

```text
Xではない
↓
WをSVDする
```

である。

> [!important]
> 「特徴マップごとにSVDする」のではない。  
> **1つのConv2d層が持つ全filterのweightをまとめて行列化し、そのweight行列をSVDする。**

---

## 2. 1filterをflattenする

`conv2.weight` の先頭軸を出力filter番号とする。

```text
W.shape = (64, 32, 3, 3)
```

filter 0を固定すると、

```text
W[0].shape
=
(32, 3, 3)
```

である。

これをflattenすると、

```text
(32, 3, 3)
↓
(288,)
```

となる。

$$
32\times3\times3
=
288
$$

である。

同様に64filterすべてを行として並べる。

```text
filter 0  → [288個のweight]
filter 1  → [288個のweight]
filter 2  → [288個のweight]
   ⋮
filter63  → [288個のweight]
```

したがって、

```text
W_mat.shape
=
(64, 288)
```

となる。

---

## 3. PyTorchでは1行で行列化できる

```python
W_mat = conv.weight.detach().flatten(
    start_dim=1,
)
```

または、

```python
W_mat = conv.weight.detach().reshape(
    conv.weight.shape[0],
    -1,
)
```

でよい。

`start_dim=1` は、

```text
axis 0は残す
axis 1以降を全部flattenする
```

という意味である。

今回なら、

```text
(64, 32, 3, 3)
↓
(64, 288)
```

になる。

手動で64filterをloopして集める必要はない。

### 手動loopと`torch.stack`の対応

`flatten(start_dim=1)` が各出力filterを同じ行順で積むことは、連番Tensorで確認できる。

```python
import torch

# Conv weightと同じ4階Tensorを、値の追跡ができる連番で作る。
weight = torch.arange(64 * 32 * 3 * 3).reshape(64, 32, 3, 3)

# 各出力filterを1行へflattenし、第0軸へ順番に積む。
rows = []
for out_channel in range(weight.shape[0]):
    row = weight[out_channel].flatten()
    rows.append(row)

weight_matrix_loop = torch.stack(rows, dim=0)
weight_matrix_direct = weight.flatten(start_dim=1)

# 値・行順・shapeが1行で行列化した結果と一致する。
assert torch.equal(weight_matrix_loop, weight_matrix_direct)
assert weight_matrix_direct.shape == (64, 288)
```

`torch.stack(rows, dim=0)` は各1次元Tensorを新しい第0軸へ積むため、`weight[out_channel]` がそのまま第 `out_channel` 行になる。

---

## 4. `shape` とTensorを混同しない

```python
conv.weight.shape
```

はshape情報であり、値そのものではない。

```python
conv.weight
```

はweight Tensor / Parameterである。

`flatten()` を行う対象は、

```text
shape tuple
```

ではなく、

```text
weight Tensor
```

である。

```python
W_mat = conv.weight.detach().flatten(start_dim=1)
```

の結果 `W_mat` も1つの `torch.Tensor` である。

```text
W_mat.shape = (64, 288)
```

であって、64個のTensorを別々に保存しているわけではない。

---

## 5. reshape時点では何も圧縮していない

元のweight数は、

$$
64\times32\times3\times3
=
18{,}432
$$

である。

行列化後も、

$$
64\times288
=
18{,}432
$$

である。

したがって、

```text
4次元 → 2次元
```

にしても、パラメータ数は1個も減らない。

値の並び方・見方を変えただけである。

---

## 6. Conv weightを行列として見られる理由

ある出力位置で、入力から切り出す局所patchは、

```text
(C_in, K_h, K_w)
```

である。

`conv2` なら、

```text
(32, 3, 3)
```

である。

これもflattenすると、

```text
x_patch
=
(288,)
```

になる。

weight側を、

```text
W_mat
=
(64, 288)
```

とすると、その1位置での出力64チャネル分は概念的に、

$$
y
=
W_{\mathrm{mat}}x_{\mathrm{patch}}+b
$$

と見られる。

つまり、Convの局所積和を、

```text
64×288 行列
×
288次元の局所入力
```

というLinearに似た形で表せる。

このため、Linearで使った低ランクSVDの考え方をConv weightへ持ち込める。

---

## 7. Reduced SVD

行列化したweightを、

$$
W_{\mathrm{mat}}
\in
\mathbb{R}^{64\times288}
$$

とする。

Reduced SVDは、

$$
W_{\mathrm{mat}}
=
U\Sigma V^{\mathsf{T}}
$$

である。

PyTorchでは、

```python
U, S, Vh = torch.linalg.svd(
    W_mat,
    full_matrices=False,
)
```

とする。

shapeは、

```text
U  : (64, 64)
S  : (64,)
Vh : (64, 288)
```

である。

最大rankは、

$$
\min(64,288)
=
64
$$

である。

---

## 8. rank $r$ へ切り詰める

上位 $r$ 個だけ残す。

```python
U_r = U[:, :rank]
S_r = S[:rank]
Vh_r = Vh[:rank, :]
```

shapeは、

```text
U_r  : (64, r)
S_r  : (r,)
Vh_r : (r, 288)
```

である。

近似は、

$$
W_{\mathrm{mat}}
\approx
U_r\Sigma_rV_r^{\mathsf{T}}
$$

である。

`rank=16` なら、

```text
(64, 288)
≈
(64, 16)
×
(16, 16)
×
(16, 288)
```

となる。

切り詰めSVDは、指定rank以下の行列の中でFrobeniusノルム誤差を最小にする低ランク近似である。

---

## 9. 因子の名前よりshapeと積の順序を見る

SVD因子の中で $\Sigma$ をどちらの因子へ吸収するかには複数の書き方がある。

現在の実装では、

$$
F_{\mathrm{first}}
=
V_r^{\mathsf{T}}
$$

$$
F_{\mathrm{second}}
=
U_r\Sigma_r
$$

とする。

したがって、

$$
W_{\mathrm{mat}}
\approx
F_{\mathrm{second}}
F_{\mathrm{first}}
$$

である。

shapeは、

```text
first factor  : (r, 288)
second factor : (64, r)
```

である。

別資料では、これらを `A` / `B` と逆の名前で呼んだり、$\Sigma^{1/2}$ を両側へ分配したりすることがある。

> [!important]
> `A` / `B` という変数名そのものに意味を固定しない。  
> **どの因子が `(r,288)` で、どの因子が `(64,r)` なのか、そして積が `(64,288)` へ戻るか**を見る。

---

## 10. denseに復元するとshapeは元へ戻る

```python
W_approx_mat = (
    U_r @ torch.diag(S_r) @ Vh_r
)
```

とすれば、

```text
W_approx_mat.shape
=
(64, 288)
```

である。

さらに、

```python
W_approx_4d = W_approx_mat.reshape(
    64,
    32,
    3,
    3,
)
```

とすれば、

```text
(64, 32, 3, 3)
```

へ戻せる。

しかし、このdense weightを1つのConv2dとして保存すると、weight要素数は元と同じである。

```text
低rankに近似した値になった
≠
モデルのパラメータ数が減った
```

本当に圧縮するには、2つの小さい因子を**因子のまま**保持する必要がある。

詳しくは [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]。

---

## 11. full rankでは元weightを再現できる

rankを数学的最大rankまで残せば、切り捨てを行っていないため、理論上は元のweightを再構成できる。

今回のConv2d実装でも、full rank分解後の2層Convと元Convの出力が数値誤差範囲で一致することをテストしている。

```text
full rank
→
SVD実装・reshape・因子配置が正しいかを確認するテスト
```

である。

ただし、

```text
full rank
≠
必ず圧縮になる
```

点には注意する。

2因子として保持することで、full rankではむしろパラメータ数が増える場合がある。

---

## 12. retained energy

rank $r$ までの特異値が、weightのFrobeniusノルム二乗をどれだけ保持しているかを、

$$
E_r
=
\frac{
\sum_{i=1}^{r}\sigma_i^2
}{
\sum_i\sigma_i^2
}
$$

で測る。

`1` に近いほど、weight行列のエネルギーを多く保持している。

ただし、

```text
retained energyが高い
≠
分類accuracyが必ず高い
```

である。

SVDが最適化しているのはweight行列の近似であって、分類lossではない。

---

## 13. Conv2dのretained energyでは先に行列化する

Conv weightへ、

```python
torch.linalg.svdvals(
    conv.weight,
)
```

を4次元のまま適用してはいけない。

`torch.linalg.svdvals` は末尾2軸を行列として扱うため、

```text
(64, 32, 3, 3)
```

をそのまま渡すと、意図した、

```text
(64, 288)
```

のSVDではなく、`3×3` 行列群に対するbatched SVDとして扱われる。

その場合、

```python
singular_values[:rank]
```

の先頭軸も「特異値rank」だけを表さず、意図したretained energyにならない。

したがって、必ず、

```python
weight_matrix = conv.weight.detach().reshape(
    conv.weight.shape[0],
    -1,
)

singular_values = torch.linalg.svdvals(
    weight_matrix,
)
```

とする。

現在の `retained_energy()` はLinear / Conv2dの両方で、この行列化を行ってから計算する。

---

## 14. `detach()` と `torch.no_grad()`

### `detach()`

SVDへ渡す元weightを、元のautograd graphから切り離す。

```python
W_mat = conv.weight.detach().flatten(
    start_dim=1,
)
```

### `torch.no_grad()`

新しく作った層へSVD因子を `copy_()` するときに、重み代入をautogradで追跡しないために使う。

```python
with torch.no_grad():
    new_layer.weight.copy_(...)
```

役割は異なる。

```text
detach
→ SVDへ渡すTensorを計算graphから切る

no_grad
→ 新しいParameterへの代入を追跡しない
```

---

## 15. このノートで押さえるポイント

- SVD対象は入力特徴マップではなく `conv.weight`
- `conv2.weight=(64,32,3,3)` の各出力filterを `(288,)` にflattenして64行に積む
- PyTorchでは `flatten(start_dim=1)` だけで `(64,288)` を作れる
- flatten / reshapeだけでは圧縮も近似も起きない
- 局所Conv計算を `64×288` 行列と288次元patchの積として見られる
- `W_mat=(64,288)` の最大rankは64
- truncated SVDで初めて近似が起こる
- denseに復元して保存するとパラメータ数は減らない
- retained energyをConvで計算するときも、先に `(out,-1)` へ行列化する
- full rank一致は実装検証であり、圧縮成立を意味しない
- SVD因子の `A/B` 名ではなく、shapeと積の順序を見る

---

## 次に読むノート

- [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]
- [[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]]
- [[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]
