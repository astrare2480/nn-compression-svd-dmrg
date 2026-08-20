---
title: Global Average PoolingとCIFAR-10モデル設計
aliases:
  - GAP
  - Global Average Pooling
  - CIFAR10 CNN設計
  - GAPとSVD圧縮
tags:
  - CIFAR10
  - CNN
  - GAP
  - SVD
  - Conv2d
  - モデル設計
---

# Global Average PoolingとCIFAR-10モデル設計

## サマリー

Global Average Pooling（GAP）は、各特徴マップの空間方向を平均し、

```text
(N, C, H, W)
↓
(N, C, 1, 1)
↓ flatten
(N, C)
```

へ変換する処理である。

今回のCIFAR-10 CNNでは、3つのConvブロックの後にGAPを置いた。

```text
Input (N, 3, 32, 32)
↓ conv1 + ReLU + pool
(N, 32, 16, 16)
↓ conv2 + ReLU + pool
(N, 64, 8, 8)
↓ conv3 + ReLU + pool
(N, 128, 4, 4)
↓ GAP
(N, 128, 1, 1)
↓ flatten
(N, 128)
↓ fc1 128 → 256
↓ ReLU
↓ Dropout(0.5)
↓ fc2 256 → 10
(N, 10)
```

この設計は、巨大なFlatten→Linearを置くよりparameter効率がよく、**Conv層のSVD圧縮を主題にしやすい**。

---

# 1. GAPの計算

入力を、

$$
X \in \mathbb{R}^{N \times C \times H \times W}
$$

とする。

GAPは各sample・各channelについて空間方向を平均する。

$$
y_{n,c}
=
\frac{1}{HW}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
X_{n,c,h,w}
$$

したがって、各特徴マップはスカラー1個へ要約される。

例えば、

```text
(N, 128, 4, 4)
```

なら、128枚の4×4特徴マップをそれぞれ平均し、

```text
(N, 128)
```

の特徴ベクトルにする。

---

# 2. Flattenとの違い

GAPを使わずに、

```text
(N, 128, 4, 4)
↓ flatten
(N, 2048)
```

とすると、後段Linearへの入力次元が大きくなる。

例えば `Linear(2048, 256)` なら、weightだけで、

$$
2048 \times 256 = 524{,}288
$$

parametersを持つ。

一方、GAP後なら、

```text
Linear(128, 256)
```

なので、

$$
128 \times 256 = 32{,}768
$$

で済む。

つまりGAPは、分類ヘッドのparameter肥大化を強く抑える。

---

# 3. なぜ今回GAPを使ったか

Fashion-MNIST CNNでは巨大な `fc1` がparameter数の大半を占めた。

そのため、Linear SVDだけでparameter数を大きく削減できた。

CIFAR-10では、次の段階として**Conv層のrank allocation**を主題にしたかった。

そこでGAPを使い、巨大なFlatten→Linearがモデル全体を支配しすぎない構造にした。

今回のweight数は、biasを除くと、

```text
conv1 =    864
conv2 = 18,432
conv3 = 73,728
fc1   = 32,768
fc2   =  2,560
```

である。

特に `conv3` が大きく、複数ConvをSVD圧縮した効果をモデル全体で観測しやすい。

> [!important]
> GAPを使った理由は「GAP自体を圧縮するため」ではない。分類ヘッドを必要以上に巨大化させず、Conv圧縮を実験の中心に置くためのモデル設計上の選択である。

---

# 4. CIFAR-10モデルのshape

現在の `CIFAR10CNN` は、

```python
conv_channels = (32, 64, 128)
hidden_dim = 256
dropout = 0.5
```

を既定値とする。

## 入力

```text
(N, 3, 32, 32)
```

RGBなので `in_channels=3`。

## conv1

```text
Conv2d(3 → 32, 3×3, padding=1)
32×32 → 32×32
MaxPool2d(2)
32×32 → 16×16
```

出力：

```text
(N, 32, 16, 16)
```

## conv2

```text
Conv2d(32 → 64, 3×3, padding=1)
MaxPool2d(2)
```

出力：

```text
(N, 64, 8, 8)
```

## conv3

```text
Conv2d(64 → 128, 3×3, padding=1)
MaxPool2d(2)
```

出力：

```text
(N, 128, 4, 4)
```

## GAP

```text
(N, 128, 4, 4)
↓
(N, 128, 1, 1)
↓ flatten
(N, 128)
```

## classifier

```text
fc1: 128 → 256
ReLU
Dropout(0.5)
fc2: 256 → 10
```

最終出力はsoftmax前のlogits。

```text
(N, 10)
```

`CrossEntropyLoss` へはlogitsをそのまま渡す。

---

# 5. Dropoutの役割

`Dropout(0.5)` は学習時に一部の活性値をランダムに0へする正則化手法である。

重要なのは、

```text
model.train()
→ Dropout有効

model.eval()
→ Dropout無効
```

というモード差。

SVD圧縮前後の評価では、必ず `model.eval()` を使う。

rank candidateのFine-tuningではDropoutの乱数もcandidate比較へ影響し得るため、seed条件を揃える。

この点は [[00_基礎理論/04_実験設計/19_再現性と乱数管理]] とつながる。

---

# 6. GAPあり・なしとSVD実験

GAPには長所と短所がある。

## GAPあり

```text
Conv特徴マップ
↓ GAP
小さいfeature vector
↓ 小さいLinear
```

- parameter効率がよい
- Flatten後の巨大Linearを避けられる
- Conv圧縮の効果を見やすい
- 空間位置の詳細は平均によって要約される

## GAPなし

```text
Conv特徴マップ
↓ Flatten
大きいfeature vector
↓ 大きいLinear
```

- Linear parameterが大きくなりやすい
- Linear SVDの圧縮効果を観測しやすい
- モデルparameterがFC層に偏りやすい

どちらが「正しい」という話ではなく、**何を圧縮実験で観測したいか**で設計が変わる。

---

# 7. GAPとCAM / Grad-CAM

GAPは可視化手法でも登場するため、混同しやすい。

## CAM

古典的なCAMでは、モデルのforward中で特徴マップをGAPし、その後のLinear weightと対応させる構造を使う。

## Grad-CAM

Grad-CAMは、モデル自体にGAPが必須ではない。

Grad-CAMでは、あるclass scoreに対する特徴マップの勾配、

$$
\frac{\partial y^c}{\partial A_{ij}^k}
$$

を空間方向に平均し、channel重要度、

$$
\alpha_k^c
=
\frac{1}{Z}
\sum_i\sum_j
\frac{\partial y^c}{\partial A_{ij}^k}
$$

を作る。

したがって、

```text
モデル内GAP
≠
Grad-CAMで勾配をGAPする処理
```

である。

今回のSVD圧縮では可視化が主題ではないが、GAPという用語の意味を分けておく。

---

# 8. GAPとConv SVDの関係

GAPそのものには学習weightがない。

SVD圧縮対象は、

```text
conv1.weight
conv2.weight
conv3.weight
```

である。

Conv weight、

$$
W
\in
\mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}
$$

を、

$$
W_{mat}
\in
\mathbb{R}^{C_{out}\times(C_{in}K_hK_w)}
$$

へ行列化してSVDする。

詳細は、

- [[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]

を参照。

GAPによりclassifierを小さくしたことで、CIFAR-10実験では**3つのConvへrankをどう配るか**が中心課題になった。

---

# 9. Fashion-MNISTからCIFAR-10への設計上の発展

```text
Fashion-MNIST MLP
→ Linear SVD

Fashion-MNIST CNN
→ Linear SVD
→ Conv SVD
→ Conv + Linear同時圧縮

CIFAR-10 CNN + GAP
→ Conv1 / Conv2 / Conv3を対象
→ model-wide rank allocation
```

この段階で、

```text
1つの層を何rankにするか
```

から、

```text
複数層へrank budgetをどう配るか
```

へ問題が発展した。

---

# 関連

- [[00_基礎理論/02_ニューラルネットワーク基礎/15_CNNとConv2dの基礎]]
- [[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]
- [[30_CIFAR10_CNN/00_CIFAR10基礎/18_CIFAR10の前処理とDataLoader]]
- [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]
- [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]
