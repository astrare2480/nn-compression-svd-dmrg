---
title: CNNとConv2dの基礎
aliases:
  - CNNの基礎
  - Conv2dの基礎
  - 畳み込み層のshape
  - カーネルとフィルタとチャネル
  - Conv出力サイズ
tags:
  - CNN
  - Conv2d
  - PyTorch
  - Fashion-MNIST
  - 畳み込み
---

# CNNとConv2dの基礎

## サマリー

CNN（Convolutional Neural Network）は、画像全体を最初から1本のベクトルとして扱うのではなく、**局所領域へ同じ重みを繰り返し適用する畳み込み**によって特徴を抽出する。

PyTorchの `nn.Conv2d` では、バッチを含む入力Tensorを通常、

```text
(N, C, H, W)
```

で扱う。

- `N`：batch size
- `C`：channel数
- `H`：height
- `W`：width

`Conv2d` が「2次元」なのはTensor全体が2次元だからではない。**カーネルを移動させる空間軸が `H` と `W` の2軸だから**である。

今回のFashion-MNIST CNNでは、

```text
入力                (N, 1, 28, 28)
  ↓ Conv2d(1 → 32, 3×3, padding=1)
conv1出力           (N, 32, 28, 28)
  ↓ MaxPool2d(2)
pool1出力           (N, 32, 14, 14)
  ↓ Conv2d(32 → 64, 3×3, padding=1)
conv2出力           (N, 64, 14, 14)
  ↓ MaxPool2d(2)
pool2出力           (N, 64, 7, 7)
  ↓ Flatten
                    (N, 3136)
  ↓ Linear(3136 → 128)
                    (N, 128)
  ↓ Linear(128 → 10)
                    (N, 10)
```

となる。

Conv-SVDへ進む前に、次の区別を明確にする。

```text
入力特徴マップ X
≠
Conv2dが持つ学習済みweight W
```

SVD圧縮の対象は、入力画像や特徴マップではなく**Conv2dのweight（フィルタ群）**である。

---

## 対応コード

現在の実装では、Fashion-MNIST用CNNは次に置く。

```text
src/nn_compression/models/cnn.py
```

公開クラス：

```python
from nn_compression.models import FashionMNISTCNN
```

Conv-SVDそのものは次のノートで扱う。

- [[00_基礎理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/17_Conv2dの低ランク2層置換]]

---

## 1. 画像Tensorの次元

### 1枚の画像

グレースケールのFashion-MNIST 1枚は、

```text
(C, H, W)
=
(1, 28, 28)
```

である。

RGB画像なら、

```text
(3, H, W)
```

となる。

### mini-batch

DataLoaderが `N` 枚をまとめると、

```text
(N, C, H, W)
```

となる。

たとえばbatch size 64なら、

```text
(64, 1, 28, 28)
```

である。

> [!important]
> `Conv2d` が2次元畳み込みであることと、入力Tensorが4次元であることは矛盾しない。  
> `N` はサンプル軸、`C` はチャネル軸であり、カーネルがスライドする空間軸は `H` と `W` である。

---

## 2. チャネルとは何か

### 入力画像では色の層

RGB画像では、

```text
R
G
B
```

の3チャネルがある。

Fashion-MNISTはグレースケールなので、入力チャネル数は1である。

### CNNの途中では特徴の層

`conv1` が32チャネルを出力する場合、32枚の特徴マップが生成される。

```text
conv1 output
(N, 32, H, W)
```

ここで各チャネルは、色ではなく、学習によって獲得された異なる特徴を表す。

したがって、

```text
out_channels
=
出力特徴マップ数
```

と考えてよい。

ただし「特徴量」という言葉は、1つのスカラー値、1枚の特徴マップ、特徴ベクトル全体などを指す場合があるため、shapeで確認する。

---

## 3. kernel / filter / weight の区別

用語は資料によって多少揺れるため、このノートでは次の意味で使う。

### kernel size

```text
3×3
5×5
```

など、**空間的に一度に見る範囲の大きさ**を指す。

### 1つの出力filter

入力チャネル数が `C_in` のConvでは、1つの出力チャネルを作るfilterは、全入力チャネルを見る必要がある。

したがって `3×3` Convの1filterは、

```text
(C_in, 3, 3)
```

の重みを持つ。

### Conv2d全体のweight

出力チャネル数が `C_out` なら、そのfilterが `C_out` 個ある。

`groups=1` の通常のConv2dでは、

```text
weight.shape
=
(C_out, C_in, K_h, K_w)
```

である。

つまり、

```text
filter
≈
ある1つの出力チャネルを作るための重み集合

weight
=
全出力filterをまとめたTensor
```

と整理すると分かりやすい。

> [!note]
> 「filter = weight」と言う説明もあるが、厳密には `conv.weight` は全filterをまとめたTensorである。  
> 1filterだけなら `conv.weight[o]` に対応する。

---

## 4. `conv2.weight.shape = (64, 32, 3, 3)` の意味

今回の `conv2` は、

```python
nn.Conv2d(
    in_channels=32,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
)
```

である。

重みshapeは、

```text
(out_channels, in_channels, kernel_height, kernel_width)
=
(64, 32, 3, 3)
```

となる。

意味は、

```text
64 : 出力filter数
32 : 入力特徴マップ数
3  : kernel height
3  : kernel width
```

である。

1つの出力filterを固定すると、

```text
conv2.weight[0].shape
=
(32, 3, 3)
```

である。

中には、

```text
入力channel 0 用の 3×3 kernel
入力channel 1 用の 3×3 kernel
...
入力channel31 用の 3×3 kernel
```

が含まれる。

これら32チャネル分の畳み込み結果を加算し、biasを加えることで、1枚の出力特徴マップを作る。

---

## 5. Conv2dの数式

入力を $X$、weightを $W$、biasを $b$ とする。

簡略化して、stride 1・dilation 1の場合、出力チャネル $o$ の位置 $(i,j)$ は、

$$
Y_{n,o,i,j}
=
b_o
+
\sum_{c=0}^{C_{\mathrm{in}}-1}
\sum_{u=0}^{K_h-1}
\sum_{v=0}^{K_w-1}
W_{o,c,u,v}
X_{n,c,i+u,j+v}
$$

という形の積和で作られる。

padding・strideを含めると参照位置は変化するが、重要なのは、

```text
1つの出力位置
=
入力channel × kernel領域に対する積和
```

という点である。

`conv2` では1出力位置につき、

$$
32\times3\times3
=
288
$$

個のweightを使う。

この `288` が、後でConv weightをSVD用行列へ変換するときの列数になる。

---

## 6. Conv2dの出力サイズ

高さ方向を例にする。

- 入力サイズ：$H$
- padding：$P$
- kernel size：$K$
- stride：$S$
- dilation：$D$
- 出力サイズ：$O$

とする。

実効kernel sizeは、

$$
K_{\mathrm{eff}}
=
D(K-1)+1
$$

である。

したがって、PyTorchの通常のConv2d出力サイズは、

$$
\boxed{
O
=
\left\lfloor
\frac{
H+2P-D(K-1)-1
}{S}
+1
\right\rfloor
}
$$

となる。

`dilation=1` なら、

$$
O
=
\left\lfloor
\frac{H+2P-K}{S}
\right\rfloor
+1
$$

である。

幅方向も同様に計算する。

### なぜ `+1` が最後に来るか

カーネル左端の開始位置を、

```text
0, S, 2S, ..., (O-1)S
```

と数える。

最後のkernelが入力範囲へ収まる条件から、置ける**開始位置の個数**を数えるため、割り算の後に `+1` が付く。

### 例：paddingなし

```text
H = 28
K = 3
S = 1
P = 0
```

なら、

$$
\left\lfloor
\frac{28-3}{1}
\right\rfloor
+1
=
26
$$

なので、

```text
28 → 26
```

となる。

### 例：`padding=1`

```text
H = 28
K = 3
S = 1
P = 1
```

なら、

$$
\left\lfloor
\frac{28+2-3}{1}
\right\rfloor
+1
=
28
$$

なので、空間サイズを維持できる。

一般に奇数kernel・stride 1・dilation 1でサイズを維持するpaddingは、

$$
P
=
\frac{K-1}{2}
$$

である。

```text
kernel 3 → padding 1
kernel 5 → padding 2
kernel 7 → padding 3
```

---

## 7. Poolingも空間サイズを変える

`MaxPool2d` も、空間方向の出力サイズについては同様の考え方で計算できる。

今回の、

```python
nn.MaxPool2d(
    kernel_size=2,
    stride=2,
)
```

では、

```text
28×28 → 14×14
14×14 → 7×7
```

となる。

Poolingには学習するweightはなく、主に空間サイズを縮小する役割を持つ。

---

## 8. Fashion-MNIST CNNのshapeを追う

現在の `FashionMNISTCNN` は次の構造である。

```text
Input
(N, 1, 28, 28)

↓ conv1: Conv2d(1, 32, 3, padding=1)
(N, 32, 28, 28)

↓ ReLU
(N, 32, 28, 28)

↓ pool1: MaxPool2d(2, 2)
(N, 32, 14, 14)

↓ conv2: Conv2d(32, 64, 3, padding=1)
(N, 64, 14, 14)

↓ ReLU
(N, 64, 14, 14)

↓ pool2: MaxPool2d(2, 2)
(N, 64, 7, 7)

↓ Flatten
(N, 64×7×7)
=
(N, 3136)

↓ fc1: Linear(3136, 128)
(N, 128)

↓ ReLU
(N, 128)

↓ fc2: Linear(128, 10)
(N, 10)
```

`64×7×7=3136` が `fc1.in_features` になる。

---

## 9. Conv2dのパラメータ数

`groups=1` なら、weight数は、

$$
C_{\mathrm{out}}
C_{\mathrm{in}}
K_hK_w
$$

である。

biasがあるなら、

$$
C_{\mathrm{out}}
$$

を加える。

### `conv2` の例

$$
64\times32\times3\times3
=
18{,}432
$$

bias 64を加えると、

$$
18{,}432+64
=
18{,}496
$$

パラメータである。

### `conv1` の例

$$
32\times1\times3\times3+32
=
320
$$

である。

このため、今回のCNNでは、Conv-SVDを最初に試す対象として `conv1` より `conv2` の方が効果を観察しやすい。

---

## 10. kernel size と patch size

`kernel_size` と `patch_size` は同じ概念ではない。

- `kernel_size`：Convが1回に見る局所領域
- `patch_size`：ViTなどで画像をトークン化するときの1patchの大きさ

ただしViTのpatch embeddingをConvで実装するときは、典型的に、

```python
nn.Conv2d(
    in_channels=C,
    out_channels=embed_dim,
    kernel_size=patch_size,
    stride=patch_size,
)
```

とする。

この場合、

```text
kernel_size = patch_size
stride      = patch_size
```

なのでpatchが重ならない。

`stride < patch_size` ならpatchが重なる。

---

## 11. よくある混乱

### `Conv2d` は3次元層？

畳み込みの空間軸は2次元。入力Tensorはbatchとchannelを含めて4次元である。

### `3×3 filter` が64個という理解でよい？

少し不十分。`conv2` の1filterは `(32,3,3)` であり、32入力チャネル分の `3×3` kernelを持つ。それが64filterある。

### filter数 = 特徴マップ数？

1つのfilterが1つの出力チャネルを作るため、`out_channels` と出力特徴マップ数は一致する。

### SVDするのは特徴マップ？

違う。SVDするのは学習済み `conv.weight` である。詳しくは [[00_基礎理論/16_Conv2d重みの行列化とSVD]]。

---

## 12. このノートで押さえるポイント

- `Conv2d` の入力は通常 `(N,C,H,W)`
- `Conv2d` が2次元なのは空間軸が `H,W` の2つだから
- channelは画像では色、CNN途中では特徴マップを表す
- `kernel_size` は空間領域の大きさ
- 1filterは全入力チャネル分のkernelを持つ
- `conv.weight` は全filterをまとめたTensor
- `conv2.weight=(64,32,3,3)` は64filter × 32入力channel × 3×3を意味する
- Convの出力サイズはkernel / stride / padding / dilationで決まる
- 今回の `conv2` では1出力位置の局所入力次元が `32×3×3=288`
- SVD圧縮するのは入力特徴マップではなく学習済みweight

---

## 次に読むノート

- [[00_基礎理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/17_Conv2dの低ランク2層置換]]
- [[00_基礎理論/11_理論計算量とベンチマーク]]
