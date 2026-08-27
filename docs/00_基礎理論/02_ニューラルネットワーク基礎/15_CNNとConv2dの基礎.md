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

- [[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]

---

## 1. 画像Tensorの次元

### 1枚の画像

グレースケール画像1枚は

$$
X\in\mathbb R^{C\times H\times W}
$$

で表す。

RGBなら $C=3$、グレースケールなら $C=1$。

### mini-batch

DataLoaderが $N$ 枚をまとめると、

$$
X\in\mathbb R^{N\times C\times H\times W}
$$

となる。

> [!important]
> `Conv2d` が2次元畳み込みであることと、入力Tensorが4次元であることは矛盾しない。`N` はサンプル軸、`C` はchannel軸であり、kernelがスライドする空間軸は `H,W` の2つである。

---

## 2. channelとは何か

RGB画像ではchannelは色成分を表すが、CNNの途中では学習によって得た特徴マップを表す。

```text
out_channels
=
出力特徴マップ数
```

と考えられる。

---

## 3. kernel / filter / weight の区別

`groups=1` の通常Conv2dでは、

$$
W
\in
\mathbb R^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_h\times K_w}
$$

である。

1つの出力channel $o$ を固定したfilterは

$$
W_o
\in
\mathbb R^{C_{\mathrm{in}}\times K_h\times K_w}
$$

で、全入力channelを見る。

したがって、

```text
filter
≈
ある1つの出力channelを作る重み集合

conv.weight
=
全filterをまとめた4階Tensor
```

と区別する。

---

## 4. `conv2.weight.shape = (64, 32, 3, 3)` の意味

```text
64 : 出力filter数
32 : 入力特徴マップ数
3  : kernel height
3  : kernel width
```

である。

1つの出力filterは

$$
32\times3\times3=288
$$

個のweightを持つ。

---

## 5. Conv2dの数式

入力を $X$、weightを $W$、biasを $b$ とする。

> [!important]
> PyTorchの `Conv2d` が実装する式は、数学でkernelを反転する狭義のconvolutionではなく、kernel indexをそのまま使う**cross-correlation**の形である。Deep Learningでは慣習的にこれもconvolutionと呼ぶ。

### 5.1 最も単純な場合

stride 1、dilation 1、padding 0なら、出力位置 $(p,q)$ は

$$
\boxed{
Y_{n,o,p,q}
=
b_o
+
\sum_{c=0}^{C_{\mathrm{in}}-1}
\sum_{a=0}^{K_h-1}
\sum_{b'=0}^{K_w-1}
W_{o,c,a,b'}
X_{n,c,p+a,q+b'}
}
$$

である。

ここではbias記号 $b_o$ とkernel width indexが衝突しないよう、width側のkernel indexを $b'$ と書いた。

### 5.2 stride / padding / dilationを含む一般形

strideを

$$
(S_h,S_w)
$$

paddingを

$$
(P_h,P_w)
$$

dilationを

$$
(D_h,D_w)
$$

とする。

zero paddingについて、元Tensorの範囲外では

$$
X_{n,c,h,w}=0
$$

とみなす。

すると、

$$
\boxed{
\begin{aligned}
Y_{n,o,p,q}
={}&b_o\\
&+
\sum_{c=0}^{C_{\mathrm{in}}-1}
\sum_{a=0}^{K_h-1}
\sum_{b'=0}^{K_w-1}
W_{o,c,a,b'}\\
&\qquad\times
X_{n,c,\,pS_h-P_h+aD_h,\,qS_w-P_w+b'D_w}
\end{aligned}
}
$$

となる。

`padding_mode` がzero以外なら、範囲外を0とする代わりに、そのpadding規則で拡張したTensor $\bar X=\mathcal P(X)$ を用いて

$$
Y_{n,o,p,q}
=
b_o+
\sum_{c,a,b'}
W_{o,c,a,b'}
\bar X_{n,c,\,pS_h+aD_h,\,qS_w+b'D_w}
$$

と書けばよい。

1出力位置は、

```text
入力channel × kernel領域
```

に対する積和である。

今回の `conv2` では1出力位置につき

$$
32\times3\times3=288
$$

個のweightを使い、この288がConv weightをSVD用行列へ変換するときの列数になる。

---

## 6. Conv2dの出力サイズを途中から導く

高さ方向を考える。

- 入力サイズ：$H$
- padding：$P$
- kernel size：$K$
- stride：$S$
- dilation：$D$
- 出力サイズ：$O$

とする。

kernelの最初と最後のサンプル位置の距離は

$$
D(K-1)
$$

なので、実効kernel sizeは

$$
\boxed{
K_{\mathrm{eff}}
=D(K-1)+1
}
$$

である。

padding後の有効入力長は

$$
H_{\mathrm{pad}}=H+2P.
$$

出力indexを

$$
p=0,1,\ldots,O-1
$$

とすると、最後のkernel開始位置は

$$
(O-1)S.
$$

そのkernel全体がpadding後入力へ収まる条件は

$$
(O-1)S+K_{\mathrm{eff}}
\le
H+2P.
$$

$K_{\mathrm{eff}}=D(K-1)+1$ を代入すると、

$$
(O-1)S+D(K-1)+1
\le
H+2P.
$$

整理して、

$$
(O-1)S
\le
H+2P-D(K-1)-1.
$$

したがって、

$$
O-1
\le
\frac{H+2P-D(K-1)-1}{S}.
$$

$O$ は整数なので、最大の許される個数は

$$
\boxed{
O
=
\left\lfloor
\frac{H+2P-D(K-1)-1}{S}
\right\rfloor
+1
}
$$

となる。

これは

$$
\boxed{
O
=
\left\lfloor
\frac{H+2P-D(K-1)-1}{S}+1
\right\rfloor
}
$$

と同値である。

`dilation=1` なら、

$$
O
=
\left\lfloor
\frac{H+2P-K}{S}
\right\rfloor+1.
$$

幅方向も同様に計算する。

### 空間サイズを維持する例

奇数kernel、stride 1、dilation 1で

$$
P=\frac{K-1}{2}
$$

とすると、

$$
\begin{aligned}
O
&=
\left\lfloor
H+2\frac{K-1}{2}-K
\right\rfloor+1\\
&=H.
\end{aligned}
$$

---

## 7. Poolingも空間サイズを変える

`MaxPool2d` も空間方向について同様に開始位置の個数を数える。

今回の

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

では

```text
28×28 → 14×14
14×14 → 7×7
```

となる。

Poolingには学習weightはない。

---

## 8. Fashion-MNIST CNNのshapeを追う

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

---

## 9. Conv2dのパラメータ数

`groups=1` ならweight数は

$$
\boxed{
P_W
=C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w
}
$$

である。

biasがあるなら

$$
P_b=C_{\mathrm{out}}
$$

なので

$$
\boxed{
P_{\mathrm{Conv}}
=C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w+C_{\mathrm{out}}
}
$$

となる。

`conv2` なら

$$
64\times32\times3\times3=18432
$$

にbias 64を加えて18496 parameters。

---

## 10. kernel size と patch size

`kernel_size` と `patch_size` は同じ概念ではない。

- `kernel_size`：Convが1回に見る局所領域
- `patch_size`：ViTなどで画像をtoken化するときの1patchの大きさ

ViTのpatch embeddingをConvで実装し

```python
kernel_size=patch_size
stride=patch_size
```

とすればpatchは重ならない。`stride < patch_size` なら重なる。

---

## 11. よくある混乱

### `Conv2d` は3次元層？

畳み込みの空間軸は2次元。入力Tensorはbatchとchannelを含めて4次元である。

### `3×3 filter` が64個という理解でよい？

少し不十分。`conv2` の1filterは `(32,3,3)` であり、32入力channel分の3×3 kernelを持つ。それが64filterある。

### SVDするのは特徴マップ？

違う。SVDするのは学習済み `conv.weight` である。

---

## 12. このノートで押さえるポイント

- `Conv2d` の入力は通常 `(N,C,H,W)`。
- PyTorchの式はkernelを反転しないcross-correlation型。
- 一般の参照位置はstride / padding / dilationをすべて含めて書ける。
- 出力サイズ式は「最後のkernelが収まる開始位置の個数」から導ける。
- 1filterは全入力channel分のkernelを持つ。
- `conv.weight` は全filterをまとめた4階Tensor。
- SVD圧縮するのは入力特徴マップではなく学習済みweight。

---

## 次に読むノート

- [[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]
- [[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]]
