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

PyTorchの確認コード：[[05_SVD基礎実装検証/15_CNNのPyTorch確認コード#PyTorch確認-001]]

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

### filterを、添字と全9要素で見る

出力channel $o$ と入力channel $c$ を1つずつ固定すると、対応する $3\times3$ kernelは

$$
W_{o,c,:,:}
=
\begin{pmatrix}
W_{o,c,0,0} & W_{o,c,0,1} & W_{o,c,0,2}\\
W_{o,c,1,0} & W_{o,c,1,1} & W_{o,c,1,2}\\
W_{o,c,2,0} & W_{o,c,2,1} & W_{o,c,2,2}
\end{pmatrix}.
$$

入力側の対応する局所領域を

$$
X_{n,c,p:p+3,q:q+3}
=
\begin{pmatrix}
X_{n,c,p,q} & X_{n,c,p,q+1} & X_{n,c,p,q+2}\\
X_{n,c,p+1,q} & X_{n,c,p+1,q+1} & X_{n,c,p+1,q+2}\\
X_{n,c,p+2,q} & X_{n,c,p+2,q+1} & X_{n,c,p+2,q+2}
\end{pmatrix}
$$

とすると、この入力channelから出力位置 $(p,q)$ への寄与は、全9要素を同じ位置どうしで掛けて

$$
\begin{aligned}
s_{n,o,c,p,q}
={}&W_{o,c,0,0}X_{n,c,p,q}
+W_{o,c,0,1}X_{n,c,p,q+1}
+W_{o,c,0,2}X_{n,c,p,q+2}\\
&+W_{o,c,1,0}X_{n,c,p+1,q}
+W_{o,c,1,1}X_{n,c,p+1,q+1}
+W_{o,c,1,2}X_{n,c,p+1,q+2}\\
&+W_{o,c,2,0}X_{n,c,p+2,q}
+W_{o,c,2,1}X_{n,c,p+2,q+1}
+W_{o,c,2,2}X_{n,c,p+2,q+2}
\end{aligned}
$$

となる。1つのfilter $W_o$ は、この $3\times3$ kernelを全入力channel $c$ について持つ。したがって `conv2` では、上の9要素が32組あり、合計は $9\times32=288$ 要素である。

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

### padding 0・1・2を同じ入力で比較する

$H=28$、$K=3$、$S=1$、$D=1$ を固定し、paddingだけを変える。

paddingなしの $P=0$ では、

$$
\begin{aligned}
O
&=
\left\lfloor
\frac{28+2\cdot0-3}{1}
\right\rfloor+1\\
&=25+1\\
&=26.
\end{aligned}
$$

$P=1$ では、

$$
\begin{aligned}
O
&=
\left\lfloor
\frac{28+2\cdot1-3}{1}
\right\rfloor+1\\
&=27+1\\
&=28.
\end{aligned}
$$

$P=2$ では、

$$
\begin{aligned}
O
&=
\left\lfloor
\frac{28+2\cdot2-3}{1}
\right\rfloor+1\\
&=29+1\\
&=30.
\end{aligned}
$$

したがってshapeは、

```text
padding=0: 28×28 → 26×26
padding=1: 28×28 → 28×28
padding=2: 28×28 → 30×30
```

となる。paddingはkernel sizeそのものを変える操作ではなく、入力の外側へ値を補ってkernelを置ける開始位置を増やす操作である。

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

### strideで割り切れない具体例

例えば、

$$
H=28,
\qquad
K=3,
\qquad
S=2,
\qquad
P=1,
\qquad
D=1
$$

を代入する。実効kernel sizeは

$$
K_{\mathrm{eff}}
=
1(3-1)+1
=
3
$$

なので、

$$
\begin{aligned}
H_{\mathrm{out}}
&=
\left\lfloor
\frac{
H+2P-D(K-1)-1
}{S}
\right\rfloor+1\\
&=
\left\lfloor
\frac{
28+2\cdot1-1(3-1)-1
}{2}
\right\rfloor+1\\
&=
\left\lfloor
\frac{27}{2}
\right\rfloor+1\\
&=
\left\lfloor13.5\right\rfloor+1\\
&=13+1\\
&=14.
\end{aligned}
$$

padding後のindexを$0$から$29$までとすると、kernel左端の開始位置は

$$
0,2,4,\ldots,26
$$

の14個である。最後の開始位置26では、kernelが使う位置は

$$
26,27,28
$$

であり、範囲内に収まる。次の開始位置28では

$$
28,29,30
$$

となり、位置30が範囲外なので置けない。この「置ける開始位置の個数」が出力サイズ14である。

また、paddingなしの$H=27,K=3,S=2$を考えると、正しい式は

$$
\left\lfloor
\frac{27-3}{2}
\right\rfloor+1
=
12+1
=
13
$$

だが、誤って

$$
\frac{H-K+1}{S}
$$

とすると

$$
\frac{27-3+1}{2}
=
12.5
$$

となる。出力個数は整数であり、$+1$はstrideで割った後、floorの外側に置く必要がある。

---

## 7. Poolingも空間サイズを変える

`MaxPool2d` も空間方向について同様に開始位置の個数を数える。

今回の

PyTorchの確認コード：[[05_SVD基礎実装検証/15_CNNのPyTorch確認コード#PyTorch確認-002]]

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

### 「特徴マップ数」と「scalarの個数」

shape `(N, 64, 14, 14)` の `64` は、1sample当たりの**特徴マップ数**である。各特徴マップは $14\times14$ 個のscalarを持つため、1sample当たりのscalar数は

$$
64\times14\times14
=64\times196
=12544
$$

である。2回目のPooling後も特徴マップ数は64のままだが、各特徴マップは $7\times7$ になるので、scalar数は

$$
64\times7\times7
=64\times49
=3136
$$

まで減る。batch全体のscalar数を数える場合だけ、さらに $N$ を掛ける。したがって「特徴量が64個」という表現はchannel数を指すのか、全scalar数を指すのかを区別する必要がある。

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

ViTのpatch embeddingをConvで実装するときは、通常

$$
K_h=K_w=P,
\qquad
S_h=S_w=P
$$

として、patch size $P$ と同じ大きさのkernelをpatch sizeずつ移動させる。これによりpatchは重ならない。

### paddingなし・重なりなしのpatch数

画像の高さを $H$、幅を $W$、正方形patchの一辺を $P$ とする。`kernel_size=P`、`stride=P`、`padding=0` を出力サイズ式へ代入すると、縦方向のpatch数は

$$
\begin{aligned}
H_{\mathrm{patch}}
&=
\left\lfloor
\frac{H-P}{P}
\right\rfloor+1\\
&=
\left\lfloor
\frac{H}{P}-1
\right\rfloor+1\\
&=
\left\lfloor
\frac{H}{P}
\right\rfloor.
\end{aligned}
$$

同様に、横方向は

$$
W_{\mathrm{patch}}
=
\left\lfloor
\frac{W}{P}
\right\rfloor
$$

なので、総patch数は

$$
\boxed{
N_{\mathrm{patch}}
=
H_{\mathrm{patch}}W_{\mathrm{patch}}
}
$$

となる。$H,W$ が $P$ で割り切れる場合は、

$$
N_{\mathrm{patch}}
=
\frac{H}{P}\frac{W}{P}
$$

と書ける。割り切れない場合、paddingなしでは右端・下端の余った領域は完全なpatchにならないため使われない。

### 28×28画像を全ケースで確認する

Fashion-MNISTの $28\times28$ 画像では、

| patch size $P$ | 縦×横のpatch数 | 総patch数 |
|---:|---:|---:|
| 1 | $28\times28$ | 784 |
| 2 | $14\times14$ | 196 |
| 4 | $7\times7$ | 49 |
| 7 | $4\times4$ | 16 |
| 14 | $2\times2$ | 4 |
| 28 | $1\times1$ | 1 |

となる。例えば $P=4$ なら、1patchはグレースケールの1channelを含めて

$$
1\times4\times4=16
$$

個の画素値を持ち、patchは

$$
\frac{28}{4}\times\frac{28}{4}
=7\times7
=49
$$

個になる。

### Convでpatch embeddingを実装する

次のConvは、重なりのない $4\times4$ patchを切り出しながら、各patchを64次元へ写像する。

```python
import torch
from torch import nn

patch_size = 4
embed_dim = 64

# 28×28の1channel画像を、重なりのない4×4 patchへ分ける。
patch_embed = nn.Conv2d(
    in_channels=1,
    out_channels=embed_dim,
    kernel_size=patch_size,
    stride=patch_size,
)

images = torch.randn(2, 1, 28, 28)
feature_map = patch_embed(images)
assert feature_map.shape == (2, 64, 7, 7)
```

入力shapeは

```text
(N, 1, 28, 28)
```

で、Conv出力は

```text
(N, 64, 7, 7)
```

となる。ここで64はpatch数ではなく、各patchを表すembeddingの次元である。Transformerへ渡すには空間軸 $7\times7$ を49個のtoken軸へ変換する。

```python
# channel軸はembedding次元として残し、7×7を49 tokenへまとめる。
x = feature_map.flatten(2)  # (N, 64, 49)
x = x.transpose(1, 2)       # (N, 49, 64)
assert x.shape == (2, 49, 64)
```

したがって最終shapeは

$$
(N,N_{\mathrm{patch}},D_{\mathrm{embed}})
=(N,49,64)
$$

である。

### strideとpatch sizeが異なる場合

例えば

```python
import torch
from torch import nn

# 4×4領域を2画素ずつ動かすため、隣り合うpatchは重なる。
overlapping_patch_embed = nn.Conv2d(
    in_channels=1,
    out_channels=64,
    kernel_size=4,
    stride=2,
)

images = torch.randn(2, 1, 28, 28)
feature_map = overlapping_patch_embed(images)
assert feature_map.shape == (2, 64, 13, 13)
```

では、$4\times4$ の領域を2画素ずつ移動させる。隣り合うpatchが重なり、縦・横のpatch数はそれぞれ

$$
\left\lfloor
\frac{28-4}{2}
\right\rfloor+1
=12+1
=13
$$

となる。shapeは

```text
(N, 1, 28, 28)
→ (N, 64, 13, 13)
→ 169 patch
```

である。

```text
stride = patch_size
→ patchは重ならない

stride < patch_size
→ patchが重なり、token数が増える

stride > patch_size
→ patch間に見ない領域が生じる
```

上記のpatch embeddingについては、次の確認コードを参照する。

PyTorchの確認コード：[[05_SVD基礎実装検証/15_CNNのPyTorch確認コード#PyTorch確認-003]]

---

## 11. よくある混乱

### `Conv2d` は3次元層？

畳み込みの空間軸は2次元。入力Tensorはbatchとchannelを含めて4次元である。

対して `Conv3d` は奥行き（動画なら時間）も含む3方向にkernelを動かす。batch付き入力は `(N,C,D,H,W)` の5次元であり、`Conv2d` の `(N,C,H,W)` と区別する。いずれも `N` と `C` はkernelを動かす空間軸の数には含めない。たとえばRGB画像1枚は `(3,H,W)` の3次元Tensorでも、使うのは通常 `Conv2d` である。

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
