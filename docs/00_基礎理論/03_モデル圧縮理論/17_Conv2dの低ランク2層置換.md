---
title: Conv2dの低ランク2層置換
aliases:
  - Conv-SVDの2層置換
  - 低ランクConv
  - 3x3 Convと1x1 Convへの分解
  - factorize_conv2d_layer
tags:
  - CNN
  - Conv2d
  - SVD
  - 低ランク近似
  - PyTorch
  - NN圧縮
---

# Conv2dの低ランク2層置換

## サマリー

Conv2d weightを行列化し、rank $r$ のSVD近似、

$$
W_{\mathrm{mat}}
\approx
U_r\Sigma_rV_r^{\mathsf{T}}
$$

を得ても、それを元と同じ4次元dense weightへ完全に戻して保存すれば、パラメータ数は減らない。

圧縮では、2因子を別々のConv層として保持する。

現在の実装では、

$$
F_1
=
V_r^{\mathsf{T}}
$$

$$
F_2
=
U_r\Sigma_r
$$

として、

$$
W_{\mathrm{mat}}
\approx
F_2F_1
$$

とする。

元の、

```text
Conv2d(C_in → C_out, K_h×K_w)
```

を、

```text
Conv2d(C_in → r, K_h×K_w)
        ↓
Conv2d(r → C_out, 1×1)
```

へ置き換える。

今回の `conv2` なら、

```text
元
Conv2d(32 → 64, 3×3)

SVD後
Conv2d(32 → r, 3×3)
        ↓
Conv2d(r → 64, 1×1)
```

となる。

2層の間にReLUなどの非線形関数は入れない。

---

## 対応コード

現在の汎用実装：

```text
src/nn_compression/compression/conv_svd.py
```

主要関数：

```python
from nn_compression.compression import (
    factorize_conv2d_layer,
    factorize_named_conv2d,
)
```

`factorize_conv2d_layer` は任意の `groups=1` の `nn.Conv2d` を対象とする。

`factorize_named_conv2d` はモデルを `copy.deepcopy()` し、指定属性のConv2dだけを置換した新しいモデルを返す。

旧Notebook向けaliasとして、

```text
factorize_Conv2d_layer
```

も残しているが、新しいコードではsnake_caseの `factorize_conv2d_layer` を使う。

---

## 1. 因子とConv weightのshape対応

元weightを、

```text
(C_out, C_in, K_h, K_w)
```

とする。

行列化後は、

```text
(C_out, C_in×K_h×K_w)
```

である。

rank $r$ なら、

```text
F1 = Vh_r
shape: (r, C_in×K_h×K_w)

F2 = U_r @ diag(S_r)
shape: (C_out, r)
```

となる。

これをConv weightのshapeへ戻す。

### 1層目

```text
(r, C_in×K_h×K_w)
↓ reshape
(r, C_in, K_h, K_w)
```

これは、

```python
nn.Conv2d(
    C_in,
    r,
    kernel_size=(K_h, K_w),
)
```

のweight shapeと一致する。

### 2層目

```text
(C_out, r)
↓ reshape
(C_out, r, 1, 1)
```

これは、

```python
nn.Conv2d(
    r,
    C_out,
    kernel_size=1,
)
```

のweight shapeと一致する。

---

## 2. なぜ2層目は `1×1 Conv` なのか

2層目の因子は、

```text
(C_out, r)
```

である。

空間方向のkernel要素を持っていない。

そのため、

```text
(C_out, r, 1, 1)
```

と解釈し、各空間位置で `r` 個の中間channelを `C_out` 個へ線形結合する `1×1 Conv` として実装できる。

```text
1×1 Conv
=
同じ位置にあるchannel同士を混ぜる線形変換
```

である。

空間近傍を見る処理は1層目の元kernel側が担当する。

---

## 3. stride / padding / dilationの引き継ぎ

元Convの空間的なサンプリングは1層目へ持たせる。

```python
first_layer = nn.Conv2d(
    in_channels=conv.in_channels,
    out_channels=rank,
    kernel_size=conv.kernel_size,
    stride=conv.stride,
    padding=conv.padding,
    dilation=conv.dilation,
    bias=False,
)
```

2層目は、既に作られた中間特徴マップの各位置をchannel方向へ混ぜるだけなので、

```python
second_layer = nn.Conv2d(
    in_channels=rank,
    out_channels=conv.out_channels,
    kernel_size=1,
    stride=1,
    padding=0,
    bias=...,
)
```

とする。

### `stride=1` の理由

元のstrideによる空間的な間引きは1層目で既に行われている。2層目でもstrideを掛けると、さらに空間サイズを縮めてしまう。

### `padding=0` の理由

`1×1` kernelは1画素だけを見るため、元の空間サイズを維持するのにpaddingは不要である。

---

## 4. biasは第2層へ置く

元Convを、

$$
y=Wx+b
$$

と考える。

2因子化後は、

$$
h=F_1x
$$

$$
y=F_2h+b
$$

とすれば、

$$
y=F_2F_1x+b
$$

となる。

したがって、

```text
1層目 bias=False
2層目 元biasを保持
```

とする。

元Convが `bias=False` なら、2層目もbiasを持たない。

---

## 5. 2層間にReLUを入れない

正しい低ランク置換は、

```text
Conv(C_in → r, K×K)
        ↓
Conv(r → C_out, 1×1)
```

である。

途中へ、

```text
ReLU
Sigmoid
GELU
```

などを入れると、

$$
F_2F_1x
$$

ではなく、

$$
F_2\phi(F_1x)
$$

となるため、元ConvのSVD近似とは別の非線形モデルになる。

元モデルで `conv2` の**後ろ**に置かれていたReLUは、そのまま2層Sequentialの後ろに残す。

```text
元
conv2
↓
ReLU

置換後
Conv factor 1
↓
Conv factor 2
↓
ReLU
```

である。

---

## 6. 出力shapeは維持する

今回の、

```text
Conv2d(32 → 64, 3×3, stride=1, padding=1)
```

を、

```text
Conv2d(32 → r, 3×3, stride=1, padding=1)
↓
Conv2d(r → 64, 1×1, stride=1, padding=0)
```

へ置き換えると、

```text
入力              (N, 32, H, W)
1層目出力          (N, r,  H, W)
2層目最終出力      (N, 64, H, W)
```

となる。

rank $r$ は内部表現のchannel数であり、元モデルの外から見た入出力shapeは変えない。

---

## 7. パラメータ数

`groups=1`、biasありとする。

### 元Conv

$$
P_{\mathrm{original}}
=
C_{\mathrm{out}}
C_{\mathrm{in}}
K_hK_w
+
C_{\mathrm{out}}
$$

### 2層Conv

1層目：

$$
rC_{\mathrm{in}}K_hK_w
$$

2層目weight：

$$
C_{\mathrm{out}}r
$$

2層目bias：

$$
C_{\mathrm{out}}
$$

合計：

$$
P_{\mathrm{factorized}}
=
r
\left(
C_{\mathrm{in}}K_hK_w
+
C_{\mathrm{out}}
\right)
+
C_{\mathrm{out}}
$$

である。

---

## 8. 圧縮が成立するrank

biasは前後で同じなので相殺できる。

$$
P_{\mathrm{factorized}}
<
P_{\mathrm{original}}
$$

より、

$$
r
<
\frac{
C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w
}{
C_{\mathrm{in}}K_hK_w+C_{\mathrm{out}}
}
$$

となる。

### `conv2` の例

```text
C_in  = 32
C_out = 64
K_h   = 3
K_w   = 3
```

なので、

$$
r
<
\frac{64\times32\times3\times3}{32\times3\times3+64}
$$

$$
r
<
\frac{18{,}432}{352}
\approx
52.36
$$

である。

整数rankでは、

```text
r <= 52
```

ならweight数を減らせる。

最大rankは64なので、

```text
数学的には有効なrank 64
≠
圧縮になるrank 64
```

である。

---

## 9. `conv2 rank=28` のパラメータ数例

元：

$$
64\times32\times3\times3+64
=
18{,}496
$$

rank 28では、1層目、

$$
28\times32\times3\times3
=
8{,}064
$$

2層目weight、

$$
64\times28
=
1{,}792
$$

bias、

$$
64
$$

なので、

$$
8{,}064+1{,}792+64
=
9{,}920
$$

となる。

`conv2` 自体では約46%のパラメータを削減できる。

ただし、モデル全体のパラメータ削減率は、その層がモデル全体の何割を占めるかで決まる。

今回のCNNでは `fc1` がパラメータの大部分を持つため、`conv2` 単体を圧縮してもモデル全体のパラメータ削減率は小さい。

---

## 10. full rankは実装検証に使う

最大rankで切り捨てを行わなければ、

```text
元Conv出力
≈
factorized Conv出力
```

となる。

現在のテストでは、full rankのConv2d分解について、数値誤差範囲で出力一致とshape一致を確認している。

これは、

- 行列化
- SVD因子の向き
- reshape
- bias配置
- stride / padding

が正しいことを確認する重要なテストである。

ただしfull rankでは、2因子化によりパラメータ数が増えることがある。

```text
full rank一致
→ 実装検証

低rank
→ 実際の圧縮
```

と役割を分ける。

---

## 11. SVD後のweightはランダム初期化し直さない

Fine-tuning前の2層Convは、ランダム初期値から作り直すのではない。

```text
学習済みConv weight
↓
SVD
↓
truncated SVD因子
↓
新しい2層Convのweightへコピー
↓
その状態からFine-tuning
```

である。

つまり、

```text
weightの値
→ 学習済みweightからSVDで引き継ぐ

optimizer
→ 新しく作り直す
```

という関係である。

---

## 12. optimizerを作り直す理由

SVD前は、

```text
1つのConv2d Parameter群
```

だった。

SVD後は、

```text
1層目ConvのParameter群
+
2層目ConvのParameter群
```

という新しい `Parameter` オブジェクトになる。

元optimizerは、作成時に登録された古いParameterを参照している。

したがって、構造変更後は、

```python
optimizer = torch.optim.Adam(
    compressed_model.parameters(),
    lr=...,
)
```

と作り直す。

Adamなら、parameterごとに、

- `step`
- `exp_avg`
- `exp_avg_sq`

などの内部状態も持つ。

したがって、

```text
weightはSVDから継承
optimizer stateは継承しない
```

と考えると分かりやすい。

詳細は [[00_基礎理論/02_ニューラルネットワーク基礎/12_PyTorch学習と評価の基礎]]。

---

## 13. Fine-tuningの役割

切り詰めSVDでは、小さい特異値方向を捨てるため、元weightと完全には一致しない。

Fine-tuningは、その近似誤差を、残されたrankの範囲内で分類タスクに合うよう再調整する。

```text
SVD
→ 行列近似として良い初期値

Fine-tuning
→ タスクlossに合わせて再調整
```

である。

通常、ゼロからの学習より小さめのlearning rateを使うことが多い。

---

## 14. `copy.deepcopy` と層置換

元モデルを残したまま候補モデルを作るなら、

```python
compressed_model = copy.deepcopy(model)
```

してから置換する。

現在の便利関数、

```python
factorize_named_conv2d(
    model,
    "conv2",
    rank,
)
```

も、元モデルをdeepcopyし、コピー側だけを変更する。

```text
元モデル
→ 変更しない

圧縮モデル
→ 独立したModule / Parameter群
```

となる。

`model.conv2` へ `(64,288)` のTensorそのものを代入してはいけない。

`model.conv2` は `nn.Module` としてforward時に呼び出されるため、置換後も、

```text
nn.Conv2d
または
nn.Sequential
```

などの `nn.Module` である必要がある。

---

## 15. deviceとdtypeを維持する

新しいConv層はデフォルトではCPU / default dtypeで作られる。

元モデルがGPU上や別dtypeにある場合に備えて、

```python
return nn.Sequential(
    first_layer,
    second_layer,
).to(
    device=conv.weight.device,
    dtype=conv.weight.dtype,
)
```

のように、元weightのdevice / dtypeを引き継ぐ。

---

## 16. `groups` の扱い

現在の `factorize_conv2d_layer()` は、

```text
groups = 1
```

のConv2dだけを対象とする。

```python
if conv.groups != 1:
    raise ValueError(...)
```

として、未対応の構造を明示する。

Depthwise Convなどへ同じfactorizationをそのまま適用するのは別問題である。

一方、MACsを数える汎用関数はgrouped Convのweight shapeを考慮している。詳しくは [[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]]。

---

## 17. `conv1` と `conv2` のどちらを圧縮するか

理論上は両方SVDできる。

### `conv1`

```text
weight.shape = (32, 1, 3, 3)
行列化       = (32, 9)
最大rank      = 9
parameter     = 320（bias含む）
```

### `conv2`

```text
weight.shape = (64, 32, 3, 3)
行列化       = (64, 288)
最大rank      = 64
parameter     = 18,496（bias含む）
```

したがって、今回の小型CNNでは、最初にConv-SVDのtrade-offを観察する対象として `conv2` の方が適している。

`conv1` を無視するという意味ではなく、**圧縮対象としての寄与が大きい層から調べる**という実験設計である。

---

## 18. Linear SVDとの対応

Linearでは、

```text
Linear(D_in → D_out)
↓
Linear(D_in → r)
↓
Linear(r → D_out)
```

だった。

Convでは、

```text
Conv(C_in → C_out, K×K)
↓
Conv(C_in → r, K×K)
↓
Conv(r → C_out, 1×1)
```

となる。

共通する本質は、

```text
大きなweight行列
↓
rank r の2因子
↓
2つの小さい線形演算として保持
```

である。

違いは、Convではweightが最初4次元なので行列化が必要であり、空間構造を保つため因子をConv kernelへreshapeする点である。

---

## 19. よくある間違い

### 特徴マップをSVDする

今回圧縮するのは `conv.weight` であり、入力activationではない。

### `(64,288)` を `model.conv2` へ代入する

TensorはConv Moduleの代わりにならない。

### SVD後のdense weightを1つのConvへ戻して「圧縮した」と考える

近似にはなるが、保存するweight数は元と同じである。

### 1層目へbiasを置く

元biasは最終出力へ一度だけ加えるため、第2層へ置く。

### 2層間へReLUを入れる

元の線形Conv演算の低ランク近似ではなくなる。

### 2層目にも元stride / paddingを設定する

空間変換を二重に適用してしまう。

### SVD後も元optimizerを使う

新しいParameterがoptimizerへ登録されない。

### full rankなら必ず軽量化されると思う

full rankは主に等価性テスト。圧縮成立rankは別に計算する。

---

## 20. このノートで押さえるポイント

- Conv weightのSVD因子を、`K×K Conv + 1×1 Conv` として保持すると圧縮できる
- 1層目は元kernel / stride / padding / dilationを引き継ぐ
- 2層目は `1×1, stride=1, padding=0`
- biasは第2層だけへ置く
- 2層間へ非線形関数を入れない
- 外から見た入出力shapeは維持する
- full rank一致は実装検証、低rankが圧縮
- SVD-derived weightからFine-tuningを始め、ランダム再初期化しない
- 構造変更でParameterが新しくなるのでoptimizerを作り直す
- 元モデルを残したい場合はdeepcopyする
- device / dtypeを引き継ぐ
- 現在のfactorization実装は `groups=1` に限定する

---

## 次に読むノート

- [[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]]
- [[00_基礎理論/02_ニューラルネットワーク基礎/12_PyTorch学習と評価の基礎]]
- [[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]
