---
title: Conv2dのTucker-2圧縮
aliases:
  - Tucker-2 Conv
  - Conv Tucker decomposition
  - 1x1 kxk 1x1 Tucker
tags:
  - Tucker
  - CNN
  - Conv2d
  - NN圧縮
  - PyTorch
---

# Conv2dのTucker-2圧縮

## サマリー

Conv2d weight

$$
W
\in
\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_h\times K_w}
$$

のchannel mode 0 / 1だけをTucker分解すると、元の1つのConvを

```text
1x1 Conv
→ kHxkW Conv
→ 1x1 Conv
```

の3層へ置き換えられる。

channelの流れは

$$
\boxed{
C_{\mathrm{in}}
\rightarrow
R_{\mathrm{in}}
\rightarrow
R_{\mathrm{out}}
\rightarrow
C_{\mathrm{out}}
}
$$

である。

---

## 1. Tucker-2の重み近似

Tucker-2では

$$
U_{\mathrm{out}}
\in
\mathbb{R}^{C_{\mathrm{out}}\times R_{\mathrm{out}}}
$$

$$
U_{\mathrm{in}}
\in
\mathbb{R}^{C_{\mathrm{in}}\times R_{\mathrm{in}}}
$$

$$
G
\in
\mathbb{R}^{R_{\mathrm{out}}\times R_{\mathrm{in}}\times K_h\times K_w}
$$

を用いて

$$
W
\approx
G
\times_0 U_{\mathrm{out}}
\times_1 U_{\mathrm{in}}
$$

と近似する。

要素表示では、

$$
\boxed{
W_{o,i,a,b}
\approx
\sum_{\alpha=1}^{R_{\mathrm{out}}}
\sum_{\beta=1}^{R_{\mathrm{in}}}
U^{\mathrm{out}}_{o,\alpha}
G_{\alpha,\beta,a,b}
U^{\mathrm{in}}_{i,\beta}
}
$$

である。

---

## 2. 3層へ分けられる理由

### 入力projection

入力feature mapをchannelごとのベクトルとして

$$
X_{:,h,w}
\in
\mathbb{R}^{C_{\mathrm{in}}}
$$

とする。

まず

$$
\boxed{
Z_{:,h,w}
=
U_{\mathrm{in}}^{\mathsf{T}}X_{:,h,w}
}
$$

で

$$
C_{\mathrm{in}}ightarrow R_{\mathrm{in}}
$$

へ圧縮する。

したがって最初の1x1 Conv weightは

$$
U_{\mathrm{in}}^{\mathsf{T}}
\in
\mathbb{R}^{R_{\mathrm{in}}\times C_{\mathrm{in}}}
$$

を4階weightへした

$$
(R_{\mathrm{in}},C_{\mathrm{in}},1,1)
$$

になる。

### core convolution

中央のcoreは

$$
G
\in
\mathbb{R}^{R_{\mathrm{out}}\times R_{\mathrm{in}}\times K_h\times K_w}
$$

なので、

$$
R_{\mathrm{in}}ightarrow R_{\mathrm{out}}
$$

の $K_h\times K_w$ Convとしてそのまま使える。

### 出力projection

core出力

$$
T_{:,h,w}\in\mathbb{R}^{R_{\mathrm{out}}}
$$

を元の出力channelへ戻す。

$$
\boxed{
Y_{:,h,w}
=
U_{\mathrm{out}}T_{:,h,w}
}
$$

$$
R_{\mathrm{out}}ightarrow C_{\mathrm{out}}
$$

なので最後の1x1 Conv weightは

$$
(C_{\mathrm{out}},R_{\mathrm{out}},1,1)
$$

であり、$U_{\mathrm{out}}$ を転置せず使う。

---

## 3. なぜ入力側だけ転置するか

factorは通常

$$
U^{(n)}\in\mathbb{R}^{I_n\times R_n}
$$

である。

- 元空間 $I_n$ からrank空間 $R_n$ へ圧縮：$U^{(n)\mathsf{T}}$
- rank空間 $R_n$ から元空間 $I_n$ へ展開：$U^{(n)}$

今回のforwardは

```text
入力側:
C_in → R_in
なので U_in.T

出力側:
R_out → C_out
なので U_out
```

となる。

形状に合わせて機械的に転置するのではなく、**今いる空間から次に必要な空間への写像方向**で決める。

---

## 4. `[:, :, None, None]` の意味

$U_{\mathrm{in}}^{\mathsf{T}}$ は2階行列。

$$
(R_{\mathrm{in}},C_{\mathrm{in}})
$$

しかし `nn.Conv2d.weight` は4階。

1x1 Convでは

$$
(R_{\mathrm{in}},C_{\mathrm{in}},1,1)
$$

が必要なので、

```python
u_in.T[:, :, None, None]
```

でsize 1の空間軸を2つ追加する。

同様に

```python
u_out[:, :, None, None]
```

は

$$
(C_{\mathrm{out}},R_{\mathrm{out}})
\rightarrow
(C_{\mathrm{out}},R_{\mathrm{out}},1,1)
$$

へ変換する。

---

## 5. biasを最後へ置く理由

元Convは概念的に

$$
Y_o
=
\operatorname{Conv}(X;W)_o+b_o
$$

である。

元biasは $C_{\mathrm{out}}$ 次元。

Tucker-2では最終1x1 Convの出力だけが元の $C_{\mathrm{out}}$ 次元になるので、biasは最後に置く。

```text
input 1x1 : biasなし
core kxk  : biasなし
output 1x1: 元biasをコピー
```

中間rank空間に元biasを入れると、元Convの出力biasと同じ意味にならない。

---

## 6. パラメータ数

元Conv weightの要素数は

$$
\boxed{
N_{\mathrm{original}}
=
C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w
}
$$

bias込みなら

$$
N_{\mathrm{original,bias}}
=
C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w
+C_{\mathrm{out}}
$$

Tucker-2のweight要素数は、

$$
\boxed{
N_{\mathrm{Tucker2}}
=
C_{\mathrm{in}}R_{\mathrm{in}}
+
R_{\mathrm{out}}R_{\mathrm{in}}K_hK_w
+
C_{\mathrm{out}}R_{\mathrm{out}}
}
$$

biasを最後に持つなら

$$
N_{\mathrm{Tucker2,bias}}
=
N_{\mathrm{Tucker2}}
+C_{\mathrm{out}}
$$

である。

### 本当に圧縮になる条件

$$
\boxed{
C_{\mathrm{in}}R_{\mathrm{in}}
+
R_{\mathrm{out}}R_{\mathrm{in}}K_hK_w
+
C_{\mathrm{out}}R_{\mathrm{out}}
<
C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w
}
$$

を満たす必要がある。

rankを元channel数と同じにすると、前後の1x1 Convが追加される分、**Tucker化したのに元より増える**ことがある。

---

## 7. MACs

1出力位置あたりの元Conv MACsは

$$
C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w
$$

である。

stride 1で空間サイズが全段同じとみなせる場合、Tucker-2では1出力位置あたり概念的に

$$
C_{\mathrm{in}}R_{\mathrm{in}}
+
R_{\mathrm{in}}R_{\mathrm{out}}K_hK_w
+
R_{\mathrm{out}}C_{\mathrm{out}}
$$

になる。

一般のstrideでは、最初の1x1は入力側空間、coreと最後の1x1は出力側空間で計算されるため、実装された各Convの出力shapeに基づいてMACsを数える。

重要なのは、

```text
parameters reduction
≠
MACs reduction
≠
wall-clock latency reduction
```

である。

---

## 8. rankと精度のtrade-off

rankを下げると、

```text
R_in / R_outが小さくなる
→ parameters・MACsが減る
→ 表現できるweight部分空間が狭くなる
→ weight reconstruction errorが増えやすい
→ accuracyも落ちやすい
```

ただし、weight errorとaccuracyは完全な1対1対応ではない。

今回のrank sweepでも、同程度のaccuracyなのにweight errorが異なるrank pairが存在した。

したがってrank選択では、

- validation accuracy / accuracy drop
- parameters
- MACs
- weight relative error
- fine-tuning後accuracy

を別々に見る。

---

## 9. PyTorch実装上の契約

今回のsrcでは、元Convの意味を保つため次を維持する。

- `groups=1` の通常Conv2dのみ対応
- 中央core Convが元の `stride / padding / dilation / padding_mode` を継承
- 前後1x1 Convはstride 1 / padding 0
- device / dtypeを維持
- 元weight / biasの `requires_grad` を維持
- 元biasは最後の1x1へコピー
- 元Conv自体を破壊しない

重みコピーは学習演算ではなく初期化なので、

```python
with torch.no_grad():
    parameter.copy_(value)
```

を使う。

`no_grad()` はコピー操作の履歴を記録しないだけで、コピー後のParameterを学習不能にするものではない。

分解対象weightは通常

```python
weight = conv.weight.detach()
```

として既存モデルのautograd graphから切り離して扱う。

---

## 10. srcとの対応

```text
src/nn_compression/compression/conv_tucker.py
├─ tucker2_decompose_conv_weight
├─ build_tucker2_conv
├─ build_tucker2_conv_from_components
├─ tucker2_hooi
├─ tucker2_hooi_sweep
└─ tucker2_effective_weight
```

検証：

- [[06_Tucker基礎実装検証/02_Tucker2_Convとrank_sweepの確認結果]]
- [[06_Tucker基礎実装検証/03_HOOIとTensorLy照合]]
