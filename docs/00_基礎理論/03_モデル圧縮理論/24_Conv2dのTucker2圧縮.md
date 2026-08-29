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

この式は、出力channel $o$ と入力channel $i$ の結合を、

$$
i
\rightarrow
\beta
\rightarrow
\alpha
\rightarrow
o
$$

という2つのrank空間を通る形へ分解したものと読める。

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
C_{\mathrm{in}}\rightarrow R_{\mathrm{in}}
$$

へ圧縮する。

成分で書けば、

$$
\boxed{
Z_{\beta,h,w}
=
\sum_{i=1}^{C_{\mathrm{in}}}
U^{\mathrm{in}}_{i,\beta}
X_{i,h,w}
}
$$

である。

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
R_{\mathrm{in}}\rightarrow R_{\mathrm{out}}
$$

の $K_h\times K_w$ Convとしてそのまま使える。

stride 1・dilation 1で境界処理を省略した概念式なら、

$$
T_{\alpha,h,w}
=
\sum_{\beta=1}^{R_{\mathrm{in}}}
\sum_{a=0}^{K_h-1}
\sum_{b=0}^{K_w-1}
G_{\alpha,\beta,a,b}
Z_{\beta,h+a,w+b}
$$

となる。

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
U_{\mathrm{out}}T_{:,h,w}+b
}
$$

成分では、

$$
\boxed{
Y_{o,h,w}
=
\sum_{\alpha=1}^{R_{\mathrm{out}}}
U^{\mathrm{out}}_{o,\alpha}
T_{\alpha,h,w}
+b_o
}
$$

である。

$$
R_{\mathrm{out}}\rightarrow C_{\mathrm{out}}
$$

なので最後の1x1 Conv weightは

$$
(C_{\mathrm{out}},R_{\mathrm{out}},1,1)
$$

であり、$U_{\mathrm{out}}$ を転置せず使う。

---

## 3. 一般のstride / padding / dilationまで含めたforward

上の $h+a,w+b$ は構造を理解するための簡略式である。現在のsrcでは中央core Convが元Convの `stride / padding / dilation / padding_mode` を引き継ぐので、一般形も分けて書く。

batch indexを $n$、出力位置を $(p,q)$、strideを $(S_h,S_w)$、dilationを $(D_h,D_w)$、paddingを $(P_h,P_w)$ とする。

最初の1x1 projectionはstride 1 / padding 0なので、

$$
Z_{n,\beta,h,w}
=
\sum_{i=1}^{C_{\mathrm{in}}}
U^{\mathrm{in}}_{i,\beta}
X_{n,i,h,w}
$$

である。

padding後のTensorを $\bar Z=\mathcal P(Z)$ と書く。zero paddingなら周囲に0を補い、その他の `padding_mode` ならその規則に従って拡張したTensorを表す。

中央core Convは、

$$
\boxed{
T_{n,\alpha,p,q}
=
\sum_{\beta=1}^{R_{\mathrm{in}}}
\sum_{a=0}^{K_h-1}
\sum_{b=0}^{K_w-1}
G_{\alpha,\beta,a,b}
\bar Z_{n,\beta,\,pS_h+aD_h,\,qS_w+bD_w}
}
$$

となる。

zero paddingを元Tensorの添字で直接書くなら、範囲外を0と約束して

$$
T_{n,\alpha,p,q}
=
\sum_{\beta,a,b}
G_{\alpha,\beta,a,b}
Z_{n,\beta,\,pS_h-P_h+aD_h,\,qS_w-P_w+bD_w}
$$

と同じ意味になる。

最後の1x1 projectionは

$$
\boxed{
Y_{n,o,p,q}
=
\sum_{\alpha=1}^{R_{\mathrm{out}}}
U^{\mathrm{out}}_{o,\alpha}
T_{n,\alpha,p,q}
+b_o
}
$$

である。

この3式を代入すると、元Convと同じ空間サンプリング規則の下でeffective weight

$$
\hat W_{o,i,a,b}
=
\sum_{\alpha=1}^{R_{\mathrm{out}}}
\sum_{\beta=1}^{R_{\mathrm{in}}}
U^{\mathrm{out}}_{o,\alpha}
G_{\alpha,\beta,a,b}
U^{\mathrm{in}}_{i,\beta}
$$

を使ったConvへ戻る。

詳細な代入は [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]] を参照。

---

## 4. なぜ入力側だけ転置するか

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

## 5. `[:, :, None, None]` の意味

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

## 6. biasを最後へ置く理由

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

## 7. パラメータ数

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

## 8. MACs

元Convの出力空間を $H_{\mathrm{out}}\times W_{\mathrm{out}}$ とすると、元ConvのMACsは

$$
\boxed{
\operatorname{MACs}_{\mathrm{original}}
=
H_{\mathrm{out}}W_{\mathrm{out}}
C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w
}
$$

である。

Tucker-2では3層を別々に数える。

最初の1x1 projectionは入力側空間 $H_{\mathrm{in}}\times W_{\mathrm{in}}$ で実行されるので、

$$
\operatorname{MACs}_{\mathrm{input}}
=
H_{\mathrm{in}}W_{\mathrm{in}}
C_{\mathrm{in}}R_{\mathrm{in}}
$$

中央core Convは、

$$
\operatorname{MACs}_{\mathrm{core}}
=
H_{\mathrm{out}}W_{\mathrm{out}}
R_{\mathrm{out}}R_{\mathrm{in}}K_hK_w
$$

最後の1x1 projectionは、

$$
\operatorname{MACs}_{\mathrm{output}}
=
H_{\mathrm{out}}W_{\mathrm{out}}
R_{\mathrm{out}}C_{\mathrm{out}}
$$

である。

したがって一般形は、

$$
\boxed{
\begin{aligned}
\operatorname{MACs}_{\mathrm{Tucker2}}
={}&
H_{\mathrm{in}}W_{\mathrm{in}}C_{\mathrm{in}}R_{\mathrm{in}}\\
&+
H_{\mathrm{out}}W_{\mathrm{out}}R_{\mathrm{out}}R_{\mathrm{in}}K_hK_w\\
&+
H_{\mathrm{out}}W_{\mathrm{out}}R_{\mathrm{out}}C_{\mathrm{out}}.
\end{aligned}
}
$$

前後を含む3層で空間サイズが同じ場合、すなわち

$$
H_{\mathrm{in}}W_{\mathrm{in}}
=
H_{\mathrm{out}}W_{\mathrm{out}}
=HW
$$

なら、

$$
\boxed{
\operatorname{MACs}_{\mathrm{Tucker2}}
=
HW
\left(
C_{\mathrm{in}}R_{\mathrm{in}}
+R_{\mathrm{out}}R_{\mathrm{in}}K_hK_w
+C_{\mathrm{out}}R_{\mathrm{out}}
\right)
}
$$

となる。

これは例えばstride 1で、padding / dilationの組合せにより入力とcore出力の空間サイズが維持される場合に使える簡略式である。**stride 1だけでは空間サイズ維持は保証されない**。

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

## 9. rankと精度のtrade-off

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

## 10. PyTorch実装上の契約

今回のsrcでは、元Convの意味を保つため次を維持する。

- `groups=1` の通常Conv2dのみ対応
- 中央core Convが元の `stride / padding / dilation / padding_mode` を継承
- 前後1x1 Convはstride 1 / padding 0 / dilation 1
- device / dtypeを維持
- 元weight / biasの `requires_grad` を維持
- 元biasは最後の1x1へコピー
- 元Conv自体を破壊せず、置換後Parameterとstorageを共有しない
- `rank_out / rank_in` はboolを拒否し、`1 <= rank_out <= C_out`, `1 <= rank_in <= C_in`
- 現行HOSVD/HOOI経路は実数dtype限定で、複素Tensorは明示的に拒否

重みコピーは学習演算ではなく初期化なので、

```python
with torch.no_grad():
    parameter.copy_(value)
```

を使う。

`no_grad()` はコピー操作の履歴を記録しないだけで、コピー後のParameterを学習不能にするものではない。

### autograd境界はAPI層で分ける

すべてのTucker分解で入力weightを一律 `detach()` するわけではない。

```text
tucker2_hooi(weight, ...)
→ Tensor-level decomposition
→ 入力weightをdetachしない

build_tucker2_conv(conv, ...)
→ Module構築・初期化
→ conv.weightをdetachして分解し、新しいleaf Parameterへcopy
```

この分離により、低レベルTensor APIでは必要以上にautograd graphを切らず、Module置換時には元モデルと新しいParameterを独立させる。

`tucker2_effective_weight()` は評価用helperなので、3層のweightをdetachして等価な4階weightを再構成する。

---

## 11. srcとの対応

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
