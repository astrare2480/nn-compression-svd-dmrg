---
title: テンソルとmode演算
aliases:
  - Tensorの基礎
  - mode-n unfolding
  - mode product
  - unfoldとfold
tags:
  - Tensor
  - Tucker
  - HOSVD
  - HOOI
  - 線形代数
---

# テンソルとmode演算

## サマリー

Tucker分解・HOSVD・HOOIを理解するには、Tensorを「多次元配列」として見るだけでなく、**各軸をmodeとして取り出し、行列化（unfold）し、行列を特定modeへ掛ける**という操作を理解する必要がある。

$N$階Tensorを

$$
\mathcal{X}
\in
\mathbb{R}^{I_0\times I_1\times\cdots\times I_{N-1}}
$$

とする。mode $n$ は第$n$軸を表し、mode-$n$ unfoldingはその軸を行方向へ移した行列

$$
X_{(n)}
\in
\mathbb{R}^{I_n\times\prod_{m\neq n}I_m}
$$

を作る操作である。

mode-$n$ productは、行列

$$
A\in\mathbb{R}^{J\times I_n}
$$

をmode $n$へ掛け、

$$
\mathcal{Y}
=
\mathcal{X}\times_n A
$$

として第$n$軸だけを $I_n\rightarrow J$ に変換する。

この2操作が、

```text
Tensor
→ 対象modeをunfold
→ SVD
→ factorを得る
```

というHOSVDと、

```text
Tensor
→ 他modeのfactorで射影
→ 対象modeをunfold
→ SVD
→ factorを更新
```

というHOOIの共通部品になる。

---

## 1. Tensorとmode

ベクトルは1階、行列は2階、Conv2dのweightは4階Tensorである。

PyTorchの通常のConv2d weightは、

$$
W
\in
\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_h\times K_w}
$$

で、各modeは次の意味を持つ。

| mode | 軸 | 意味 |
| ---: | --- | --- |
| 0 | $C_{\mathrm{out}}$ | 出力channel |
| 1 | $C_{\mathrm{in}}$ | 入力channel |
| 2 | $K_h$ | kernel高さ |
| 3 | $K_w$ | kernel幅 |

今回のTucker-2では、channel方向のmode 0 / 1だけを圧縮し、空間mode 2 / 3は保持する。

---

## 2. mode-n unfolding

### 定義

Tensor

$$
\mathcal{X}
\in
\mathbb{R}^{I_0\times\cdots\times I_{N-1}}
$$

をmode $n$でunfoldすると、

$$
X_{(n)}
\in
\mathbb{R}^{I_n\times\prod_{m\neq n}I_m}
$$

という行列になる。

重要なのは、**mode $n$の次元を行方向に置き、残りの軸を列方向へまとめる**こと。

Conv weight

$$
W\in\mathbb{R}^{64\times32\times3\times3}
$$

なら、mode 0 unfoldingは

$$
W_{(0)}
\in
\mathbb{R}^{64\times(32\cdot3\cdot3)}
=
\mathbb{R}^{64\times288}
$$

mode 1 unfoldingは

$$
W_{(1)}
\in
\mathbb{R}^{32\times(64\cdot3\cdot3)}
=
\mathbb{R}^{32\times576}
$$

になる。

この行列へ通常のSVDを行えば、mode 0では出力channel側、mode 1では入力channel側の主要部分空間を得られる。

### SVDとHOSVDの名前の違い

単一modeをunfoldして行うのは、あくまで**1回の行列SVD**である。

```text
mode 0だけunfold → SVD
= mode 0 unfoldingに対する行列SVD

mode 1だけunfold → SVD
= mode 1 unfoldingに対する行列SVD
```

複数modeについて元Tensorをそれぞれunfoldし、factor一式とcoreを組み立てるアルゴリズム全体をHOSVDと呼ぶ。

---

## 3. fold

`fold`はunfoldの逆操作である。

$$
\operatorname{fold}_n(X_{(n)})=\mathcal{X}
$$

したがって、正しく実装されているなら

$$
\operatorname{fold}_n(
\operatorname{unfold}_n(\mathcal{X})
)
=
\mathcal{X}
$$

が成立する。

自作srcでは、対象modeを先頭へ移してflattenし、foldでは元shapeを復元した後、先頭modeを元位置へ戻すことでこの対応を実装する。

参照実装：`src/nn_compression/tensor/operations.py`

---

## 4. mode-n product

行列

$$
A
\in
\mathbb{R}^{J\times I_n}
$$

をTensor $\mathcal{X}$ のmode $n$へ掛ける操作を

$$
\mathcal{Y}
=
\mathcal{X}\times_n A
$$

と書く。

結果のshapeは

$$
\mathcal{Y}
\in
\mathbb{R}^{
I_0\times\cdots\times I_{n-1}
\times J
\times I_{n+1}\times\cdots\times I_{N-1}
}
$$

となり、第$n$軸だけが $I_n\rightarrow J$ に変わる。

要素表示では、

$$
Y_{i_0,\ldots,i_{n-1},j,i_{n+1},\ldots,i_{N-1}}
=
\sum_{i_n=1}^{I_n}
A_{j,i_n}
X_{i_0,\ldots,i_n,\ldots,i_{N-1}}
$$

である。

unfoldを使えば、

$$
Y_{(n)}=AX_{(n)}
$$

と理解できる。

---

## 5. factorの転置が「圧縮方向」になる理由

Tuckerのfactorを

$$
U^{(n)}
\in
\mathbb{R}^{I_n\times R_n}
$$

とする。

元のmode次元 $I_n$ をrank次元 $R_n$ へ落とすには、

$$
U^{(n)\mathsf{T}}
\in
\mathbb{R}^{R_n\times I_n}
$$

を掛ける。

$$
I_n
\overset{U^{(n)\mathsf{T}}}{\longrightarrow}
R_n
$$

逆に、rank空間から元の空間へ戻すときは転置しないfactorを使う。

$$
R_n
\overset{U^{(n)}}{\longrightarrow}
I_n
$$

これはConv Tucker-2で、入力側に $U_{\mathrm{in}}^{\mathsf{T}}$、出力側に $U_{\mathrm{out}}$ を使う理由そのものである。

---

## 6. Conv weightでの具体例

$$
W
\in
\mathbb{R}^{64\times32\times3\times3}
$$

に対し、入力channel側factorを

$$
U_{\mathrm{in}}
\in
\mathbb{R}^{32\times R_{\mathrm{in}}}
$$

とする。

mode 1をrankへ圧縮すると、

$$
W\times_1 U_{\mathrm{in}}^{\mathsf{T}}
\in
\mathbb{R}^{64\times R_{\mathrm{in}}\times3\times3}
$$

になる。

出力channel側factor

$$
U_{\mathrm{out}}
\in
\mathbb{R}^{64\times R_{\mathrm{out}}}
$$

でも射影すれば、

$$
W
\times_0 U_{\mathrm{out}}^{\mathsf{T}}
\times_1 U_{\mathrm{in}}^{\mathsf{T}}
\in
\mathbb{R}^{R_{\mathrm{out}}\times R_{\mathrm{in}}\times3\times3}
$$

となり、これがTucker-2のcore shapeになる。

---

## 7. 実装との対応

今回のsrcでは次を共通部品としている。

```text
src/nn_compression/tensor/operations.py
├─ unfold
├─ fold
└─ mode_dot
```

学習Notebook：

```text
notebooks/20_tucker/00_fundamentals/
├─ 00_tucker_hosvd_basics.ipynb
├─ 01_rank_error_tradeoff.ipynb
└─ 02_hooi.ipynb
```

次に読む：

- [[00_基礎理論/01_数学基礎/02_テンソル代数/22_Tucker分解とHOSVD]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/23_HOOI]]
