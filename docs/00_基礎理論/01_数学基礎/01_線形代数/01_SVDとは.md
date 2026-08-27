---
title: SVDとは
aliases:
  - 特異値分解
  - Singular Value Decomposition
tags:
  - 線形代数
  - SVD
  - 低ランク近似
  - NN圧縮
---

# SVDとは

## サマリー

SVD（Singular Value Decomposition、特異値分解）は、**任意の実行列を、入力側の直交方向・方向ごとの拡大率・出力側の直交方向へ分解する方法**である。

行列

$$
W\in\mathbb R^{m\times n}
$$

が表す写像

$$
W:\mathbb R^n\to\mathbb R^m
$$

を

$$
\boxed{
W=U\Sigma V^{\mathsf T}
}
$$

と分解する。

右から順に

```text
V^T : 入力を右特異ベクトル基底で表す
Σ   : 各特異方向をσ_i倍する
U   : 左特異ベクトル方向へ写す
```

と読める。

小さい特異値に対応する成分を捨てると、固定rankの範囲で元行列をよく近似できる。この低rank近似がNN weight圧縮の基本になる。

---

## 1. SVDが扱う対象

$$
W\in\mathbb R^{m\times n},
\qquad
x\in\mathbb R^n
$$

に対し、

$$
y=Wx\in\mathbb R^m
$$

である。

SVDは $m=n$ の正方行列に限らず、$m\ne n$ の長方形行列にも適用できる。`nn.Linear` weightは一般に長方形なので、この性質が重要である。

---

## 2. 完全SVD

完全SVDでは

$$
W=U\Sigma V^{\mathsf T}
$$

で、

$$
U\in\mathbb R^{m\times m},
\qquad
\Sigma\in\mathbb R^{m\times n},
\qquad
V\in\mathbb R^{n\times n}
$$

である。

$U,V$ は直交行列なので、

$$
U^{\mathsf T}U=UU^{\mathsf T}=I_m
$$

$$
V^{\mathsf T}V=VV^{\mathsf T}=I_n
$$

を満たす。

$U$ の列を $u_i$、$V$ の列を $v_i$ と書く。

$$
U=\begin{bmatrix}u_1&u_2&\cdots&u_m\end{bmatrix}
$$

$$
V=\begin{bmatrix}v_1&v_2&\cdots&v_n\end{bmatrix}
$$

特異値は

$$
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_k\ge0,
\qquad
k=\min(m,n)
$$

と並べる。

完全SVDのshapeは

$$
(m\times m)(m\times n)(n\times n)
=m\times n
$$

で元の $W$ と一致する。

---

## 3. Reduced SVD

数値計算・NN圧縮ではReduced SVDを使うことが多い。

$$
\boxed{
W=U_k\Sigma_kV_k^{\mathsf T}
}
$$

$$
U_k\in\mathbb R^{m\times k},
\qquad
\Sigma_k\in\mathbb R^{k\times k},
\qquad
V_k\in\mathbb R^{n\times k}
$$

なので、

$$
(m\times k)(k\times k)(k\times n)
=m\times n.
$$

PyTorchの

```python
torch.linalg.svd(W, full_matrices=False)
```

はこの形に対応し、`S` は特異値の1次元Tensor、`Vh` は $V_k^{\mathsf T}$ を返す。

詳細なshapeを含む途中導出は [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出]] を参照する。

---

## 4. 特異値と特異ベクトルの意味

第$i$右特異ベクトルへ $W$ を作用させると、

$$
\boxed{
Wv_i=\sigma_i u_i
}
$$

となる。

したがって、

- $v_i$：入力空間 $\mathbb R^n$ の特異方向
- $u_i$：対応する出力空間 $\mathbb R^m$ の特異方向
- $\sigma_i$：その方向の拡大率

である。

$\sigma_i=0$ なら

$$
Wv_i=0
$$

となり、その入力方向はkernel（零空間）へ写る。

逆方向の関係は

$$
\boxed{
W^{\mathsf T}u_i=\sigma_i v_i
}
$$

である。

完全SVDでは転置を途中から書くと

$$
\boxed{
W^{\mathsf T}
=(U\Sigma V^{\mathsf T})^{\mathsf T}
=V\Sigma^{\mathsf T}U^{\mathsf T}
}
$$

であり、長方形 $\Sigma$ の場合に $\Sigma^{\mathsf T}$ を落としてはいけない。

---

## 5. 幾何学的イメージ

$$
y=Wx=U\Sigma V^{\mathsf T}x
$$

は右から順に作用する。

```text
x
↓
V^T
右特異ベクトル基底での座標
↓
Σ
各方向をσ_i倍
↓
U
出力空間へ写す
↓
y
```

直交行列 $Q$ は長さを保つ。

$$
Q^{\mathsf T}Q=I
$$

より、

$$
\begin{aligned}
\|Qx\|_2^2
&=(Qx)^{\mathsf T}(Qx)\\
&=x^{\mathsf T}Q^{\mathsf T}Qx\\
&=x^{\mathsf T}x\\
&=\|x\|_2^2.
\end{aligned}
$$

したがって、

$$
\boxed{
\|Qx\|_2=\|x\|_2
}
$$

であり、SVDで長さを直接変えるのは $\Sigma$ である。

---

## 6. 固有値分解との関係

$$
Wv_i=\sigma_i u_i
$$

へ左から $W^{\mathsf T}$ を掛ける。

$$
W^{\mathsf T}Wv_i
=
\sigma_iW^{\mathsf T}u_i.
$$

さらに

$$
W^{\mathsf T}u_i=\sigma_i v_i
$$

を代入すると、

$$
\begin{aligned}
W^{\mathsf T}Wv_i
&=\sigma_i(\sigma_i v_i)\\
&=\sigma_i^2v_i.
\end{aligned}
$$

よって、

$$
\boxed{
W^{\mathsf T}Wv_i=\sigma_i^2v_i
}
$$

である。

同様に、

$$
\boxed{
WW^{\mathsf T}u_i=\sigma_i^2u_i
}
$$

となる。

したがって、

```text
W^T W の固有ベクトル = v_i
WW^T  の固有ベクトル = u_i
対応固有値             = σ_i^2
```

である。

特異値は

$$
\sigma_i=\sqrt{\lambda_i}
$$

と固有値の非負平方根として読める。

---

## 7. rankと特異値

行列rankは非零特異値の個数に等しい。

$$
\boxed{
\operatorname{rank}(W)
=
\#\{i\mid\sigma_i>0\}
}
$$

浮動小数点では数値的なthreshold $\varepsilon$ を用いて

$$
\operatorname{rank}_{\mathrm{num}}(W)
=
\#\{i\mid\sigma_i>\varepsilon\}
$$

と考える。

---

## 8. SVDをrank-1行列の和として見る

Reduced SVDから

$$
\boxed{
W
=
\sum_{i=1}^{k}
\sigma_i u_i v_i^{\mathsf T}
}
$$

と展開できる。

任意入力 $x$ へ作用させると、

$$
\begin{aligned}
Wx
&=
\sum_{i=1}^{k}
\sigma_i u_i v_i^{\mathsf T}x\\
&=
\sum_{i=1}^{k}
\sigma_i
(v_i^{\mathsf T}x)
u_i.
\end{aligned}
$$

すなわち、入力を右特異方向へ射影し、その係数を $\sigma_i$ 倍して左特異方向へ出力する。

---

## 9. truncated SVD

上位 $r$ 成分だけ残すと、

$$
\boxed{
W_r
=
\sum_{i=1}^{r}
\sigma_i u_i v_i^{\mathsf T}
=
U_r\Sigma_rV_r^{\mathsf T}
}
$$

となる。

$$
U_r\in\mathbb R^{m\times r},
\quad
\Sigma_r\in\mathbb R^{r\times r},
\quad
V_r^{\mathsf T}\in\mathbb R^{r\times n}
$$

なので、

$$
(m\times r)(r\times r)(r\times n)
=m\times n.
$$

また

$$
\operatorname{rank}(W_r)\le r.
$$

---

## 10. Eckart–Young–Mirskyの定理

rankが$r$以下の行列集合に対して、切り詰めSVD $W_r$ はFrobenius normの最小誤差を達成する。

$$
\boxed{
W_r
\in
\arg\min_{\operatorname{rank}(B)\le r}
\|W-B\|_F
}
$$

spectral normについても、

$$
\boxed{
W_r
\in
\arg\min_{\operatorname{rank}(B)\le r}
\|W-B\|_2
}
$$

である。

ここで `∈ argmin` と書くのは、最適解が常に一意とは限らないためである。特に境界で特異値が重複する場合、同じ最小誤差を達成する別のrank-$r$部分空間が存在し得る。

したがって定理の意味は、

> rank$r$以下の別の行列が、切り詰めSVDより**小さい**行列近似誤差を持つことはない。

ということである。

証明スケッチと途中式は [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出#7. Eckart–Young–Mirskyの最適性を式で追う]] を参照する。

---

## 11. Frobenius normと近似誤差

$$
\|A\|_F
=
\sqrt{
\sum_i\sum_j|a_{ij}|^2
}
$$

である。

SVDでは

$$
\boxed{
\|W\|_F^2
=
\sum_{i=1}^{k}\sigma_i^2
}
$$

となる。

切り詰めSVDでは

$$
\boxed{
\|W-W_r\|_F^2
=
\sum_{i=r+1}^{k}\sigma_i^2
}
$$

である。

したがって

$$
\boxed{
\|W-W_r\|_F
=
\sqrt{
\sum_{i=r+1}^{k}\sigma_i^2
}
}
$$

となる。

$r=k$ なら空和なので誤差は0。

---

## 12. spectral normと近似誤差

行列spectral normは

$$
\|A\|_2
=
\max_{\|x\|_2=1}\|Ax\|_2.
$$

切り詰めSVDでは $r<k$ のとき

$$
\boxed{
\|W-W_r\|_2
=
\sigma_{r+1}
}
$$

である。

$r=k$ では

$$
\|W-W_k\|_2=0.
$$

任意入力について

$$
\|(W-W_r)x\|_2
\le
\|W-W_r\|_2\|x\|_2
$$

なので、$r<k$ なら

$$
\boxed{
\|(W-W_r)x\|_2
\le
\sigma_{r+1}\|x\|_2
}
$$

となる。

---

## 13. 特異値energy

上位$r$成分がFrobenius norm二乗の何割を保持するかを

$$
\boxed{
E(r)
=
\frac{\sum_{i=1}^{r}\sigma_i^2}
{\sum_{i=1}^{k}\sigma_i^2}
}
$$

で表す。

相対Frobenius誤差は

$$
\varepsilon_F(r)
=
\frac{\|W-W_r\|_F}{\|W\|_F}
$$

なので、

$$
\begin{aligned}
\varepsilon_F(r)^2
&=
\frac{\sum_{i=r+1}^{k}\sigma_i^2}
{\sum_{i=1}^{k}\sigma_i^2}\\
&=
1-
\frac{\sum_{i=1}^{r}\sigma_i^2}
{\sum_{i=1}^{k}\sigma_i^2}\\
&=1-E(r).
\end{aligned}
$$

したがって、

$$
\boxed{
\varepsilon_F(r)=\sqrt{1-E(r)}
}
$$

である。

energy保持率とclassification accuracy保持率は別物である。

---

## 14. NN weightの2因子化

Linear weightを

$$
W\in\mathbb R^{D_{\mathrm{out}}\times D_{\mathrm{in}}}
$$

とする。

$$
W_r
=U_r\Sigma_rV_r^{\mathsf T}
$$

に対して、例えば

$$
A
=
\Sigma_rV_r^{\mathsf T}
\in
\mathbb R^{r\times D_{\mathrm{in}}}
$$

$$
B
=
U_r
\in
\mathbb R^{D_{\mathrm{out}}\times r}
$$

と置けば、

$$
\boxed{W_r=BA}
$$

である。

元の

$$
y=Wx+b
$$

を

$$
h=Ax
$$

$$
y=Bh+b
$$

へ置き換えると、

$$
y=BAx+b=W_rx+b.
$$

2層の間に非線形関数を入れると $BAx$ ではなくなるため、単純なSVD置換では挟まない。

---

## 15. 実装上のSVD

NumPy：

```python
U, S, Vh = np.linalg.svd(W, full_matrices=False)
```

PyTorch：

```python
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
```

ここで

```text
U  : (m, k)
S  : (k,)
Vh : (k, n)
```

である。

rank$r$なら

```python
U_r = U[:, :r]
S_r = S[:r]
Vh_r = Vh[:r, :]
W_r = (U_r * S_r) @ Vh_r
```

と書ける。

---

## 16. よくある誤解

### SVDは正方行列だけ

誤り。長方形行列にも適用できる。

### $U$ が入力方向、$V$ が出力方向

このノートの約束

$$
W:\mathbb R^n\to\mathbb R^m
$$

では逆である。

$$
v_i\in\mathbb R^n
\xrightarrow{W}
\sigma_i u_i\in\mathbb R^m
$$

なので、$v_i$ が入力側、$u_i$ が出力側。

### 特異値の単純和を99%残す

Frobenius energyでは

$$
\sum_i\sigma_i^2
$$

を使う。

### low rankなら必ず推論が速い

parameters / MACsが減っても、kernel起動やハードウェア利用効率によりwall-clock latencyは改善しない場合がある。

---

## 17. このノートで押さえるポイント

- $W=U\Sigma V^{\mathsf T}$。
- $Wv_i=\sigma_i u_i$ なので $v_i$ は入力側、$u_i$ は出力側。
- 完全SVDの転置は $W^{\mathsf T}=V\Sigma^{\mathsf T}U^{\mathsf T}$。
- $W^{\mathsf T}Wv_i=\sigma_i^2v_i$、$WW^{\mathsf T}u_i=\sigma_i^2u_i$。
- truncated SVDは固定rankでFrobenius / spectral normの最良近似を与えるが、最適解が常に一意とは限らない。
- $\|W-W_r\|_F^2=\sum_{i>r}\sigma_i^2$。
- $r<k$ なら $\|W-W_r\|_2=\sigma_{r+1}$、full rankなら0。
- $\varepsilon_F(r)=\sqrt{1-E(r)}$。
- NNではweight近似の良さとtask accuracyを分けて評価する。

---

## 18. 次に読むノート

- [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出]]
- [[00_基礎理論/02_ニューラルネットワーク基礎/02_nn.Linearとは]]
- [[00_基礎理論/01_数学基礎/01_線形代数/03_SVDによる低ランク近似]]
- [[00_基礎理論/03_モデル圧縮理論/04_Linear層を2層へ置き換える]]
- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]
- [[50_DMRG/01_SVDからDMRGへのつながり]]
