---
title: TuckerとTTの比較
aliases:
  - Tucker vs TT
  - TuckerとTensor Train
  - TuckerとMPS
tags:
  - Tucker
  - TT
  - TensorDecomposition
  - TensorNetwork
  - 比較
---

# TuckerとTTの比較

## サマリー

TuckerとTTはどちらも高階テンソルを低rank構造で表すが、「情報をどこに置くか」が大きく異なる。

- Tucker：各modeをfactor matrixで縮め、全mode間の関係を中央のdense coreへ集める。
- TT：中央の巨大coreを持たず、各modeの小さな3階coreをbondで鎖状につなぐ。

この構造差により、rankの意味、保存量、高階化したときのスケーリング、適した利用場面が変わる。

---

## 1. 式で比較

元テンソルを

$$
X_{i_1\cdots i_d}
\in
\mathbb R^{n_1\times\cdots\times n_d}
$$

とする。

### Tucker

$$
\boxed{
X_{i_1\cdots i_d}
\approx
\sum_{a_1,\ldots,a_d}
\mathcal C_{a_1\cdots a_d}
U^{(1)}_{i_1,a_1}
\cdots
U^{(d)}_{i_d,a_d}
}
$$

factorは

$$
U^{(k)}\in\mathbb R^{n_k\times R_k},
$$

中央coreは

$$
\mathcal C
\in
\mathbb R^{R_1\times\cdots\times R_d}.
$$

### TT

$$
\boxed{
X_{i_1\cdots i_d}
\approx
\sum_{\alpha_1,\ldots,\alpha_{d-1}}
G^{(1)}_{1,i_1,\alpha_1}
G^{(2)}_{\alpha_1,i_2,\alpha_2}
\cdots
G^{(d)}_{\alpha_{d-1},i_d,1}
}
$$

各coreは

$$
G^{(k)}
\in
\mathbb R^{r_{k-1}\times n_k\times r_k},
\qquad
r_0=r_d=1.
$$

---

## 2. 構造の違い

```text
Tucker

 i1      i2                  id
 |       |                    |
U(1)    U(2)                U(d)
 \       |                  /
      dense central core C
```

Tuckerでは、全mode間の相互作用を中央core

$$
\mathcal C_{a_1\cdots a_d}
$$

がまとめて持つ。

一方TTは

```text
 i1         i2                    id
 |          |                      |
G(1) --r1-- G(2) --r2-- ... -- G(d)
```

となり、相関を

$$
\alpha_1,\ldots,\alpha_{d-1}
$$

というbondへ分散して保持する。

---

## 3. 分解手順の違い

HOSVD型のTuckerでは、各mode factorは元の同じテンソルから独立に求める。

```text
X → mode-1 unfolding → SVD → U^(1)
X → mode-2 unfolding → SVD → U^(2)
...
```

全factorを得た後、

$$
\mathcal C
=
X
\times_1U^{(1)T}
\times_2U^{(2)T}
\cdots
\times_dU^{(d)T}
$$

と中央coreへ射影する。

TT-SVDは異なる。

```text
X
→ 第1cutをSVD
→ U^(1)をcoreとして固定
→ ΣV^Tだけを次へ渡す
→ 次のphysical indexと前のbondをまとめて再SVD
→ ...
```

つまりTTは**逐次的**であり、前段で得たbond indexを次段へ引き継ぐ。

---

## 4. rankの意味が違う

$R_k$ 本のfactor列からなる基底は、一つの物理mode $i_k$ の空間を近似する基底である。
一方、TTの $r_k$ 本の左ブロック基底は、物理modeを $1$ から $k$ までまとめた空間に属する。
例えば第2cutの左状態は $|i_1,i_2\rangle$ の線形結合であり、サイト2だけの $|i_2\rangle$ の線形結合ではない。
そのため、$R_2$ と $r_2$ に同じ整数を指定しても、同じ部分空間を残すという条件にはならない。

[[31_TT-rankとunfolding]] のrank反例は、この違いを同じTensorの具体行列で確認するための例である。
本章では数値例を再掲せず、以下のrank定義の「行にまとめる物理添字」を比較する。
なお、定義上のexact rankと、圧縮時に指定するfactor列数・bondサイズも別である。
後者は保持する容量の指定であり、そのサイズの表現から復元したTensorの実際のrankが必ず等しいとは限らない。

### Tucker rank

Tuckerのmultilinear rankは各mode-n unfoldingのrankである。

$$
R_k
=
\operatorname{rank}(X_{(k)}).
$$

これは「第 $k$ modeだけ」と「その他すべて」を分けるrankである。

### TT-rank

TT-rankは左から順にまとめたcut unfoldingのrankである。

$$
r_k
=
\operatorname{rank}
\left(X^{\langle k\rangle}\right).
$$

$$
(i_1,\ldots,i_k)
\mid
(i_{k+1},\ldots,i_d)
$$

というbipartitionを測る。

したがって、Tucker rankとTT-rankは同じ「rank」という名前でも、見ている行列化が異なる。

---

## 5. パラメータ数

### 境界coreと内部coreを分けて数える

まず不均一な次元とrankのまま数えると、保存する全要素数は

$$
P_{\mathrm{Tucker}}
=\prod_{k=1}^{d}R_k+\sum_{k=1}^{d}n_kR_k,
\qquad
P_{\mathrm{TT}}=\sum_{k=1}^{d}r_{k-1}n_kr_k,
\qquad r_0=r_d=1.
$$

$d\ge2$、全physical dimensionを $n$、全Tucker rankと内部TT bondを $r$ とする場合、
Tuckerのcoreは $r$ 要素の軸が $d$ 本、factorは $n\times r$ 行列が $d$ 枚なので

$$
P_{\mathrm{Tucker}}=\underbrace{r\cdots r}_{d\text{ 本}}
+\underbrace{nr+\cdots+nr}_{d\text{ 枚}}=r^d+dnr.
$$

TTの第1・最終coreはそれぞれ $1\times n\times r$、
$r\times n\times1$、残る $d-2$ 個は $r\times n\times r$ なので

$$
\begin{aligned}
P_{\mathrm{TT}}
&=1\cdot n\cdot r
+\sum_{k=2}^{d-1}r\cdot n\cdot r+r\cdot n\cdot1\\
&=nr+(d-2)nr^2+nr\\
&=2nr+(d-2)nr^2.
\end{aligned}
$$

これは指定した一様なshapeでの正確な**保存要素数**であり、
Gauge自由度を除いた独立自由度の数ではない。
以下の概算表記はこの一様rankの仮定を省略したものである。

簡単のため全mode sizeを $n$、Tucker/TT rankを概ね $r$、階数を $d$ とする。

### dense

$$
P_{\mathrm{dense}}=n^d.
$$

### Tucker

中央coreが $r^d$、factorが $d$ 個で各 $nr$ なので、

$$
\boxed{
P_{\mathrm{Tucker}}
\approx
r^d+dnr
}
$$

となる。

### TT

端のcoreは概ね $nr$、中央coreは $nr^2$ なので、

$$
\boxed{
P_{\mathrm{TT}}
\approx
2nr+(d-2)nr^2
=
O(dnr^2)
}
$$

である。

このため、rankが一定程度に抑えられるなら、TTは階数 $d$ に対してほぼ線形に増える。一方Tuckerは中央core $r^d$ が高階で大きくなりやすい。

### 6階・各軸4・rank 3を最後まで数える

具体例として、

$$
X\in\mathbb R^{4\times4\times4\times4\times4\times4},
\qquad d=6,
\qquad n_k=4
$$

を考える。元のdenseテンソルの要素数は

$$
\begin{aligned}
P_{\mathrm{dense}}
&=4\cdot4\cdot4\cdot4\cdot4\cdot4\\
&=4^6=4096.
\end{aligned}
$$

Tuckerでは、全factorの列数を $R_1=\cdots=R_6=3$ と指定する。中央coreのshapeは $(3,3,3,3,3,3)$、各factorのshapeは $(4,3)$ なので、

$$
\begin{aligned}
P_{\mathrm{core}}
&=3\cdot3\cdot3\cdot3\cdot3\cdot3=3^6=729,\\
P_{\mathrm{factors}}
&=\underbrace{4\cdot3+4\cdot3+\cdots+4\cdot3}_{6\text{ 枚}}\\
&=6\cdot4\cdot3=72,\\
P_{\mathrm{Tucker}}
&=729+72=801.
\end{aligned}
$$

TTでは内部bondを $r_1=\cdots=r_5=3$、境界を $r_0=r_6=1$ とする。第1coreは $(1,4,3)$、第2から第5coreは各 $(3,4,3)$、第6coreは $(3,4,1)$ だから、

$$
\begin{aligned}
P_1&=1\cdot4\cdot3=12,\\
P_2=P_3=P_4=P_5&=3\cdot4\cdot3=36,\\
P_6&=3\cdot4\cdot1=12,\\
P_{\mathrm{TT}}
&=12+36+36+36+36+12\\
&=12+4\cdot36+12\\
&=12+144+12=168.
\end{aligned}
$$

この指定shapeで保存する要素数は、denseが4096、Tuckerが801、TTが168となる。TTの全coreを内部coreと同じ形だとみなして $6\cdot3\cdot4\cdot3=216$ と数えると、サイズ1の境界bondを見落としてしまう。

ただし、Tuckerのfactor列数3とTTのbondサイズ3は、同じ近似誤差を指定する条件ではない。ここで比較したのは指定shapeの保存要素数であり、任意の $X$ を同精度で近似できることや、Gauge自由度を除いた独立自由度の比較ではない。

---

## 6. 3階・4階ではTuckerも扱いやすい

TTの $O(dnr^2)$ という形は高階テンソルで特に強い。一方、3階や4階程度でmodeごとの意味が明確ならTuckerの中央coreはまだ扱いやすく、factorごとの圧縮解釈も直感的である。

例えば画像・CNN weightの

$$
C_{\mathrm{out}}\times C_{\mathrm{in}}\times k_h\times k_w
$$

のような4階テンソルでは、出力channelと入力channelだけを縮めるTucker-2が自然な場合がある。

TTは、Dense weightを多数のinput/output modeへtensorizeして高階化する場合や、本来から高階の状態・関数を扱う場合に鎖構造の利点が出やすい。

---

## 7. 物理的な違い

Tuckerは一般の多線形低rank分解として使われ、中央coreが全mode間の結合をまとめて持つ。

TTは開放境界MPSと同じ構造なので、各bondが

$$
(1,\ldots,k)\mid(k+1,\ldots,d)
$$

という切断を自然に持つ。

このためTT-rankは物理的にbond dimension / Schmidt rankとして読める。canonical form、entanglement、DMRGへ直接つながるのはTT/MPS側の特徴である。

---

## 8. どちらもSVDを使うが役割が違う

Tucker/HOSVDでは、各mode unfoldingから「そのmodeを表すfactor basis」を求める。

TT-SVDでは、各bipartitionでSVDし、左側のbasisを現在coreへ取り出し、残りを右へ渡す。

したがって

```text
Tucker:
各modeの基底を同じ元Tensorから集める
→ 中央coreへ情報を集約

TT:
左から順にbasisを確定する
→ 情報をbondへ段階的に分散
```

という違いがある。

---

## 9. まとめ表

| 観点 | Tucker | TT / MPS |
| --- | --- | --- |
| 基本構造 | factor matrices + dense central core | 3階coreのchain |
| rank | mode-n unfolding rank | sequential cut unfolding rank |
| 情報の置き場所 | 中央coreへ集約 | bondへ分散 |
| 高階化 | core $r^d$ が増えやすい | rank一定なら概ね $O(dnr^2)$ |
| 分解 | 各modeを元Tensorから独立SVD | remainderを左から逐次SVD |
| 物理解釈 | 一般の多線形分解 | MPS、Schmidt、bond dimension |
| NNでの自然な例 | Conv Tucker-2 | 高階tensorized Dense / TT-matrix |

---

## 10. このプロジェクトでの位置づけ

このリポジトリでは

```text
SVD
→ Tucker / HOSVD / HOOI
→ TT / MPS
→ gauge / canonical form
→ MPO / TT-matrix
→ DMRG
```

と段階的につなぐ。

Tuckerで学んだ

- unfolding
- SVD
- rank truncation
- Frobenius誤差
- factor basis

はTTでも生きる。一方、TTでは新たに

- sequential cut
- bond index
- chained cores
- basisを右へ渡す逐次構造
- MPS / Schmidt分解

が加わる。

---

## 11. 層ごとに分解形式を選ぶhybrid圧縮

層ごとに圧縮形式を選ぶ方法を考える。
一つのモデルの全層を同じ形式にする必要はなく、
非圧縮・SVD・Tucker-2・TT/MPSを候補として比較できる。
「LinearにはSVD、3×3 ConvにはTucker、大channelにはTT」は候補の例であり、
層の種類だけで最良形式を決められる規則ではない。
実装可能な因子化forwardを持たない候補は、実用的な圧縮層としては選べない。

### rank配分と形式選択は別の選択である

層 $\ell$ の候補を

$$
q_\ell=(m_\ell,\,\boldsymbol r_\ell,\,\tau_\ell)
\in\mathcal Q_\ell,
\qquad
m_\ell\in\{\mathrm{dense},\mathrm{SVD},\mathrm{Tucker2},\mathrm{TT}\}
$$

と書く。$m_\ell$ は形式、$\boldsymbol r_\ell$ は形式に応じたrank指定、
$\tau_\ell$ はTTなどで必要なtensorizationの指定である。
denseではrank・tensorizationの指定を省略できる。
SVDのrank一つ、Tucker-2の二つのfactor rank、TTのbond列は、同じ条件ではない。
候補集合 $\mathcal Q_\ell$ は、層のshape・対応実装・rank上限を満たすものだけに限定する。

複数層でSVD rankだけを選ぶrank allocationは、$m_\ell$ を固定した部分問題である。
hybridでは形式そのものも候補に含める。
異なる単位の指標を比較できるようcostを正規化して書くと、例えば

$$
c_\ell(q_\ell)
=\alpha\frac{P_\ell(q_\ell)}{P_{\ell,\mathrm{ref}}}
+\beta\frac{M_\ell(q_\ell)}{M_{\ell,\mathrm{ref}}}
+\gamma\frac{t_\ell(q_\ell)}{t_{\ell,\mathrm{ref}}}
+\lambda\frac{E_\ell(q_\ell)}{E_{\ell,\mathrm{ref}}}
$$

である。$P_\ell$ は保存するparameter数、$M_\ell$ は同じ入力shapeでの理論MACs、
$t_\ell$ は固定条件で測った層latency、$E_\ell$ は共通calibration入力に対する層出力RMSEとする。
参照値はすべて正、重み $\alpha,\beta,\gamma,\lambda$ は非負とする。
denseの出力誤差は0なので、$E_{\ell,\mathrm{ref}}$ にその0を使わず、正の許容誤差などを指定する。
重みと参照値を変えると選択結果も変わるため、実験条件として記録する。

共有parameterを持たない直列層ではparameter数を加算でき、
同じ入力条件・MACsの定義なら理論MACsも加算できる。
固定部分の値を $P_{\mathrm{fixed}},M_{\mathrm{fixed}}$ として、全体予算内の候補選択を

$$
\begin{aligned}
\min_{q_\ell\in\mathcal Q_\ell}\quad
&\sum_{\ell=1}^{L}c_\ell(q_\ell),\\
\text{subject to}\quad
&P_{\mathrm{fixed}}+\sum_{\ell=1}^{L}P_\ell(q_\ell)\le P_{\mathrm{budget}},\\
&M_{\mathrm{fixed}}+\sum_{\ell=1}^{L}M_\ell(q_\ell)\le M_{\mathrm{budget}}
\end{aligned}
$$

と表せる。biasを残すなら各候補の $P_\ell$ に含め、
共有parameterがあるモデルでは単純な層別加算で二重計上しない。
層latencyの和や層出力RMSEの和は、モデル全体の実測latency・logit誤差とは一般に一致しない。
このcostは**探索用の代理指標**であり、全体精度の最適性を保証する目的関数ではない。

### 二層・四候補を全要素の行列で計算する

以下は説明用の仮想候補で、実測した実験結果ではない。
二層に四形式の候補があり、行は層1・層2、列はdense・SVD・Tucker-2・TTの順とする。
rankやtensorizationは候補ごとに固定済みとし、固定部分のparameter数・MACsは0と置く。
parameter数とMACsを、全要素で

$$
\boldsymbol P
=\begin{pmatrix}
100&60&55&45\\
200&130&100&80
\end{pmatrix},
\qquad
\boldsymbol M
=\begin{pmatrix}
1000&650&600&550\\
2000&1400&1100&950
\end{pmatrix}
$$

とする。層latencyの単位をms、層出力誤差をRMSEとして、

$$
\boldsymbol t
=\begin{pmatrix}
1.0&0.8&0.9&1.2\\
2.0&1.5&1.3&1.8
\end{pmatrix}\ \mathrm{ms},
\qquad
\boldsymbol E
=\begin{pmatrix}
0&0.10&0.15&0.20\\
0&0.12&0.18&0.25
\end{pmatrix}
$$

とする。TTのparameter数が最小でも、この仮想例の層latencyは最小ではない。
RMSEの単位は層出力の単位であり、accuracy低下を表す百分率ではない。
両層で参照値と重みを

$$
\begin{aligned}
P_{\ell,\mathrm{ref}}&=100,&
M_{\ell,\mathrm{ref}}&=1000,&
t_{\ell,\mathrm{ref}}&=1\ \mathrm{ms},&
E_{\ell,\mathrm{ref}}&=0.1,\\
(\alpha,\beta,\gamma,\lambda)&=(0.4,0.3,0.2,0.1)
\end{aligned}
$$

と置く。例えば層1のSVDと層2のTTは、各項を代入して

$$
\begin{aligned}
c_1(\mathrm{SVD})
&=0.4\frac{60}{100}+0.3\frac{650}{1000}
+0.2\frac{0.8}{1}+0.1\frac{0.10}{0.1}\\
&=0.240+0.195+0.160+0.100=0.695,\\
c_2(\mathrm{TT})
&=0.4\frac{80}{100}+0.3\frac{950}{1000}
+0.2\frac{1.8}{1}+0.1\frac{0.25}{0.1}\\
&=0.320+0.285+0.360+0.250=1.215
\end{aligned}
$$

となる。同じ計算を全候補へ行うと、costの全要素は

$$
\boldsymbol c
=\begin{pmatrix}
0.900&0.695&0.730&0.785\\
1.800&1.360&1.170&1.215
\end{pmatrix}
$$

である。
層1・層2の候補番号を $a,b\in\{0,1,2,3\}$ とすると、組合せparameter数は
$P_{1,a}+P_{2,b}$ である。
行を層1の形式、列を層2の形式として全16組合せを展開すると、

$$
\boldsymbol P_{\mathrm{pair}}
=\begin{pmatrix}
300&230&200&180\\
260&190&160&140\\
255&185&155&135\\
245&175&145&125
\end{pmatrix}.
$$

$P_{\mathrm{budget}}=150$、$M_{\mathrm{budget}}=1700$ と置くと、
両予算を満たす候補は次の四つである。

$$
\begin{aligned}
(\mathrm{SVD},\mathrm{TT}) &: P=60+80=140,
&M&=650+950=1600,&C&=0.695+1.215=1.910,\\
(\mathrm{Tucker2},\mathrm{TT}) &: P=55+80=135,
&M&=600+950=1550,&C&=0.730+1.215=1.945,\\
(\mathrm{TT},\mathrm{Tucker2}) &: P=45+100=145,
&M&=550+1100=1650,&C&=0.785+1.170=1.955,\\
(\mathrm{TT},\mathrm{TT}) &: P=45+80=125,
&M&=550+950=1500,&C&=0.785+1.215=2.000.
\end{aligned}
$$

ここで $C$ は二層costの和である。
予算なしの最小候補 $(\mathrm{SVD},\mathrm{Tucker2})$ は、
$C=0.695+1.170=1.865$ でも $P=60+100=160>150$ のため除外する。
予算内では $(\mathrm{SVD},\mathrm{TT})$ がこの代理costで最小となり、
parameter数だけで選んだ $(\mathrm{TT},\mathrm{TT})$ とは異なる。

この「最小」は、固定した候補・重み・代理costで全16組合せを調べた結果に限る。
すべてのrank・tensorization・学習法を含むglobal optimumでも、accuracyの最高値でもない。
実験では同じbaselineから各組合せを独立に作り、
validationでモデル全体の出力誤差・accuracy・latency・peak memoryを改めて測る。
前段層の圧縮で後段層への入力も変わるため、単独候補の出力誤差だけでは全体性能を確定できない。
FTの有無・予算をそろえ、候補選択用validationと最終testを分ける。

指標の基本導出は [[10_SVD圧縮モデルの評価設計]]、
[[11_理論計算量とベンチマーク]]、[[25_Tucker_HOOI圧縮の評価設計]] を参照する。
ここでは形式選択の組合せ計算に限定し、DMRGの局所更新や新しい探索実装は追加しない。
