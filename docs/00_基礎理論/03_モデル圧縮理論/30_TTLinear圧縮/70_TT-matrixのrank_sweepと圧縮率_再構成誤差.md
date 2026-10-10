---
title: TT-matrixのrank sweepと圧縮率・再構成誤差
tags:
  - TT-matrix
  - TT-rank
  - rank-sweep
  - parameter-count
  - reconstruction-error
---

# TT-matrixのrank sweepと圧縮率・再構成誤差

## 1. このノートの到達点

tensorizationを固定したTT-matrixについて、共通内部rankだけを変え、次の2本の関係を分離して評価する。

$$
\boxed{
\text{TT-rank}
\longrightarrow
\text{parameter count}
\longrightarrow
\text{compression ratio}
}
$$

$$
\boxed{
\text{TT-rank}
\longrightarrow
\text{weight reconstruction error}
}
$$

最初のrank sweepでは、weightそのものの保存量と再構成誤差だけを見る。logits error、prediction agreement、classification accuracy、fine-tuningは次の段階へ分ける。

これより前には、

$$
\Delta W
\longrightarrow
\Delta L
\longrightarrow
\text{sample-wise prediction stability}
\longrightarrow
\text{batch内全予測不変の十分条件}
$$

を学んだ。この誤差伝播と予測保証は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/68_TTLinear圧縮の誤差上界と予測安定性_不等式の全体像]]へまとめている。本ノートはいったんその後半から離れ、上流にある

$$
\text{rank}
\longrightarrow
\Delta W
$$

のうち、rankとweight reconstructionの関係だけを調べる。

bond dimensionが表現能力を制限する理由は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/69_TT-matrixのbond_dimensionと表現能力]]を前提とする。

## 2. 固定するものと変えるもの

dense weightを

$$
W\in\mathbb R^{m\times n}
$$

とし、

$$
m=\prod_{k=1}^{d}m_k,
\qquad
n=\prod_{k=1}^{d}n_k
$$

とtensorizeする。

rankの影響だけを調べるため、全候補で次を固定する。

- 元のweight $W$
- 乱数seed
- 出力mode列 $(m_1,\ldots,m_d)$
- 入力mode列 $(n_1,\ldots,n_d)$
- dtypeとdevice
- TT-SVDとdense reconstructionの添字順

変えるのは内部rankの上限だけである。簡単なsweepでは、

$$
(r_0,r_1,\ldots,r_{d-1},r_d)
=
(1,r,\ldots,r,1)
$$

とし、スカラー $r$ を候補集合から選ぶ。

「rank $=4$」は、テンソルの行列rankが4という意味ではなく、

$$
r_1=r_2=\cdots=r_{d-1}=4
$$

という共通内部rank上限を指定する意味である。

## 3. requested rankとactual rankを分ける

指定した $r$ はrank上限である。第 $k$ 段階のSVD行列のshapeが $(a_k,b_k)$ なら、利用できる特異成分は最大

$$
\min(a_k,b_k)
$$

個である。従って実際に保持できるbond dimensionは、少なくともshape上

$$
r_k^{\mathrm{actual}}
\le
\min\{r,a_k,b_k\}
$$

となる。数値rankを追加判定する実装なら、さらに小さくなることもある。

そこで結果には、

$$
\text{requested rank}=r
$$

と、core shapeから読み取った

$$
\text{actual rank vector}
=
(1,r_1^{\mathrm{actual}},\ldots,r_{d-1}^{\mathrm{actual}},1)
$$

を別々に記録する。

## 4. TT-matrixのparameter count

第 $k$ coreは

$$
G^{(k)}
\in
\mathbb R^{r_{k-1}\times m_k\times n_k\times r_k}
$$

なので、保存する要素数は

$$
r_{k-1}m_kn_kr_k
$$

である。core自体を加算するのではなく、各coreの要素数を加算する。

$$
\boxed{
N_{\mathrm{TT}}
=
\sum_{k=1}^{d}
r_{k-1}m_kn_kr_k
}
$$

$N_{\mathrm{TT}}$ は全coreへ実際に保存するスカラー数である。Gauge自由度を差し引いた多様体の自由度ではない。このノートではweightだけを数え、biasは含めない。

dense weightの保存要素数は

$$
N_{\mathrm{dense}}=mn.
$$

## 5. 3-core・$8\times8$ のparameter countを全項から求める

固定tensorizationを

$$
(m_1,m_2,m_3)=(2,2,2),
\qquad
(n_1,n_2,n_3)=(2,2,2)
$$

とする。このとき

$$
m=n=8,
\qquad
N_{\mathrm{dense}}=8\cdot8=64.
$$

共通内部rankを

$$
(r_0,r_1,r_2,r_3)=(1,r,r,1)
$$

と指定する。core shapeと要素数は

$$
G^{(1)}:(1,2,2,r),
\qquad
\#G^{(1)}=1\cdot2\cdot2\cdot r=4r,
$$

$$
G^{(2)}:(r,2,2,r),
\qquad
\#G^{(2)}=r\cdot2\cdot2\cdot r=4r^2,
$$

$$
G^{(3)}:(r,2,2,1),
\qquad
\#G^{(3)}=r\cdot2\cdot2\cdot1=4r.
$$

従って、

$$
\begin{aligned}
N_{\mathrm{TT}}
&=4r+4r^2+4r\\
&=\boxed{8r+4r^2}.
\end{aligned}
$$

中央coreは左右にrank軸を持つため、保存量が $r^2$ に比例する。$r$ を2倍にしても、TT全体の保存量が単純に2倍になるとは限らない。

具体的には、

$$
\begin{array}{c|c|c}
r&\text{rank vector}&N_{\mathrm{TT}}\\ \hline
1&(1,1,1,1)&4+4+4=12\\
2&(1,2,2,1)&8+16+8=32\\
4&(1,4,4,1)&16+64+16=96
\end{array}
$$

である。

## 6. compression ratioの定義と読み方

この教材では、weightのcompression ratioを

$$
\boxed{
C
:=
\frac{N_{\mathrm{dense}}}{N_{\mathrm{TT}}}
}
$$

と定義する。

$$
\boxed{
\begin{array}{ll}
C>1&\text{TTの保存要素数がdenseより少ない},\\
C=1&\text{保存要素数が同じ},\\
C<1&\text{TTの保存要素数がdenseより多い}.
\end{array}
}
$$

3-core・$8\times8$ の例では、

$$
C(r)=\frac{64}{8r+4r^2}
$$

なので、

$$
\begin{array}{c|c|c|c}
r&N_{\mathrm{TT}}&C&\text{parameter count上の判定}\\ \hline
1&12&64/12\approx5.33&\text{圧縮}\\
2&32&64/32=2&\text{圧縮}\\
4&96&64/96\approx0.67&\text{非圧縮}
\end{array}
$$

となる。$C=2$ はTTの保存要素数がdenseの半分という意味であり、推論速度が2倍という意味ではない。実メモリ量にはdtypeや保存形式も、実行時間には縮約順序やkernelも影響する。

## 7. dense reconstructionと二つの誤差

rank上限 $r$ でTT-SVDし、全coreを縮約してdense shapeへ戻したweightを

$$
\widetilde W(r)
\in
\mathbb R^{m\times n}
$$

とする。

### 7.1 absolute Frobenius reconstruction error

$$
\boxed{
E_{\mathrm{abs}}(r)
:=
\|W-\widetilde W(r)\|_F
}
$$

である。全要素へ展開すると、

$$
E_{\mathrm{abs}}(r)
=
\sqrt{
\sum_{i=1}^{m}
\sum_{j=1}^{n}
\left(
W_{ij}-\widetilde W(r)_{ij}
\right)^2
}.
$$

### 7.2 relative Frobenius reconstruction error

$\|W\|_F>0$ のとき、

$$
\boxed{
E_{\mathrm{rel}}(r)
:=
\frac{\|W-\widetilde W(r)\|_F}{\|W\|_F}
}
$$

とする。これはweight全体のnormに対する誤差の割合であり、各要素が同じ割合だけずれるという意味ではない。

### 7.3 小さい行列での確認

$$
W=
\begin{pmatrix}
3&0&0\\
0&2&0\\
0&0&1
\end{pmatrix},
\qquad
\widetilde W=
\begin{pmatrix}
3&0&0\\
0&2&0\\
0&0&0
\end{pmatrix}
$$

なら、

$$
\begin{aligned}
E_{\mathrm{abs}}
&=
\sqrt{(1-0)^2}
=1,\\
\|W\|_F
&=
\sqrt{3^2+2^2+1^2}
=\sqrt{14},\\
E_{\mathrm{rel}}
&=
\frac{1}{\sqrt{14}}
\approx0.2673.
\end{aligned}
$$

## 8. rank上限を増やすと、最良近似誤差は悪化しない

固定tensorizationのもと、全内部rankが $r$ 以下のTT-matrixで表せるweightの集合を

$$
\mathcal S_r
:=
\left\{
B\in\mathbb R^{m\times n}
\ \middle|\
\operatorname{TT\text{-}rank}(B)\le(1,r,\ldots,r,1)
\right\}
$$

とする。

追加したbond成分をゼロにすれば、小さいrankの表現を大きいrankでも表せる。従って、

$$
\mathcal S_1
\subseteq
\mathcal S_2
\subseteq
\mathcal S_4.
$$

最良TT近似誤差を

$$
E_{\mathrm{best}}(r)
:=
\min_{B\in\mathcal S_r}
\|W-B\|_F
$$

と定義する。候補集合が広がるので、

$$
\boxed{
E_{\mathrm{best}}(4)
\le
E_{\mathrm{best}}(2)
\le
E_{\mathrm{best}}(1)
}
$$

である。「必ず小さくなる」ではなく「悪化しない」である。

## 9. 最良TT近似と逐次TT-SVDの実測誤差は別物

rank sweepで実際に計算するのは、TT-SVDが返した1つの近似 $\widetilde W_{\mathrm{TT\text{-}SVD}}(r)$ の誤差

$$
E_{\mathrm{TT\text{-}SVD}}(r)
:=
\left\|
W-\widetilde W_{\mathrm{TT\text{-}SVD}}(r)
\right\|_F
$$

である。

$E_{\mathrm{best}}(r)$ は許されるTT全体から選んだ理論上の最小値、$E_{\mathrm{TT\text{-}SVD}}(r)$ は逐次アルゴリズムが返した1候補の値である。従って、

$$
E_{\mathrm{best}}(r)
\le
E_{\mathrm{TT\text{-}SVD}}(r).
$$

TT-SVDは各段階で、その段階のSVD行列に対する最良rank制限近似を作る。しかし、前段で選んだ部分空間が後段の入力行列を変えるため、全coreを同時に選ぶ大域最良TT近似と一致するとは限らない。

$$
\boxed{
\text{各段階の行列近似が最良}
\ \not\Rightarrow\
\text{TT全体の近似が最良}
}
$$

これは反復最適化が局所解へ陥るという話ではない。TT-SVDが全coreを同時最適化せず、左から右へ逐次構成することによる違いである。

## 10. TT-SVDの準最適性をrank sweepでどう読むか

標準TT-SVDには、core数を $d$ として

$$
\boxed{
E_{\mathrm{best}}(r)
\le
E_{\mathrm{TT\text{-}SVD}}(r)
\le
\sqrt{d-1}\,E_{\mathrm{best}}(r)
}
$$

という準最適性保証がある。

この導出では、[[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/47_TT-SVDの準最適性と誤差上界]]の記号を使う。同ノートでは、

- $\varepsilon_k$：元の第 $k$ cut unfoldingに対する最良rank-$r_k$行列近似誤差
- $\delta_k$：TT-SVDの第 $k$ 段階で実際に捨てた局所誤差

であり、

$$
\delta_1=\varepsilon_1,
\qquad
\delta_k\le\varepsilon_k
\quad(k\ge2),
$$

$$
\|W-\widetilde W_{\mathrm{TT\text{-}SVD}}\|_F^2
=
\sum_{k=1}^{d-1}\delta_k^2
$$

が成り立つ。さらに各cutで

$$
\varepsilon_k
\le
E_{\mathrm{best}}(r)
$$

なので、

$$
\begin{aligned}
E_{\mathrm{TT\text{-}SVD}}(r)^2
&=
\sum_{k=1}^{d-1}\delta_k^2\\
&\le
\sum_{k=1}^{d-1}\varepsilon_k^2\\
&\le
(d-1)E_{\mathrm{best}}(r)^2.
\end{aligned}
$$

この証明で使う中間行列 $M_k$、縮約 $Q_{k-1}^{\mathsf T}\otimes I$、Frobeniusノルムの非増大、直交誤差の二乗和は、リンク先で添字と具体行列まで展開している。

## 11. 準最適性から実測誤差のrank単調性は直接出ない

rank上限を増やすと

$$
E_{\mathrm{best}}(4)
\le
E_{\mathrm{best}}(2)
$$

である。また各rankで

$$
E_{\mathrm{TT\text{-}SVD}}(r)
\le
\sqrt{d-1}\,E_{\mathrm{best}}(r)
$$

である。しかし、この二つだけから

$$
E_{\mathrm{TT\text{-}SVD}}(4)
\le
E_{\mathrm{TT\text{-}SVD}}(2)
$$

は導けない。異なるrankで、実測値がそれぞれ別の許容区間に入ることと、実測値同士の順序は別だからである。

従って、multi-coreのrank sweepでは、実測誤差の単調非増加を一般定理として必須assertにしない。今回のデータでどうなったかを観察し、結果として記録する。

coreが2個ならSVDは1回だけであり、Eckart--Young--Mirsky定理からtruncated SVDが大域最良行列近似になる。この場合はrank上限増加に対する誤差の非増加を直接言える。multi-coreへ同じ結論を無条件に移さない。

## 12. 3-core rank sweepの設計

小規模確認では、

$$
W\in\mathbb R^{8\times8},
$$

$$
(m_1,m_2,m_3)
=
(n_1,n_2,n_3)
=
(2,2,2),
$$

$$
r\in\{1,2,4\}
$$

とする。

combined physical sizeは

$$
s_k=m_kn_k=4
$$

なので、TT-SVDが扱う3階テンソルのshapeは

$$
(s_1,s_2,s_3)=(4,4,4)
$$

である。

第1cutと第2cutのunfolding shapeは、

$$
A^{\langle1\rangle}:(4,16),
\qquad
A^{\langle2\rangle}:(16,4).
$$

従ってどちらのcut rankも最大4であり、requested rank $r=4$ ならshape上の全特異成分を保持できる。exact reconstructionが期待される一方、保存量は

$$
N_{\mathrm{TT}}=96>64=N_{\mathrm{dense}}
$$

なので、parameter count上は圧縮にならない。

## 13. sweepで各rankについて行う処理

各 $r$ について、次の順に処理する。

1. 同じ $W$ を、同じtensorizationでTT-SVDする。
2. core shapeからactual rank vectorを読む。
3. 各coreの要素数を足して $N_{\mathrm{TT}}$ を求める。
4. coreを縮約して $\widetilde W(r)$ をdense reconstructionする。
5. compression ratio、absolute error、relative errorを求める。

最低限、次を記録する。

$$
\begin{array}{l}
\text{requested rank},\\
\text{actual rank vector},\\
N_{\mathrm{TT}},\\
C=N_{\mathrm{dense}}/N_{\mathrm{TT}},\\
E_{\mathrm{abs}},\\
E_{\mathrm{rel}},\\
\text{parameter count上で圧縮かどうか}.
\end{array}
$$

## 14. このrank sweepで分かることと分からないこと

### 分かること

- 固定tensorizationでrank上限を変えたときのcore shape
- parameter countとcompression ratio
- TT-SVDが実際に返したweightのabsolute／relative error
- exact reconstructionとparameter compressionが両立するか
- 今回のweightで実測誤差がどう変化したか

### まだ分からないこと

- Fashion-MNISTでのprediction agreement
- ground-truthに対するaccuracy
- certified accuracy
- logits errorとclassification margin
- fine-tuning後の回復
- tensorization候補間の優劣
- 不均一rankや自動rank選択
- Pareto frontier

weight reconstruction errorが小さいことは重要だが、それだけで分類精度の維持は保証されない。次段階では、既に導出した

$$
\Delta W
\longrightarrow
\Delta L
\longrightarrow
\text{prediction stability}
$$

へ接続する。

## 15. PyTorch確認への接続

3-core TT-SVD、軸順、dense reconstruction、rank sweep、`records[1:]`・`zip`・`all` の読み方、保存済み数値結果は、[[08_TT_MPS基礎実装検証/20_TT-matrixのrank_sweepと圧縮誤差のPyTorch確認]]へ分ける。
