---
title: TTLinearとdense Linearのforward等価性
tags:
  - TT-matrix
  - TT-Linear
  - Linear
  - forward
  - equivalence
---

# TTLinearとdense Linearのforward等価性

## 1. このノートで示すこと

学習済みのdense線形層をTT-matrixへ変換するとき、最初に確認すべきことは圧縮率ではなく、同じ線形写像を表しているかである。

PyTorchのdense線形層を

$$
Y_{\mathrm{dense}}=XW^{\mathsf T}+b
$$

とし、TT-matrixコアから表される重みを $W_{\mathrm{TT}}$ とする。TTLinearが計算すべき写像は

$$
Y_{\mathrm{TT}}=XW_{\mathrm{TT}}^{\mathsf T}+b
$$

である。したがって、truncationなしのTT-SVDにより

$$
W_{\mathrm{TT}}=W
$$

が成り立ち、TTLinearの直接縮約が同じ添字和を計算していれば、

$$
\boxed{
Y_{\mathrm{TT}}=Y_{\mathrm{dense}}
}
$$

となる。

このノートでは、この等価性を次の順で途中式から確認する。

1. `nn.Linear.weight`の行・列が何を表すか
2. 行列式 $XW^{\mathsf T}+b$ を一要素まで展開する
3. $W_{ij}$ をTT-matrixコアの積へ置き換える
4. 2 siteの直接縮約を全添字で書く
5. 縮約順序、境界rank、bias、flatten順序を区別する
6. forwardの正しさと圧縮効果を別々に判定する

TT-matrixの定義、tensorization、dense reconstruction、一般site数の縮約は、それぞれ次を参照する。

- [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/54_TT-matrixの定義とKronecker積表現]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/55_dense重みのTT-matrix tensorizationとTT-SVD初期化]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/56_TT-matrixのdense reconstruction]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/57_TT-Linear_forwardの縮約とshape]]

## 2. `nn.Linear`が保存する重みの向き

batch sizeを $B$、入力特徴数を $n$、出力特徴数を $m$ とする。

$$
X\in\mathbb R^{B\times n},
\qquad
W\in\mathbb R^{m\times n},
\qquad
b\in\mathbb R^m.
$$

PyTorchの`nn.Linear`は、重みをshape $(m,n)$、biasをshape $(m,)$ で保持する。重みの第0軸が出力特徴、第1軸が入力特徴である。

$$
\boxed{
W[i,j]
=
\text{出力特徴 }i\text{ に対する入力特徴 }j\text{ の重み}
}
$$

変数名 $i_{\mathrm{flat}}$、$j_{\mathrm{flat}}$ は任意だが、$i_{\mathrm{flat}}$ が出力側になることは、重みを $(\mathrm{out\_features},\mathrm{in\_features})$ で保存する規約から決まる。

## 3. なぜforwardでは転置するのに要素は $W_{ij}$ なのか

PyTorchのforwardは

$$
\boxed{
Y=XW^{\mathsf T}+b
}
$$

である。$X$ は $(B,n)$、保存された $W$ は $(m,n)$ なので、$XW$ は内側の次元が一致しない。

$$
(B,n)(m,n)
$$

そこで $W$ を転置し、

$$
(B,n)(n,m)=(B,m)
$$

とする。

具体的に $n=3$、$m=2$ なら、保存された重みは

$$
W=
\begin{pmatrix}
w_{00}&w_{01}&w_{02}\\
w_{10}&w_{11}&w_{12}
\end{pmatrix}
\in\mathbb R^{2\times3}
$$

であり、転置は

$$
W^{\mathsf T}=
\begin{pmatrix}
w_{00}&w_{10}\\
w_{01}&w_{11}\\
w_{02}&w_{12}
\end{pmatrix}
\in\mathbb R^{3\times2}
$$

である。

batch中の一つのサンプルを

$$
X_{\beta,:}
=
\begin{pmatrix}
x_{\beta0}&x_{\beta1}&x_{\beta2}
\end{pmatrix}
$$

とすると、出力の全要素は

$$
\begin{aligned}
Y_{\beta0}
&=x_{\beta0}w_{00}
+x_{\beta1}w_{01}
+x_{\beta2}w_{02}
+b_0,\\
Y_{\beta1}
&=x_{\beta0}w_{10}
+x_{\beta1}w_{11}
+x_{\beta2}w_{12}
+b_1.
\end{aligned}
$$

一般の成分表示は

$$
\begin{aligned}
(XW^{\mathsf T})_{\beta i}
&=
\sum_{j=0}^{n-1}
X_{\beta j}(W^{\mathsf T})_{ji}\\
&=
\sum_{j=0}^{n-1}
X_{\beta j}W_{ij}.
\end{aligned}
$$

したがって、forwardで転置を使っていても、元の保存tensorでは $W[i,j]$ を参照する。転置後のtensorでは、同じ値が $W^{\mathsf T}[j,i]$ にある。

$$
\boxed{
(W^{\mathsf T})_{ji}=W_{ij}
}
$$

数学的には、重みを最初から

$$
\widetilde W\in\mathbb R^{n\times m}
$$

で保存し、$X\widetilde W$ と計算する規約も可能である。PyTorchは各出力ユニットの重みベクトルを一行へ置くため、$(m,n)$ で保存し、forward時に転置する規約を採用している。したがって「どちらが数学的に正しいか」ではなく、保存規約とforward規約を一貫させることが重要である。

## 4. Module、関数、行列式は同じ写像を表す

PyTorchでは、parameterを内部に保持する`nn.Linear`、入力・重み・biasを明示的に受け取る`torch.nn.functional.linear`、および行列式 $XW^{\mathsf T}+b$ が同じ写像を表す。

TTLinearの直接縮約は、dense重みによる積和を、TT-matrixコアを使う積和へ置き換えたものである。各APIの具体的な呼び分けと数値比較は、実装検証ノートへ分ける。

## 5. dense forwardの一要素を一項ずつ計算する

出力の一要素は

$$
Y_{\beta i}
=
b_i+
\sum_{j=0}^{n-1}X_{\beta j}W_{ij}
$$

である。固定した $(\beta,i)$ に対して、まず $b_i$ を取り、$j=0,\ldots,n-1$ の各項 $X_{\beta j}W_{ij}$ を順に足せば、dense forwardの一要素を行列演算なしで再現できる。

この確認はTT固有の処理ではない。TTLinearが最終的に再現すべき対象を、「入力一行」と「重み一行」の内積にbiasを足したものとして固定する役割がある。Python loopによる検証は実装ノートへ置く。

## 6. flat indexを2 siteへ分ける

2 siteの場合、出力特徴数と入力特徴数を

$$
m=m_1m_2,
\qquad
n=n_1n_2
$$

と分解する。0始まりで最後のindexが速く変化する規約では、

$$
i_{\mathrm{flat}}=i_1m_2+i_2,
\qquad
j_{\mathrm{flat}}=j_1n_2+j_2.
$$

逆変換は整数の商と余りである。

$$
i_1=\left\lfloor\frac{i_{\mathrm{flat}}}{m_2}\right\rfloor,
\qquad
i_2=i_{\mathrm{flat}}\bmod m_2.
$$

入力側も同様に、$j_1$ は $j_{\mathrm{flat}}$ を $n_2$ で割った商、$j_2$ は余りである。例えば $m_2=4$、$i_{\mathrm{flat}}=7$ なら、

$$
7=1\cdot4+3
$$

なので、

$$
(i_1,i_2)=(1,3)
$$

である。

tensorization後のinterleaved tensorを $\mathcal W$ とすれば、要素対応は

$$
\boxed{
W[i_1m_2+i_2,\,j_1n_2+j_2]
=
\mathcal W[i_1,j_1,i_2,j_2]
}
$$

である。

## 7. 2 site TT-matrixで重みを表す

2 siteのコアを

$$
G^{(1)}
\in
\mathbb R^{1\times m_1\times n_1\times r_1},
$$

$$
G^{(2)}
\in
\mathbb R^{r_1\times m_2\times n_2\times1}
$$

とする。重みの一要素は

$$
\boxed{
W_{\mathrm{TT}}
[i_1m_2+i_2,\,j_1n_2+j_2]
=
\sum_{\alpha_1=0}^{r_1-1}
G^{(1)}[0,i_1,j_1,\alpha_1]
G^{(2)}[\alpha_1,i_2,j_2,0]
}
$$

である。

端のrankについて、次の三つを混同しない。

- rankのサイズは $r_0=r_2=1$
- 数学で1始まりの添字を使えば、唯一の境界添字は $\alpha_0=\alpha_2=1$
- PyTorchで0始まりの添字を使えば、唯一の境界添字は `0`

式中の `0` はrank値が0という意味ではなく、size 1の軸に存在する唯一の位置である。

## 8. exact 2-core TT-SVDの途中式

interleaved tensorを

$$
\mathcal W
\in
\mathbb R^{m_1\times n_1\times m_2\times n_2}
$$

とし、site 1とsite 2の間で行列化する。

$$
A
=
\operatorname{reshape}
(\mathcal W,(m_1n_1,m_2n_2)).
$$

SVDを

$$
A=U\Sigma V^{\mathsf T}
$$

とする。truncationなしで $r_1=\operatorname{rank}(A)$、またはthin SVDの全列を保持し、

$$
G^{(1)}
=
\operatorname{reshape}
(U,(1,m_1,n_1,r_1)),
$$

$$
G^{(2)}
=
\operatorname{reshape}
(\Sigma V^{\mathsf T},(r_1,m_2,n_2,1))
$$

とする。このgaugeでは第1コアへ $U$、第2コアへ $\Sigma V^{\mathsf T}$ を入れる。

行列要素では

$$
A_{(i_1,j_1),(i_2,j_2)}
=
\sum_{\alpha_1=0}^{r_1-1}
U_{(i_1,j_1),\alpha_1}
(\Sigma V^{\mathsf T})_{\alpha_1,(i_2,j_2)}.
$$

reshape後は、そのまま

$$
\mathcal W[i_1,j_1,i_2,j_2]
=
\sum_{\alpha_1=0}^{r_1-1}
G^{(1)}[0,i_1,j_1,\alpha_1]
G^{(2)}[\alpha_1,i_2,j_2,0]
$$

となる。これがexact TT-SVDによる $W_{\mathrm{TT}}=W$ の内容である。

## 9. 2 site direct contractionを全添字で追う

入力をtensorizeして

$$
\mathcal X
\in
\mathbb R^{B\times n_1\times n_2}
$$

とする。右端コアから $j_2$ を縮約する。

$$
\boxed{
T[\beta,j_1,\alpha_1,i_2]
=
\sum_{j_2=0}^{n_2-1}
\mathcal X[\beta,j_1,j_2]
G^{(2)}[\alpha_1,i_2,j_2,0]
}
$$

shapeは

$$
(B,n_1,n_2)
\longrightarrow
(B,n_1,r_1,m_2)
$$

である。次に $j_1$ と $\alpha_1$ を縮約する。

$$
\boxed{
Z[\beta,i_2,i_1]
=
\sum_{j_1=0}^{n_1-1}
\sum_{\alpha_1=0}^{r_1-1}
T[\beta,j_1,\alpha_1,i_2]
G^{(1)}[0,i_1,j_1,\alpha_1]
}
$$

この実装上の出力軸順は

$$
(B,m_2,m_1)
$$

なので、

$$
(B,m_2,m_1)
\longrightarrow
(B,m_1,m_2)
\longrightarrow
(B,m_1m_2)
$$

と並べ直してflattenする。ここで

$$
i_{\mathrm{flat}}=i_1m_2+i_2
$$

という元のdense行列の行順へ戻る。

上の二式を代入して一つにまとめると、

$$
\begin{aligned}
Z[\beta,i_1,i_2]
&=
\sum_{j_1,j_2,\alpha_1}
\mathcal X[\beta,j_1,j_2]
G^{(1)}[0,i_1,j_1,\alpha_1]
G^{(2)}[\alpha_1,i_2,j_2,0]\\
&=
\sum_{j_1,j_2}
\mathcal X[\beta,j_1,j_2]
W_{\mathrm{TT}}
[i_1m_2+i_2,\,j_1n_2+j_2].
\end{aligned}
$$

したがって、

$$
Z=XW_{\mathrm{TT}}^{\mathsf T}
$$

である。

## 10. 縮約順序はネットワークの見た目だけでは決まらない

入力tensor $\mathcal X[\beta,j_1,\ldots,j_d]$ は、全ての入力index $j_k$ を持つ。したがって、任意のコア $G^{(k)}$ と共有する $j_k$ について縮約できる。

$$
\sum_{j_k}
\mathcal X[\beta,j_1,\ldots,j_d]
G^{(k)}[\alpha_{k-1},i_k,j_k,\alpha_k]
$$

図の上で二つのtensorが横に隣接して見える必要はない。縮約を定義するのは共有indexである。

ただし、TTコア同士では共有bondが必要である。$G^{(1)}$ と $G^{(2)}$ は $\alpha_1$ を共有するが、$G^{(1)}$ と $G^{(3)}$ は直接共有するbondを持たない。この意味で、

- 入力は全ての $j_k$ を持つため、どのsiteからでも作用させられる
- コア同士のchain接続はbond indexに従う

を区別する。

全ての有限和を正しく実行すれば、左からでも右からでも最終値は同じである。一方で、途中tensorのshape、軸順、計算量、実装の読みやすさは異なる。右から左を選ぶことは、数学的な唯一性ではなく実装上の縮約戦略である。

ここで「最終値が同じ」という主張は、共有添字と出力添字を保ったまま有限和の括り方を変えられる、という意味である。テンソルの軸を無断で交換してよい、あるいは行列積 $AB$ と $BA$ が同じ、という意味ではない。添字labelと最終出力軸の順序は保存しなければならない。

## 11. 縮約後の軸順と軸置換

二つのtensorを縮約したときの出力軸は、ここで採用する規約では

$$
\boxed{
A\text{の未縮約軸}
+
B\text{の未縮約軸}
}
$$

の順に並ぶ。縮約した軸だけが消え、他の軸のサイズは変わらない。

軸置換は縮約結果を数学的に正しくするための必須操作とは限らない。次回の縮約軸を毎回明示的に追跡すれば、縮約直後の軸順のまま計算できる。ただし、反復ごとにstateを同じ軸規約へ正規化すれば、同じ縮約式を繰り返し使いやすい。

したがって、

- 軸置換あり：stateの軸順を固定し、次の縮約式を単純にする
- 軸置換なし：軸移動を減らす代わりに、各時点の縮約軸を追跡する

という二つの実装方針がある。ただし最後は、output modeを

$$
(m_1,m_2,\ldots,m_d)
$$

の順に揃えてからflattenしなければならない。

## 12. biasはTT化せず最後に加える

TT-matrixが表すのは重み $W$ に対応する線形写像である。`nn.Linear`全体はbiasを含むaffine mapなので、まず

$$
Z=XW_{\mathrm{TT}}^{\mathsf T}
$$

を計算し、その後

$$
Y_{\mathrm{TT}}=Z+b
$$

とする。

$$
(B,m)+(m)\longrightarrow(B,m)
$$

はbatch軸に対するbroadcastである。最初のcorrectness検証では、元のbias vector $b$ と同じ値をそのまま使えばよく、biasをTT分解する必要はない。

## 13. forward等価性の証明

exact TT-SVDにより全ての要素で

$$
(W_{\mathrm{TT}})_{ij}=W_{ij}
$$

が成り立つとする。すると、

$$
\begin{aligned}
(Y_{\mathrm{TT}})_{\beta i}
&=
\sum_{j=0}^{n-1}
X_{\beta j}(W_{\mathrm{TT}})_{ij}+b_i\\
&=
\sum_{j=0}^{n-1}
X_{\beta j}W_{ij}+b_i\\
&=
(Y_{\mathrm{dense}})_{\beta i}.
\end{aligned}
$$

よって、

$$
\boxed{
W_{\mathrm{TT}}=W
\quad\Longrightarrow\quad
Y_{\mathrm{TT}}=Y_{\mathrm{dense}}
}
$$

である。

数値計算ではSVDと積和の丸め誤差があるため、厳密なbit一致ではなく、dtypeに応じた十分小さい許容誤差で比較する。

## 14. 三段階の検証を混ぜない

実装検証は次の三段階へ分ける。

### 14.1 dense Linear自身の確認

$$
\operatorname{nn.Linear}(X)
\approx
\operatorname{F.linear}(X,W,b)
\approx
XW^{\mathsf T}+b.
$$

ここではPyTorchの重み向きとbiasの扱いを確認する。

### 14.2 TTコアが表す重みの確認

$$
W_{\mathrm{reconstructed}}
\approx
W.
$$

ここではtensorization、TT-SVD、bond縮約、interleavedからgroupedへの復元順を確認する。

### 14.3 denseを作らないforwardの確認

$$
Y_{\mathrm{TT}}
\approx
\operatorname{F.linear}(X,W,b).
$$

ここでは入力index、bond index、出力axis、flatten順、bias加算を確認する。

この分離により、差が出た場合に、

- dense-to-TTのmode順が誤っている
- TTコアからの重み復元が誤っている
- direct contractionの縮約軸が誤っている
- 最終output modeの順が誤っている
- biasが一致していない

のどこを調べるべきか切り分けられる。

## 15. exact表現と圧縮は別の判定である

truncationなしでexact rankを保持すると、TTコアのparameter数が元のdense重みより多くなることがある。2 siteなら、重み部分のparameter数は

$$
N_{\mathrm{TT}}
=
m_1n_1r_1
+
r_1m_2n_2
$$

であり、dense重みは

$$
N_{\mathrm{dense}}=mn=m_1m_2n_1n_2
$$

である。

forward等価性の検証では、rankを落とさず $W_{\mathrm{TT}}\approx W$ を確かめることが目的である。圧縮では、その後にrankを制限し、

- parameter数
- 重み近似誤差
- 層出力誤差
- task accuracy
- fine-tuning後の回復
- 実測時間とメモリ

を別々に評価する。

$$
\boxed{
\text{forwardが正しい}
\neq
\text{parameter圧縮に成功した}
}
$$

## 16. 学習済みFashion-MNIST MLPへ進む前の境界

小さいrandom `nn.Linear`でforward等価性を確認しても、学習済みFashion-MNIST MLPの層置換が完了したことにはならない。実モデルで必要なのは、少なくとも次である。

1. 対象となる学習済み`nn.Linear`を特定する
2. `out_features`と`in_features`に合うmode分解を選ぶ
3. 対象weightをTT-SVDで初期化する
4. 元のbiasを引き継ぐ
5. 層単体で元dense出力とTT出力を比較する
6. モデル全体のlogitsとaccuracyを比較する
7. rank truncation後に圧縮率と精度を評価する
8. 必要ならTTコアをfine-tuningする

この順序を守ることで、添字実装の誤りとrank truncationによる近似誤差を分けられる。

PyTorchで行った2-coreの具体的な確認と保存済み数値結果は、[[08_TT_MPS基礎実装検証/11_TTLinearとdense_Linearのforward等価性のPyTorch確認]]を参照する。

学習済みFashion-MNIST MLPの一層を置き換え、activationと最終logitsまで等価性を追う段階は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/59_TTLinearによるMLP一層置換とlogits等価性]]と[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/00_TTLinear一層置換とlogits等価性]]へ進む。
