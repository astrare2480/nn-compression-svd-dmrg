---
title: TT-Linear forwardの縮約とshape
tags:
  - TT-matrix
  - TT-Linear
  - contraction
  - einsum
  - shape
---

# TT-Linear forwardの縮約とshape

## 1. dense Linearと同じ入出力規約を使う

batch sizeを $B$ とし、PyTorchの`Linear`と同じ規約を使う。

$$
X\in\mathbb R^{B\times n},
\qquad
W\in\mathbb R^{m\times n},
\qquad
b\in\mathbb R^m.
$$

forwardは

$$
\boxed{Y=XW^{\mathsf T}+b}
$$

であり、要素表示は

$$
Y[b,i]
=
\sum_{j=0}^{n-1}X[b,j]W[i,j]+b[i].
$$

ここでbatch sizeは一度に処理するサンプル数であり、特徴次元やTT-rankではない。外部から受け取る入力は通常の2階tensor

$$
X:(B,n)
$$

である。forward内部でのみ

$$
X\longrightarrow
\mathcal X\in
\mathbb R^{B\times n_1\times\cdots\times n_d}
$$

へreshapeする。

## 2. TT-matrixを代入する

出力と入力を

$$
m=\prod_{k=1}^{d}m_k,
\qquad
n=\prod_{k=1}^{d}n_k
$$

と分ける。TT-matrixコアを代入すると、

$$
\begin{aligned}
&\mathcal Y[b,i_1,\ldots,i_d]\\
&=
\sum_{j_1,\ldots,j_d}
\sum_{\alpha_1,\ldots,\alpha_{d-1}}
\mathcal X[b,j_1,\ldots,j_d]
\prod_{k=1}^{d}
G^{(k)}[\alpha_{k-1},i_k,j_k,\alpha_k].
\end{aligned}
$$

forwardの本質は、各siteで

$$
\boxed{j_k\text{ を縮約して消し、対応する }i_k\text{ を残す}}
$$

ことである。中央siteでは同時に

$$
\boxed{\alpha_k\text{ を消し、左bond }\alpha_{k-1}\text{ を残す}}
$$

ので、右から左へbond情報を運べる。

右から始める理由は、reshape直後の入力で $j_d$ が最後のaxisにあり、最後のコアも $(\alpha_{d-1},i_d,j_d,1)$ を持つためである。最初の縮約で $j_d$ を直接消し、$i_d$ と次に必要な $\alpha_{d-1}$ を残せる。左から縮約する定式化も可能だが、ここでは入力の末尾axisと右端rank 1を利用するとshapeを追いやすいため、右から左を採用する。

## 3. $d=3$ の右から左への途中式

入力とコアを

$$
\mathcal X[b,j_1,j_2,j_3],
$$

$$
G^{(1)}:(1,m_1,n_1,r_1),
$$

$$
G^{(2)}:(r_1,m_2,n_2,r_2),
$$

$$
G^{(3)}:(r_2,m_3,n_3,1)
$$

とする。

### Step 3

最後の入力index $j_3$ を縮約する。

$$
\boxed{
T_3[b,j_1,j_2,i_3,\alpha_2]
=
\sum_{j_3}
\mathcal X[b,j_1,j_2,j_3]
G^{(3)}[\alpha_2,i_3,j_3,0]
}
$$

shapeは

$$
(B,n_1,n_2,n_3)
\longrightarrow
(B,n_1,n_2,m_3,r_2)
$$

である。

### Step 2

次に $j_2$ と現在のbond $\alpha_2$ を縮約する。

$$
\boxed{
T_2[b,j_1,i_2,i_3,\alpha_1]
=
\sum_{j_2,\alpha_2}
T_3[b,j_1,j_2,i_3,\alpha_2]
G^{(2)}[\alpha_1,i_2,j_2,\alpha_2]
}
$$

shapeは

$$
(B,n_1,n_2,m_3,r_2)
\longrightarrow
(B,n_1,m_2,m_3,r_1)
$$

である。

### Step 1

最後に $j_1$ と $\alpha_1$ を縮約する。

$$
\boxed{
\mathcal Y[b,i_1,i_2,i_3]
=
\sum_{j_1,\alpha_1}
T_2[b,j_1,i_2,i_3,\alpha_1]
G^{(1)}[0,i_1,j_1,\alpha_1]
}
$$

shapeは

$$
(B,n_1,m_2,m_3,r_1)
\longrightarrow
(B,m_1,m_2,m_3)
$$

である。最後に

$$
(B,m_1,m_2,m_3)
\longrightarrow
(B,m_1m_2m_3)
$$

へreshapeし、biasを加える。

## 4. 入力indexが「消える」とは和を取ることである

例えば

$$
T_3[b,j_1,j_2,i_3,\alpha_2]
=
\sum_{j_3}
\mathcal X[b,j_1,j_2,j_3]
G^{(3)}[\alpha_2,i_3,j_3,0]
$$

では、固定した $(b,j_1,j_2,i_3,\alpha_2)$ に対して、$j_3$ の全値を掛け合わせて足す。$j_3$ は出力tensorの自由indexではなくなり、代わりに $i_3$ と $\alpha_2$ が残る。

`einsum`では、入力側に現れるが出力側に書かれないlabelについて積和を取る。

```python
next_state = torch.einsum(
    "...j,pij->...ip",
    state,
    last_core,
)
```

ここで

- `j` は $j_d$
- `p` は $\alpha_{d-1}$
- `i` は $i_d$
- `...` はbatchと未処理の入力軸

を表す。

## 5. 端のコアが特別になる理由

中央コアは

$$
G^{(k)}:(r_{k-1},m_k,n_k,r_k)
$$

で左右に非自明なbondを持つ。一方、両端は

$$
G^{(1)}:(1,m_1,n_1,r_1),
$$

$$
G^{(d)}:(r_{d-1},m_d,n_d,1)
$$

である。右から左へ縮約する実装では、

1. 最後のコアで右端rank 1を落とし、最初の状態を作る
2. 中央コアで $(j_k,\alpha_k)$ を縮約する
3. 最初のコアで $(j_1,\alpha_1)$ を縮約し、左端rank 1も消す

と分けるとshapeを追いやすい。

## 6. 一般 $d$ のstate不変条件

site $k+1,\ldots,d$ を処理済みとする。縮約前のstateを

$$
(B,n_1,\ldots,n_k,m_{k+1},\ldots,m_d,r_k)
$$

という順で持つ。

```text
(batch, 未処理input, 処理済みoutput, 現在bond)
```

このままでは今回消す $j_k$ と $\alpha_k$ の間に処理済みoutputが入る。したがって、$j_k$ をbondの直前へ移す。

$$
(B,n_1,\ldots,n_{k-1},m_{k+1},\ldots,m_d,j_k,\alpha_k).
$$

そこで同じ縮約

$$
(\ldots,j_k,\alpha_k)
\times
(\alpha_{k-1},i_k,j_k,\alpha_k)
\longrightarrow
(\ldots,i_k,\alpha_{k-1})
$$

を繰り返せる。

Pythonのloop indexを `k` とし、`cores[k]` が数学上のsite $k+1$ を表す実装では、消したい $n_{k+1}$ はstateのaxis `k + 1` にある。

```python
for k in range(len(cores) - 2, 0, -1):
    # cores[k] は数学上の site k+1。
    # その入力軸を現在bondの直前へ移す。
    state = state.movedim(k + 1, state.ndim - 2)

    state = torch.einsum(
        "...ja,pija->...ip",
        state,
        cores[k],
    )
```

数学上のsite番号とPythonの0始まりindexを混ぜると、`k` と `k + 1` を取り違える。どちらの記号を採用したかをコメントで固定する。

## 7. なぜ`movedim`は中央の各反復で必要か

$d=3$ で最後のコアを処理した直後を

$$
(B,n_1,n_2,m_3,r_2)
$$

とする。$n_2$ をbondの直前へ動かせば

$$
(B,n_1,m_3,n_2,r_2)
$$

となり、`"...ja,pija->...ip"` を適用できる。その結果は

$$
(B,n_1,m_3,m_2,r_1)
$$

である。次に消す $n_1$ はbondの直前ではないため、再び移動が必要である。

「最初に一度だけ出力軸を移せば、中央loopでは常に末尾二軸が $(j_k,\alpha_k)$ になる」という考えは一般には成り立たない。各反復で新しい $i_k$ が挿入されるためである。正しい不変条件は、毎回どのaxisが $j_k$ かを追跡し、bond直前へ移してから縮約することである。

## 8. `einsum`のoperand順も式の一部である

次の式では、第1operandが `"...ja"`、第2operandが `"pija"` に対応する。

```python
state = torch.einsum(
    "...ja,pija->...ip",
    state,
    cores[k],
)
```

coreを先に渡したいなら、文字列も同時に入れ替える。

```python
state = torch.einsum(
    "pija,...ja->...ip",
    cores[k],
    state,
)
```

operandだけを入れ替え、添字列をそのままにすると各tensorの階数・軸意味が一致しない。

## 9. 最初のコアとoutput軸の反転

中央loop後、$n_1$ をbondの直前へ移し、最初のコアを処理する。

```python
state = state.movedim(1, state.ndim - 2)

state = torch.einsum(
    "...ja,pija->...i",
    state,
    cores[0],
)
```

左端bond $p$ も出力に書かれていないため和で消えるが、size 1なので値は変わらない。別案として、`cores[0].squeeze(0)` を使い

```python
state = torch.einsum(
    "...ja,ija->...i",
    state,
    cores[0].squeeze(0),
)
```

と書いてもよい。

この一般実装では処理済みoutputを右から追加するため、最後は

$$
(B,m_d,m_{d-1},\ldots,m_1)
$$

になっている。batch軸を固定し、それ以外を逆順にする。

```python
d = len(cores)
perm = [0] + list(range(d, 0, -1))
state = state.permute(*perm)
```

これにより

$$
(B,m_d,\ldots,m_1)
\longrightarrow
(B,m_1,\ldots,m_d)
$$

となる。`permute`後はnon-contiguousの場合があるため、最後は`view`より`reshape`が安全である。

## 10. rank 1、2 siteの数値例

$$
(m_1,m_2)=(2,2),
\qquad
(n_1,n_2)=(2,2),
\qquad
(r_0,r_1,r_2)=(1,1,1)
$$

とする。

$$
A=
\begin{pmatrix}
1&2\\
3&4
\end{pmatrix},
\qquad
B=
\begin{pmatrix}
5&6\\
7&8
\end{pmatrix}.
$$

TT-matrixコアは

$$
G^{(1)}=\operatorname{reshape}(A,(1,2,2,1)),
$$

$$
G^{(2)}=\operatorname{reshape}(B,(1,2,2,1))
$$

であり、重みは

$$
W=A\otimes B
=
\begin{pmatrix}
5&6&10&12\\
7&8&14&16\\
15&18&20&24\\
21&24&28&32
\end{pmatrix}.
$$

入力を

$$
X=
\begin{pmatrix}
1&2&3&4
\end{pmatrix}
$$

とし、

$$
\mathcal X=
\begin{pmatrix}
1&2\\
3&4
\end{pmatrix}
$$

へreshapeする。

まず第2siteを処理する。

$$
T[j_1,i_2]
=
\sum_{j_2}
\mathcal X[j_1,j_2]B[i_2,j_2].
$$

$j_1=0$ では

$$
T[0,0]=1\cdot5+2\cdot6=17,
$$

$$
T[0,1]=1\cdot7+2\cdot8=23.
$$

$j_1=1$ では

$$
T[1,0]=3\cdot5+4\cdot6=39,
$$

$$
T[1,1]=3\cdot7+4\cdot8=53.
$$

したがって

$$
T=
\begin{pmatrix}
17&23\\
39&53
\end{pmatrix}.
$$

次に第1siteを処理する。

$$
\mathcal Y[i_1,i_2]
=
\sum_{j_1}T[j_1,i_2]A[i_1,j_1].
$$

全要素は

$$
\mathcal Y[0,0]=1\cdot17+2\cdot39=95,
$$

$$
\mathcal Y[0,1]=1\cdot23+2\cdot53=129,
$$

$$
\mathcal Y[1,0]=3\cdot17+4\cdot39=207,
$$

$$
\mathcal Y[1,1]=3\cdot23+4\cdot53=281.
$$

よって

$$
Y=
\begin{pmatrix}
95&129&207&281
\end{pmatrix}.
$$

dense計算でも

$$
XW^{\mathsf T}
=
\begin{pmatrix}
95&129&207&281
\end{pmatrix}
$$

となる。index変換として見ると

$$
(j_1,j_2)
\longrightarrow
(j_1,i_2)
\longrightarrow
(i_1,i_2)
$$

である。

## 11. `einsum`を通常の行列積から読む

通常の行列積

$$
C[i,k]
=
\sum_j A[i,j]B[j,k]
$$

は

```python
C = torch.einsum("ij,jk->ik", A, B)
```

と書ける。`j`は両operandに現れるが出力`ik`には現れないため、掛けてから $j$ について和を取る。`i`と`k`は出力側に書かれているため残る。

2 site、rank 1のTT-Linearでも同じ規則を使う。入力を

$$
\mathcal X\in\mathbb R^{B\times n_1\times n_2}
$$

とすると、第2siteは

$$
T[b,j_1,i_2]
=
\sum_{j_2}
\mathcal X[b,j_1,j_2]B[i_2,j_2]
$$

なので、例えばlabelを

$$
b=\text{batch},
\quad
p=j_1,
\quad
q=j_2,
\quad
s=i_2
$$

と割り当てれば

```python
T = torch.einsum("bpq,sq->bps", X_tensor, B)
```

となる。次に

$$
\mathcal Y[b,i_1,i_2]
=
\sum_{j_1}T[b,j_1,i_2]A[i_1,j_1]
$$

は

```python
Y_tensor = torch.einsum("bps,rp->brs", T, A)
```

である。出力文字列`brs`は値だけでなく出力axis順 $(b,i_1,i_2)$ も指定する。`einsum`を読むときは、

1. 各文字がどのindexか
2. 出力に書かれず縮約される文字は何か
3. 出力文字がどの順に並ぶか

を分けて確認する。

## 12. 検証を三つに分ける

random TT-matrixコアが定義するdense重みを $W_{\mathrm{TT}}$ とする。次を別々に比較する。

$$
Y_{\mathrm{TT}}
=
\operatorname{TTLinear}(X,\{G^{(k)}\},b),
$$

$$
Y_{\mathrm{dense}}
=
XW_{\mathrm{TT}}^{\mathsf T}+b,
$$

$$
Y_{\mathrm{F.linear}}
=
\operatorname{F.linear}(X,W_{\mathrm{TT}},b).
$$

- $Y_{\mathrm{TT}}=Y_{\mathrm{dense}}$：コア直接縮約とdense reconstructionの添字が一致する。
- $Y_{\mathrm{dense}}=Y_{\mathrm{F.linear}}$：PyTorchのweight orientationが一致する。
- $Y_{\mathrm{TT}}=Y_{\mathrm{F.linear}}$：TT-Linearが通常のLinearと同じ写像を表す。

比較対象はrandomコアが定義する $W_{\mathrm{TT}}$ である。truncated TT-SVDで元の $W$ を近似した場合、$Y_{\mathrm{TT}}$ と $XW^{\mathsf T}+b$ の差にはforward実装誤差だけでなく重み近似誤差も含まれる。

`nn.Linear.weight`の向きからexact TT-SVD後のforward等価性までを一続きで確認する場合は、[[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/58_TTLinearとdense_Linearのforward等価性]]を参照する。PyTorchでの一般実装contractと保存済み数値結果は [[08_TT_MPS基礎実装検証/10_TT-matrix_Dense_Reconstruction_TT-Linear_ForwardのPyTorch確認]]、2-core `nn.Linear`等価性の具体例は [[08_TT_MPS基礎実装検証/11_TTLinearとdense_Linearのforward等価性のPyTorch確認]]、別方式の一括`einsum`実装は [[08_TT_MPS基礎実装検証/45_TT-matrix線形層のPyTorch直接forward]] を参照する。
