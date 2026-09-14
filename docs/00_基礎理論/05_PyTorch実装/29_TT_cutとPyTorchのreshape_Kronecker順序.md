---
title: TT cutとPyTorchのreshape・Kronecker順序
aliases:
  - TT cut reshape順
  - TT unfoldingとmode unfolding
  - PyTorch reshapeとkron
  - TTの置換行列
tags:
  - TT
  - PyTorch
  - reshape
  - KroneckerProduct
  - permutation
---

# TT cutとPyTorchのreshape・Kronecker順序

## サマリー

本章のmode名は数学の1始まりである。「mode-2」は物理添字 $i_2$ に対応する第2軸を指し、PyTorchのaxis番号では $1$ になる。
[[21_テンソルとmode演算]] とTucker実装の章はmode番号も0始まりで数えるため、同じ第2軸を「mode 1」と呼ぶ。
本章の `permute(1, 0, 2)` の整数はPyTorchの0始まりaxis番号であり、数式のmode-2と矛盾しない。
mode名の番号だけで比較せず、「どの物理添字を行側へ置いたか」を合わせる。

TT/MPS基礎実装では、次の三つを混同しないことが重要である。

1. TTの第2cut

$$
(i_1,i_2)\mid i_3
$$

2. Tuckerで使うmode-2 unfolding

$$
i_2\mid(i_1,i_3)
$$

3. 第1 SVD後のbasis変換

$$
X_{\mathrm{cut2}}
=
(U\otimes I_{n_2})B_{\mathrm{cut2}}.
$$

PyTorchでは、

```python
X_cut2 = X.reshape(n1 * n2, n3)
B_cut2 = B.reshape(r1 * n2, n3)
L2 = torch.kron(U, torch.eye(n2))
```

の三つが同じ複合index順で揃っているため、明示的な置換行列 $P$ をコードへ入れる必要はない。

---

## 1. TTの第2cut

3階Tensor

$$
X\in\mathbb R^{n_1\times n_2\times n_3}
$$

について、TTのcutは左から連続して切る。

```text
第1 TT cut:
i1 | i2 i3

第2 TT cut:
i1 i2 | i3
```

したがって第2cutは

$$
\boxed{(i_1,i_2)\mid i_3}
$$

である。

PyTorchでは、元のaxis順

```text
(i1, i2, i3)
```

を変えずに先頭2軸をまとめればよい。

```python
X_cut2 = X.reshape(n1 * n2, n3)
```

shapeは

$$
\boxed{
X_{\mathrm{cut2}}
\in
\mathbb R^{(n_1n_2)\times n_3}
}
$$

である。

---

## 2. `i2 | (i1,i3)` はTTの第2cutではない

同じcutの中で $(i_1,i_2)$ の行位置を入れ替えるだけなら、行の置換であり、rankは変わらない。
しかし $(i_1,i_2)\mid i_3$ を $i_2\mid(i_1,i_3)$ に変えると、$i_1$ が行側から列側へ移る。
これは同じ行列の行・列を並べ替える操作とは異なり、どの空間同士を分けるかという分割そのものの変更である。
元Tensorの値を捨ててはいないが、出来上がる行列のrankは異なることがある。
後で扱う置換行列 $P$ は、同じ分割内の複合添字の順序をそろえる道具であって、任意のTT cutとmode unfoldingのrankを等しくする道具ではない。

もし本当に

$$
i_2\mid(i_1,i_3)
$$

へ行列化したいなら、これはmode-2 unfoldingである。

元のaxis順は

```text
(i1, i2, i3)
```

なので、$i_2$ を先頭へ移す必要がある。

```python
X_mode2 = X.permute(1, 0, 2).reshape(
    n2,
    n1 * n3,
)
```

shapeは

$$
X_{(2)}
\in
\mathbb R^{n_2\times(n_1n_3)}.
$$

したがって、

| コード | 分割 | 用途 |
| --- | --- | --- |
| `X.reshape(n1*n2, n3)` | $(i_1,i_2)\mid i_3$ | TTの第2cut |
| `X.permute(1,0,2).reshape(n2,n1*n3)` | $i_2\mid(i_1,i_3)$ | mode-2 unfolding |

である。

これは [[31_TT-rankとunfolding]] で説明した「TT cut unfolding」と「Tucker mode-n unfolding」の実装上の具体例である。

---

## 3. `reshape`で複合indexはどう並ぶか

```python
X_cut2 = X.reshape(n1 * n2, n3)
```

では、行番号は実質

$$
\boxed{
\operatorname{row}_X
=
i_1n_2+i_2
}
$$

である。

例えば

```text
i1 = 0, 1, 2
i2 = 0, 1
```

なら行順は

```text
(i1, i2)
(0,0)
(0,1)
(1,0)
(1,1)
(2,0)
(2,1)
```

となる。

$i_2$ が内側のindexとして先に変化する。

---

## 4. 第1 SVD後のremainder `B`

第1cut

$$
i_1\mid(i_2,i_3)
$$

をexact SVDして、

$$
X^{\langle1\rangle}
=
U\Sigma V^T
$$

とする。

TT-SVDでは

$$
B
=
\Sigma V^T
$$

を右へ渡す。

3階へ戻せば、

$$
B
\in
\mathbb R^{r_1\times n_2\times n_3}
$$

で、添字は

$$
B_{\alpha_1,i_2,i_3}
$$

である。

重要なのは、第3サイト $i_3$ が消えていないことである。

第2 SVD直前の行列は

```python
B_cut2 = B.reshape(r1 * n2, n3)
```

で、

$$
B_{\mathrm{cut2}}
\in
\mathbb R^{(r_1n_2)\times n_3}.
$$

行indexは

$$
(\alpha_1,i_2),
$$

列indexは

$$
i_3
$$

である。

---

## 5. `B`側の複合index順

```python
B_cut2 = B.reshape(r1 * n2, n3)
```

の行番号は実質

$$
\boxed{
\operatorname{row}_B
=
\alpha_1n_2+i_2
}
$$

である。

したがって行順は

```text
(α1, i2)
(0,0)
(0,1)
...
(1,0)
(1,1)
...
```

となる。

元の $X$ と $B$ はどちらも、第2cutの列側に同じ $i_3$ を残す。

| 行列 | 行index | 列index | shape |
| --- | --- | --- | --- |
| `X_cut2` | $(i_1,i_2)$ | $i_3$ | $(n_1n_2,n_3)$ |
| `B_cut2` | $(\alpha_1,i_2)$ | $i_3$ | $(r_1n_2,n_3)$ |

違うのは行側のbasisだけである。

---

## 6. `torch.kron(U, I)`はどの順に作用するか

$$
U\in\mathbb R^{n_1\times r_1}
$$

とし、

```python
I_n2 = torch.eye(
    n2,
    dtype=U.dtype,
    device=U.device,
)

L2 = torch.kron(U, I_n2)
```

とする。

$$
L_2
=
U\otimes I_{n_2}.
$$

成分は

$$
\boxed{
(L_2)_{i_1n_2+i_2,\;\alpha_1n_2+j_2}
=
U_{i_1,\alpha_1}
(I_{n_2})_{i_2,j_2}
}
$$

である。

単位行列なので、

$$
(I_{n_2})_{i_2,j_2}
=
\delta_{i_2,j_2}.
$$

したがって、

$$
\boxed{
(L_2)_{(i_1,i_2),(\alpha_1,j_2)}
=
U_{i_1,\alpha_1}\delta_{i_2,j_2}
}
$$

である。

つまり、

```text
入力:  (α1, i2)
          ↓
       Uは α1 → i1
       Iは i2 をそのまま通す
          ↓
出力:  (i1, i2)
```

という変換になる。

---

## 7. なぜ `X_cut2 = L2 @ B_cut2` がそのまま成立するか

三つの複合index順を並べる。

```text
X_cut2 の行:
(i1, i2)

B_cut2 の行:
(α1, i2)

L2 = U ⊗ I の作用:
(α1, i2) → (i1, i2)
```

すべて同じ「第1indexを外側、第2indexを内側」という順に揃っている。

したがって、

$$
\boxed{
X_{\mathrm{cut2}}
=
(U\otimes I_{n_2})B_{\mathrm{cut2}}
}
$$

をPyTorch上でも直接検証できる。

```python
transformed = L2 @ B_cut2

relation_error = torch.linalg.matrix_norm(
    X_cut2 - transformed,
    ord="fro",
).item()
```

exact SVDなら `relation_error` は丸め誤差程度になる。

---

## 8. $I\otimes U$ が出るblock順

同じ添字ペアを、別の順に並べることもできる。

例えば `i2` ごとにblockを作るなら、

```text
(i2, i1)
(0,0)
(0,1)
(0,2)
(1,0)
(1,1)
(1,2)
```

のように並ぶ。

この順では、「各 $i_2$ blockへ同じ $U$ を作用させる」という見た目になり、

$$
I_{n_2}\otimes U
$$

がblock対角的に現れる。

$$
I_{n_2}\otimes U
=
\begin{bmatrix}
U&0&\cdots\\
0&U&\cdots\\
\vdots&\vdots&\ddots
\end{bmatrix}.
$$

しかし、これはPyTorchの上記 `reshape` と**異なる行順**を採用している。

---

### 元資料の $3\times2$ 行列：blockの中身と順序を省略しない

添付 `TT_MPS基礎理論.md` の27239–27383行にある例を、元と同じ
$n_1=3,r_1=2,n_2=2$ で回収する。本節の添字はPyTorchと同じ0始まりである。

$$
U_1=\begin{pmatrix}a&b\\c&d\\e&f\end{pmatrix},\qquad
B^{(j)}_{\alpha_1,i_3}=B_{\alpha_1,j,i_3},\qquad
B^{(0)},B^{(1)}\in\mathbb R^{2\times n_3}.
$$

「block」は $i_2=j$ を固定したsliceである。$i_3$ は各sliceの列に残り、
サイト2の値が違う二つのsliceへ、同じ $U_1$ を別々に掛ける。

$$
\begin{aligned}
B_{\mathrm{block}}
&=\begin{pmatrix}B^{(0)}\\B^{(1)}\end{pmatrix}
\in\mathbb R^{4\times n_3},\\
X_{\mathrm{block}}
&=\begin{pmatrix}U_1B^{(0)}\\U_1B^{(1)}\end{pmatrix}
=\begin{pmatrix}U_1&0\\0&U_1\end{pmatrix}
\begin{pmatrix}B^{(0)}\\B^{(1)}\end{pmatrix}.
\end{aligned}
$$

このときの0は $3\times2$ の零blockである。まず、sliceの中身を確認する。

元資料27190–27217行の $i_2=0$ に固定したsliceの中身も、
行のphysical/bond添字を明記して

$$
X^{(0)}=
\begin{pmatrix}
X_{0,0,0}&\cdots&X_{0,0,n_3-1}\\
\vdots&\ddots&\vdots\\
X_{n_1-1,0,0}&\cdots&X_{n_1-1,0,n_3-1}
\end{pmatrix},\qquad
B^{(0)}=
\begin{pmatrix}
B_{0,0,0}&\cdots&B_{0,0,n_3-1}\\
\vdots&\ddots&\vdots\\
B_{r_1-1,0,0}&\cdots&B_{r_1-1,0,n_3-1}
\end{pmatrix},\qquad X^{(0)}=U_1B^{(0)}
$$

となる。どちらも第2添字は固定され、第3添字 $i_3$ は列のままである。
同じことを $i_2=j$ ごとに行うため、一般のblock対角行列は

$$
I_{n_2}\otimes U_1=
\begin{pmatrix}
U_1&0&\cdots&0\\
0&U_1&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&U_1
\end{pmatrix}.
$$

これが元資料26997–27002行、27320–27325行、28778–28781行の表示の意味である。
先の $n_2=2$ の例へ戻り、その二つのblockを全要素で展開する。

$$
I_2\otimes U_1
=\begin{pmatrix}
a&b&0&0\\c&d&0&0\\e&f&0&0\\
0&0&a&b\\0&0&c&d\\0&0&e&f
\end{pmatrix}
\in\mathbb R^{6\times4}.
$$

入力の行は $(i_2,\alpha_1)=(0,0),(0,1),(1,0),(1,1)$、
出力の行は $(i_2,i_1)=(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)$ の順である。
積の第1–3行には $B^{(0)}$ だけ、第4–6行には $B^{(1)}$ だけが寄与する。
各列 $i_3$ の要素まで展開すれば

$$
X_{\mathrm{block}}(:,i_3)=
\begin{pmatrix}
aB_{0,0,i_3}+bB_{1,0,i_3}\\
cB_{0,0,i_3}+dB_{1,0,i_3}\\
eB_{0,0,i_3}+fB_{1,0,i_3}\\
aB_{0,1,i_3}+bB_{1,1,i_3}\\
cB_{0,1,i_3}+dB_{1,1,i_3}\\
eB_{0,1,i_3}+fB_{1,1,i_3}
\end{pmatrix}.
$$

同じ元資料のPyTorch順との比較を、この長方形でも最後まで展開すると

$$
U_1\otimes I_2
=\begin{pmatrix}
a&0&b&0\\0&a&0&b\\
c&0&d&0\\0&c&0&d\\
e&0&f&0\\0&e&0&f
\end{pmatrix}.
$$

こちらの入力は $(\alpha_1,i_2)$ 順、出力は $(i_1,i_2)$ 順である。
次節の定義に従い、torch順からblock順への出力置換を要素で書くと

$$
P_{\mathrm{out}}=
\begin{pmatrix}
1&0&0&0&0&0\\
0&0&1&0&0&0\\
0&0&0&0&1&0\\
0&1&0&0&0&0\\
0&0&0&1&0&0\\
0&0&0&0&0&1
\end{pmatrix}.
$$

入力置換 $P_{\mathrm{in}}$ は次節の $4\times4$ の $P$ と同じである。
$P_{\mathrm{out}}$ は行を1,3,5,2,4,6行目の順へ、
$P_{\mathrm{in}}^T$ は列を1,3,2,4列目の順へ並べる。したがって

$$
\begin{aligned}
P_{\mathrm{out}}(U_1\otimes I_2)
&=\begin{pmatrix}
a&0&b&0\\c&0&d&0\\e&0&f&0\\
0&a&0&b\\0&c&0&d\\0&e&0&f
\end{pmatrix},\\
P_{\mathrm{out}}(U_1\otimes I_2)P_{\mathrm{in}}^T
&=\begin{pmatrix}
a&b&0&0\\c&d&0&0\\e&f&0&0\\
0&0&a&b\\0&0&c&d\\0&0&e&f
\end{pmatrix}
=I_2\otimes U_1.
\end{aligned}
$$

$P_{\mathrm{out}}$ の全要素表示と置換積は、元資料のblock例から導いた補足である。
元資料の $3\times2$ の $U_1$ と $6\times4$ のblock対角行列そのものとは区別する。
長方形では出力置換が $6\times6$、入力置換が $4\times4$ なので、
同じ $P$ を両側へ掛ける式にはできない。
最初から `reshape(n1*n2,n3)`・`reshape(r1*n2,n3)`・$U_1\otimes I_2$
を使えば、この並べ替えを実装する必要はない。

---

## 9. 置換行列 $P$ は何をするか

PyTorch順とblock順には、同じ添字ペアが全て含まれている。ただし並ぶ位置が違う。

```text
PyTorch順:
(i1, i2)

block順:
(i2, i1)
```

この順序だけを入れ替えるものが置換行列 $P$ である。

$P$ は0と1だけからなり、値を混ぜたり情報を削除したりしない。

### $n_1=n_2=2$ の置換行列

PyTorch順のベクトルを

$$
z_{\mathrm{torch}}
=
\begin{pmatrix}
z_{0,0}\\
z_{0,1}\\
z_{1,0}\\
z_{1,1}
\end{pmatrix}
$$

とする。これは

$$
(i_1,i_2)
=
(0,0),(0,1),(1,0),(1,1)
$$

の順である。block順

$$
(i_2,i_1)
=
(0,0),(0,1),(1,0),(1,1)
$$

へ並べ替えると、元の添字ペアとしては

$$
(i_1,i_2)
=
(0,0),(1,0),(0,1),(1,1)
$$

の順になる。この変換を行う0/1行列は

$$
P
=
\begin{pmatrix}
1&0&0&0\\
0&0&1&0\\
0&1&0&0\\
0&0&0&1
\end{pmatrix}
$$

であり、

$$
z_{\mathrm{block}}
=
Pz_{\mathrm{torch}}
=
\begin{pmatrix}
z_{0,0}\\
z_{1,0}\\
z_{0,1}\\
z_{1,1}
\end{pmatrix}
$$

となる。

ここで

$$
U
=
\begin{pmatrix}
1&2\\
3&4
\end{pmatrix}
$$

とする。この節では並び順だけを確認するため、$U$ の直交性は仮定しない。PyTorch順でサイト1へ作用する行列は

$$
U\otimes I_2
=
\begin{pmatrix}
1&0&2&0\\
0&1&0&2\\
3&0&4&0\\
0&3&0&4
\end{pmatrix},
$$

block順で同じ作用を書く行列は

$$
I_2\otimes U
=
\begin{pmatrix}
1&2&0&0\\
3&4&0&0\\
0&0&1&2\\
0&0&3&4
\end{pmatrix}
$$

である。実際に行と列の両方を $P$ で並べ替えると、

$$
\boxed{
I_2\otimes U
=
P
\left(U\otimes I_2\right)
P^T
}
$$

この積でも、中間行列を省略せず確認できる。左から $P$ を掛けると第2・第3行が入れ替わり、

$$
P(U\otimes I_2)
=\begin{pmatrix}
1&0&2&0\\
3&0&4&0\\
0&1&0&2\\
0&3&0&4
\end{pmatrix}.
$$

次に右から $P^T$ を掛けると第2・第3列が入れ替わるので、

$$
\begin{aligned}
P(U\otimes I_2)P^T
&=\begin{pmatrix}
1&0&2&0\\3&0&4&0\\0&1&0&2\\0&3&0&4
\end{pmatrix}
\begin{pmatrix}
1&0&0&0\\0&0&1&0\\0&1&0&0\\0&0&0&1
\end{pmatrix}\\
&=\begin{pmatrix}
1&2&0&0\\3&4&0&0\\0&0&1&2\\0&0&3&4
\end{pmatrix}\\
&=I_2\otimes U.
\end{aligned}
$$

ここでの $U$ は順序を確認する一般行列であり、SVDの列直交性を仮定した数値例ではない。

となる。つまり両者は異なる作用ではなく、同じ添字操作を $(i_1,i_2)$ 順と $(i_2,i_1)$ 順で表した行列である。この例では入出力の次元がどちらも $2\times2$ なので同じ $P$ を使えるが、一般の次元では後述する $P_{\mathrm{in}}$ と $P_{\mathrm{out}}$ を区別する。

置換行列は直交行列なので、

$$
P^TP
=
PP^T
=
I.
$$

したがってrankも変えない。

$$
\operatorname{rank}(PX)
=
\operatorname{rank}(X).
$$

block順で

$$
X_{\mathrm{block}}
=
(I_{n_2}\otimes U)
B_{\mathrm{block}}
$$

と書いた式をPyTorch順へ戻せば、概念的には

$$
X_{\mathrm{torch}}
=
P_{\mathrm{out}}^T
(I_{n_2}\otimes U)
P_{\mathrm{in}}
B_{\mathrm{torch}}
$$

となる。

そしてこの全体をPyTorch順で直接書いたものが、

$$
\boxed{U\otimes I_{n_2}}
$$

である。

したがって、

$$
U\otimes I_{n_2}
=
P_{\mathrm{out}}^T
(I_{n_2}\otimes U)
P_{\mathrm{in}}
$$

一般の長方形 $U$ では、二つの置換は同じshapeとは限らない。0始まりの添字で、torch順からblock順へ写す置換を

$$
\begin{aligned}
(P_{\mathrm{out}})_{j_2n_1+i_1,\ i'_1n_2+j'_2}
&=\delta_{i_1,i'_1}\delta_{j_2,j'_2},\\
(P_{\mathrm{in}})_{j_2r_1+\alpha_1,\ \alpha'_1n_2+j'_2}
&=\delta_{\alpha_1,\alpha'_1}\delta_{j_2,j'_2}
\end{aligned}
$$

と定義する。shapeは各々

$$
P_{\mathrm{out}}\in\mathbb R^{(n_1n_2)\times(n_1n_2)},
\qquad
P_{\mathrm{in}}\in\mathbb R^{(r_1n_2)\times(r_1n_2)}.
$$

成分で順序変換を確かめると、

$$
\begin{aligned}
[P_{\mathrm{out}}(U\otimes I_{n_2})P_{\mathrm{in}}^T]
_{(j_2,i_1),(k_2,\alpha_1)}
&=(U\otimes I_{n_2})_{(i_1,j_2),(\alpha_1,k_2)}\\
&=U_{i_1,\alpha_1}\delta_{j_2,k_2}\\
&=(I_{n_2}\otimes U)_{(j_2,i_1),(k_2,\alpha_1)}.
\end{aligned}
$$

したがって、上の変換式は入出力の両方の並べ替えから従う。$n_1=r_1=n_2=2$ の例では両方が同じ対称行列 $P$ になるが、一般に同じ置換や対称な置換になるとは限らない。

という関係として理解できる。

---

## 10. なぜNotebookコードでは $P$ が不要か

Notebookでは最初から、

```python
X_cut2 = X.reshape(n1 * n2, n3)
B_cut2 = B.reshape(r1 * n2, n3)
L2 = torch.kron(U.contiguous(), I_n2)
```

と、三つ全てをPyTorchの同じ複合index順に揃えている。

したがって、

```text
PyTorch順
→ Pでblock順へ変換
→ I ⊗ U
→ P^TでPyTorch順へ戻す
```

という回り道をコードへ書く必要がない。

$$
\boxed{
\text{reshapeの複合index順と}
\ U\otimes I\text{ の因子順を最初から合わせている}
\Rightarrow
P\text{不要}
}
$$

である。

---

## 11. `reshape`と`torch.kron`が「常に同じ順」なのではない

注意する。

```text
reshapeとtorch.kronは常に自動的に同じ並びになる
```

わけではない。

今回、**reshapeのshapeとKronecker積の因子順を同じ複合index規約になるよう意図して選んだ**ので一致している。

| 書き方 | 対応する複合index順 |
| --- | --- |
| `X.reshape(n1*n2, n3)` | $(i_1,i_2)$ |
| `B.reshape(r1*n2, n3)` | $(\alpha_1,i_2)$ |
| `torch.kron(U, I_n2)` | $(\alpha_1,i_2)\to(i_1,i_2)$ |
| `torch.kron(I_n2, U)` | $(i_2,\alpha_1)\to(i_2,i_1)$ |

順序を混ぜた場合だけ、`permute` や置換行列などで規約を合わせる必要がある。

---

## 12. contiguous/strideは別の問題

`U\otimes I` の添字順の話と、Tensorのmemory layoutは別の問題である。

資料の実行環境では、SVD後の細い `U` と `torch.kron` の組み合わせでstride/layout由来のエラーが出る場合があった。

通常は

```python
L2 = torch.kron(
    U.contiguous(),
    I_n2,
)
```

とする。

それでも環境依存の問題が残る場合に、

```python
U_safe = U.flatten().clone().view(U.shape)
```

として同じ値をcontiguous storageへ実体化した。

これは**複合index順を変える操作ではない**。

- 数値は同じ
- shapeは同じ
- $U^TU$ の直交性は同じ
- $U\otimes I$ の数学的な値は同じ
- 変わるのはstorage/stride

である。

詳細は [[28_TT_MPS実装で使うPyTorch_Python操作メモ]] を参照する。

---

## 13. この章で固定する理解

- TTの第2cutは $(i_1,i_2)\mid i_3$ であり、`reshape(n1*n2,n3)` で作る。
- $i_2\mid(i_1,i_3)$ はmode-2 unfoldingで、`permute` が必要。
- 第1SVD後の $B$ は $(\alpha_1,i_2,i_3)$ を持ち、$i_3$ は列側に残る。
- PyTorch `reshape` では `X_cut2` は $(i_1,i_2)$、`B_cut2` は $(\alpha_1,i_2)$ の順。
- `torch.kron(U,I_n2)` は $(\alpha_1,i_2)\to(i_1,i_2)$ に対応する。
- 三者の複合index順が揃っているので、Notebookコードでは置換行列 $P$ は不要。
- $I_{n_2}\otimes U$ は別のblock順で同じ添字操作を表す説明に使える。
- contiguous/stride問題は添字順とは別の実装上のlayout問題である。

理論的なrank不変性の証明は [[34_基底変換とTT-rank不変性]]、TT-SVDのsrc実装は [[27_TT_MPS基礎のPyTorch実装]] を参照する。
