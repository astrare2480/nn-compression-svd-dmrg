---
title: MPOとenvironmentによる1-site局所有効問題
aliases:
  - Hamiltonian environment
  - 1-site有効Hamiltonian
  - MPOによるDMRG局所更新
tags:
  - DMRG
  - MPS
  - MPO
  - environment
  - 有効Hamiltonian
  - matrix-free
---

# MPOとenvironmentによる1-site局所有効問題

## このノートの位置づけ

[[50_DMRG/02_中心テンソルの線形写像と2-site局所更新]]では、小さい全Hamiltonian $H$ と、中心テンソルから全状態への明示的な写像 $P$ を使い、

$$
H_{\mathrm{eff}}=P^{\mathsf T}HP
$$

を直接確認した。

本ノートでは、同じ局所有効問題を、HamiltonianのMPO表現と左右のenvironmentから作る。対応する学習用Notebookは次である。

- [[notebooks/40_dmrg/00_fundamentals/01_mpo_environment_one_site_local_update_exercises.ipynb|01_mpo_environment_one_site_local_update_exercises.ipynb]]

Notebook 01で確認する流れは次である。

```text
full Hamiltonianと同じ演算子を表すMPO
                    ↓
bra・MPO・ketの左／右部分縮約
                    ↓
Hamiltonian environment L, R
                    ↓
1-site有効Hamiltonian H_eff
                    ↓
中心MPS coreだけを1回更新
```

扱う範囲は、指定した1サイトにおける**1回の局所更新**までである。中心を移動する1-site sweep、2-site sweep、SVD打ち切り、discarded weight、収束判定はまだ扱わない。

---

## 1. 数値例と添字規約

Notebook 01では、サイト数 $L=6$ の開境界横磁場Ising鎖を用いる。

$$
H
=
-J\sum_{i=1}^{L-1}Z_iZ_{i+1}
-g\sum_{i=1}^{L}X_i,
\qquad
J=1,
\qquad
g=1.2.
$$

$X,Z$ は固有値が $\pm1$ のPauli行列であり、$S^\alpha=\sigma^\alpha/2$ ではない。

MPS coreとMPO coreの添字順序をそれぞれ

$$
A^{[k]}_{\alpha_{k-1},s_k,\alpha_k},
$$

$$
W^{[k]}_{a_{k-1},s'_k,s_k,a_k}
$$

とする。

- $\alpha_{k-1},\alpha_k$：MPSのvirtual bond。
- $a_{k-1},a_k$：MPOのvirtual bond。
- $s_k$：MPOの入力側、すなわちket側の物理添字。
- $s'_k$：MPOの出力側、すなわちbra側の物理添字。

Notebookは実数・`torch.float64`に限定している。そのため式では転置 $\mathsf T$ を使っているが、複素数の場合はbra側を複素共役し、$\mathsf T$ を $\dagger$ に置き換える。

---

## 2. full HamiltonianとMPOは同じ演算子を異なる形で保持する

full Hamiltonianは、全Hilbert空間に作用する $2^L\times2^L$ の行列として $H$ を保持する。一方、MPOは局所演算子を持つcore列として同じ $H$ を保持する。

$$
H
=
\sum_{a_1,\ldots,a_{L-1}}
W^{[1]}_{a_1}
\otimes
W^{[2]}_{a_1a_2}
\otimes\cdots\otimes
W^{[L]}_{a_{L-1}}.
$$

ここでMPO bond $a_k$ は、「左側でどの演算子の途中状態を保持しているか」を区別する。物理層・MPO層・ket層のどれかを表す層番号ではない。

Notebookでは、MPOとは独立にKronecker積から `H_DENSE` を構築し、MPOをdense化した行列との一致を確認する。`H_DENSE` は検証用であり、environmentによる局所更新の入力には使わない。

---

## 3. 横磁場Ising HamiltonianのMPOを導く

### 3.1 結合をまたぐ演算子の状態

サイト $1$ から $k$ までを見たとき、次の3つの左演算子を考える。

$$
\begin{aligned}
D_k
&=
\sum_{i\le k}(-gX_i)
+
\sum_{i<k}(-JZ_iZ_{i+1}),
\\
P_k
&=
I^{\otimes(k-1)}\otimes(-JZ),
\\
E_k
&=
I^{\otimes k}.
\end{aligned}
$$

- $D_k$：左側だけで完成した項の和。
- $P_k$：次のサイトの $Z$ と結ばれるのを待つ、$ZZ$ 項の左半分。
- $E_k$：まだHamiltonian項を開始していない恒等演算子列。

結合でHamiltonianを切ると、

$$
H
=
H_{\le k}\otimes I
+
(-JZ_k)\otimes Z_{k+1}
+
I\otimes H_{>k}
$$

と分けられる。MPO bond次元3は、この3種類を運ぶためのものである。この例では、結合をまたぐ演算子Schmidt階数も3である。

### 3.2 1サイト追加する漸化式

サイト $k+1$ を追加すると、

$$
\begin{aligned}
D_{k+1}
&=
D_k\otimes I
+P_k\otimes Z
+E_k\otimes(-gX),
\\
P_{k+1}
&=
E_k\otimes(-JZ),
\\
E_{k+1}
&=
E_k\otimes I.
\end{aligned}
$$

$A^{(k)}=(D_k,P_k,E_k)$ とし、

$$
A^{(k+1)}_b
=
\sum_a A^{(k)}_a\otimes W_{ab}
$$

と書けば、bulk coreは

$$
W
=
\begin{pmatrix}
I & 0 & 0\\
Z & 0 & 0\\
-gX & -JZ & I
\end{pmatrix}
$$

となる。ここでは**行が現在のbond状態 $a$、列が次のbond状態 $b$**である。表の各要素はスカラーではなく、1サイトに作用する局所演算子である。

### 3.3 開境界core

サイトを1つも見ていない左端では、

$$
A^{(0)}=(0,0,1)
$$

である。したがって左端coreは

$$
W^{[1]}
=
\begin{pmatrix}
-gX & -JZ & I
\end{pmatrix}.
$$

右端では、最終的に完成済みの $D_L=H$ だけを取り出すので、右境界ベクトルは $(1,0,0)^{\mathsf T}$ であり、

$$
W^{[L]}
=
\begin{pmatrix}
I\\
Z\\
-gX
\end{pmatrix}
$$

となる。

端のshapeはそれぞれ $(1,d,d,3)$ と $(3,d,d,1)$、bulkは $(3,d,d,3)$ である。左端・bulk・右端を縮約すると、元の横磁場Ising Hamiltonianが得られる。

---

## 4. Hamiltonian environmentとは何か

### 4.1 期待値の部分縮約

Hamiltonian environmentは新しい物理量ではない。期待値

$$
\langle\psi|H|\psi\rangle
$$

を表すbra MPS・MPO・ket MPSの3層networkを、端から途中まで縮約した中間量である。

```text
bra MPS :  A* ─ A* ─ …
              │    │
MPO     :   W ── W ── …
              │    │
ket MPS :   A ── A ─ …
```

inner product用environmentとの違いは、Hamiltonian environmentには中央のMPO層 $W$ と、そのvirtual bondが加わることである。

### 4.2 left environment

サイト $k-1$ までを取り込んだleft environmentを

$$
L^{[k-1]}_{abc}
$$

とする。$a,b,c$ は、それぞれbra MPS、MPO、ket MPSの右向きbondである。

サイト $k$ を取り込む更新は、

$$
L^{[k]}_{ijk}
=
\sum_{a,b,c,d,e}
L^{[k-1]}_{abc}
\overline{A^{[k]}_{adi}}
W^{[k]}_{bdej}
A^{[k]}_{cek}.
$$

縮約されるのは、古い3本のbond $a,b,c$ と、bra・ketの物理添字 $d,e$ である。新しい3本のbond $i,j,k$ は、次のサイトとの接続として残る。

まだ何も取り込んでいない左境界は、shape $(1,1,1)$ の値1で初期化する。0ではなく1なのは、空積の単位元だからである。

### 4.3 right environment

right environmentは同じ3層networkを右端から左へ縮約したものである。Notebookの添字規約では、

$$
R^{[k]}_{abc}
=
\sum_{i,j,k,d,e}
\overline{A^{[k]}_{adi}}
W^{[k]}_{bdej}
A^{[k]}_{cek}
R^{[k+1]}_{ijk}.
$$

右境界もshape $(1,1,1)$ の値1から始める。

すべてのサイトを取り込むと、left側から縮約してもright側から縮約してもスカラーになり、full stateから計算した $\langle\psi|H|\psi\rangle$ と一致する。

### 4.4 中心に使うenvironmentの位置

0始まりの中心siteを `c` とすると、中心より左だけを含む

```python
LEFT_ENVIRONMENTS[c]
```

と、中心より右だけを含む

```python
RIGHT_ENVIRONMENTS[c + 1]
```

を使う。中心site自身はどちらにも入れない。中心のMPO coreは `MPO_CORES[c]` として別に挟む。

---

## 5. 左右environmentから1-site有効Hamiltonianを作る

中心より左を $L$、中心より右を $R$ へ縮約済みとする。添字を

$$
L_{abc},
\qquad
W_{bdej},
\qquad
R_{ijk}
$$

と書くと、有効Hamiltonianは

$$
\boxed{
(H_{\mathrm{eff}})_{adi,cek}
=
\sum_{b,j}
L_{abc}W_{bdej}R_{ijk}
}
$$

である。

ここで縮約して消えるのは、MPOを左右につなぐbond $b,j$ である。残る添字は、

- $(a,d,i)$：中心coreのbra側の添字で、$H_{\mathrm{eff}}$ の行になる。
- $(c,e,k)$：中心coreのket側の添字で、$H_{\mathrm{eff}}$ の列になる。

である。

「MPOしか縮約しない」という表現は正確ではない。中心より外側のbra・MPO・ketはすでに $L,R$ の中へ縮約されている。$H_{\mathrm{eff}}$ を組み立てる段階では、その $L,W,R$ 間のMPO bondだけを閉じ、中心のbra・ketを接続する添字は変数を入れる場所として残している。

### なぜ中心core $A$ はまだ入れないのか

$H_{\mathrm{eff}}$ は、任意の中心候補 $A$ に作用できる局所演算子として先に作るからである。中心ketを $A_{cek}$、braを $A^*_{adi}$ とすれば、期待値の分子は

$$
\langle\psi|H|\psi\rangle
=
\sum_{a,d,i,c,e,k}
A^*_{adi}
(H_{\mathrm{eff}})_{adi,cek}
A_{cek}.
$$

すなわち、中心添字をflattenしたベクトル $c=\operatorname{vec}(A)$ を使えば、

$$
\langle\psi|H|\psi\rangle
=
c^\dagger H_{\mathrm{eff}}c
$$

となる。$A$ は $H_{\mathrm{eff}}$ を作る段階ではなく、演算子を作用させる段階、または二次形式を作る段階で現れる。

---

## 6. 「周囲を固定する」の正確な意味

中心を $n$ 番目のMPS core $A_n$ とする。1-site局所最適化では、

$$
A_n\text{ だけを変え},
\qquad
A_m\ (m\neq n)\text{ と全MPO core }W_m\text{ は変えない}
$$

と**最適化変数を指定する**。

左右environmentは、

$$
L
=
L(A_1,\ldots,A_{n-1};W_1,\ldots,W_{n-1}),
$$

$$
R
=
R(A_{n+1},\ldots,A_L;W_{n+1},\ldots,W_L)
$$

から作られ、現在変えている $A_n$ を含まない。したがって局所最適化中は、

$$
\delta A_n\neq0,
\qquad
\delta A_m=0\ (m\neq n),
\qquad
\delta W_m=0
$$

より、

$$
\delta L=\delta R=\delta H_{\mathrm{eff}}=0
$$

である。

重要なのは、**縮約したから固定になるのではない**という点である。中心以外を変えないという局所最適化の規則を先に置くため、その固定テンソルから構成される $L,R,H_{\mathrm{eff}}$ も固定される。

Hamiltonianを表すMPO自体はsweep中も固定される。一方、中心を移した後は固定されるMPS coreの集合が変わるため、$L,R$ と $H_{\mathrm{eff}}$ は更新される。

「局所最適化」とは、Hamiltonianの局所項だけを評価するという意味ではない。変える変数を1個のcoreに限定するという意味であり、中心と全周囲の相互作用は $L,R$ を通じて評価される。

---

## 7. mixed-canonical formと通常の固有値問題

中心から全状態への線形写像を $P$ と書くと、

$$
|\psi(A)\rangle=P|c\rangle,
\qquad
c=\operatorname{vec}(A)
$$

である。

中心より左を左正準形、右を右正準形にしたmixed-canonical MPSでは、

$$
P^\dagger P=I
$$

となる。したがって全状態のRayleigh商は、

$$
E(c)
=
\frac{c^\dagger P^\dagger HPc}{c^\dagger P^\dagger Pc}
=
\frac{c^\dagger H_{\mathrm{eff}}c}{c^\dagger c}
$$

と書ける。

正規化条件 $c^\dagger c=1$ に未定乗数 $E$ を導入すると、

$$
\mathcal L(c,c^*)
=
c^\dagger H_{\mathrm{eff}}c
-E(c^\dagger c-1)
$$

であり、$c^*$ に関する停留条件から

$$
H_{\mathrm{eff}}c=Ec
$$

を得る。最小固有値に対応する固有ベクトルを、新しい中心coreへreshapeする。

非正準形では $P^\dagger P=N_{\mathrm{eff}}\neq I$ となるため、

$$
H_{\mathrm{eff}}c
=
E N_{\mathrm{eff}}c
$$

という一般化固有値問題になる。Notebook 01はmixed-canonical formに限定し、この非正準形との数値比較は行わない。

---

## 8. environment式と明示的な射影式の一致

Notebook 01では、中心空間の基底を1本ずつ中心coreへ入れ、得られたfull stateを列に並べて、検証用の $P$ を明示的に作る。そして、

$$
P^{\mathsf T}P=I
$$

と、

$$
H_{\mathrm{eff}}^{(\mathrm{env})}
=
P^{\mathsf T}H_{\mathrm{dense}}P
$$

を確認する。

この一致が検証するのは、次の2つが同じ局所演算子を表していることである。

1. 全状態空間へ埋め込み、full Hamiltonianを作用させ、中心空間へ戻す方法。
2. 中心以外を左右environmentへ縮約し、中心MPO coreを挟む方法。

実際のDMRGでは巨大な $P$ やfull Hamiltonianを作らない。ここでの明示式は、小さい数値例でMPO規約、environmentの位置、添字順序、reshape順序が正しいことを確認するための参照実装である。

---

## 9. $H_{\mathrm{eff}}$を作らないmatrix-free作用

入力Tensor $X_{cek}$ を中心候補とし、

$$
Y_{adi}
=
\sum_{b,j,c,e,k}
L_{abc}
W_{bdej}
R_{ijk}
X_{cek}
$$

を直接縮約する。flattenしたベクトルを $x=\operatorname{vec}(X)$、$y=\operatorname{vec}(Y)$ とすれば、

$$
y=H_{\mathrm{eff}}x
$$

である。

ここでのmatrix-freeとは、密行列 $H_{\mathrm{eff}}$ も、その前段階の6添字effective tensorも明示的に作らず、入力に対する作用だけを返すことを意味する。

Notebookでは、次の一致を確認する。

$$
\operatorname{vec}(Y)
=
H_{\mathrm{eff}}\operatorname{vec}(X),
$$

$$
x^\mathsf{T}y
=
x^\mathsf{T}H_{\mathrm{eff}}x,
$$

$$
\langle\psi(X)|H|\psi(X)\rangle
=
x^\mathsf{T}H_{\mathrm{eff}}x,
$$

$$
\langle\psi(X)|\psi(X)\rangle
=
x^\mathsf{T}x.
$$

$X$ は任意の局所入力であり、必ずしも規格化されていない。そのため $x^\mathsf{T}H_{\mathrm{eff}}x$ は期待値の**分子**であり、単独ではエネルギーと呼ばない。Rayleigh商にするには $x^\mathsf{T}x$ で割る必要がある。

matrix-free作用を関数として渡せれば、将来は密な $H_{\mathrm{eff}}$ を作らず、Lanczos法などの反復固有値solverへ接続できる。Notebook 01では作用の一致までを扱い、反復solverは実装しない。

---

## 10. 1回の1-site局所更新で最適化される範囲

固定した $L,W,R$ に対して、

$$
H_{\mathrm{eff}}c_{\mathrm{opt}}
=
E_{\mathrm{opt}}c_{\mathrm{opt}}
$$

の最小固有対を求め、$c_{\mathrm{opt}}$ を中心coreのshapeへ戻す。左右のMPS coreは変更しない。

この更新が厳密に最小化するのは、

- 現在の左右MPS coreを固定し、
- 現在のbond dimensionを固定し、
- 中心1サイトのcoreだけを自由にした

部分空間におけるエネルギーである。

したがって、更新後エネルギーが局所有効問題の最小固有値と一致しても、全Hilbert空間における厳密基底エネルギーと一致する必要はない。1-site更新ではbond dimensionも増やせない。

---

## 11. PyTorch実装と添字の対応

### 11.1 MPO coreのshape

局所演算子 `IDENTITY`、`PAULI_X`、`PAULI_Z` のshapeはすべて $(d,d)$ である。

```python
# 元の局所演算子とdtype・device・shapeが同じ零演算子を作る。
zero = torch.zeros_like(IDENTITY)
```

左境界では、3個の局所演算子を `dim=-1` で積む。`dim=-1` は既存の出力・入力添字の後ろに、右MPO bondを追加する指定である。

```python
# (d, d, 3)を作り、左MPO bondの次元1を先頭へ追加する。
left_boundary = torch.stack(
    [x_term, z_start, IDENTITY],
    dim=-1,
).unsqueeze(0)
# shape: (1, d, d, 3)
```

bulkでは、内側の `stack(..., dim=-1)` がMPOの各行をshape $(d,d,3)$ で作り、外側の `stack(..., dim=0)` が3行を左MPO bondとして先頭へ積む。

```python
bulk = torch.stack(
    [
        torch.stack([IDENTITY, zero, zero], dim=-1),
        torch.stack([PAULI_Z, zero, zero], dim=-1),
        torch.stack([x_term, z_start, IDENTITY], dim=-1),
    ],
    dim=0,
)
# shape: (3, d, d, 3)
```

右境界では、3個の局所演算子を左MPO bondとして先頭へ積み、右MPO bondの次元1を末尾へ追加する。

```python
right_boundary = torch.stack(
    [IDENTITY, PAULI_Z, x_term],
    dim=0,
).unsqueeze(-1)
# shape: (3, d, d, 1)
```

### 11.2 environment更新の`einsum`

left environmentの式

$$
L^{[k]}_{ijk}
=
\sum_{a,b,c,d,e}
L^{[k-1]}_{abc}
\overline{A^{[k]}_{adi}}
W^{[k]}_{bdej}
A^{[k]}_{cek}
$$

は、次に直接対応する。

```python
updated_left_environment = torch.einsum(
    "abc,adi,bdej,cek->ijk",
    left_environment,
    mps_core.conj(),
    mpo_core,
    mps_core,
)
```

right environmentは、同じsiteの3テンソルを右側の既存environmentへ取り込み、左向きの3本のbondを残す。

```python
updated_right_environment = torch.einsum(
    "ijk,adi,bdej,cek->abc",
    right_environment,
    mps_core.conj(),
    mpo_core,
    mps_core,
)
```

### 11.3 有効Hamiltonianとmatrix-free作用の`einsum`

$H_{\mathrm{eff}}$ を6添字Tensorとして作る縮約は、

```python
effective_tensor = torch.einsum(
    "abc,bdej,ijk->adicek",
    left_environment,
    center_mpo_core,
    right_environment,
)
```

である。出力の `adi` がbra側、`cek` がket側であり、この順序のまま

```python
effective_hamiltonian = effective_tensor.reshape(
    center_core.numel(),
    center_core.numel(),
)
```

とすれば、行と列のflatten順序が中心coreの `reshape(-1)` と一致する。

matrix-free作用では、入力core `cek` も同時に縮約し、出力側 `adi` だけを残す。

```python
output_core = torch.einsum(
    "abc,bdej,ijk,cek->adi",
    left_environment,
    center_mpo_core,
    right_environment,
    input_core,
)
```

添字文字そのものに物理的意味があるのではなく、同じ文字を持つ軸を縮約し、出力指定に残した文字の順序で結果の軸を並べることが重要である。

---

## 12. Notebook 01の保存済み検証結果

以下はNotebookに保存された出力の確認であり、このノート作成時にFresh Run Allを行った結果ではない。

### 12.1 検証ステージ

保存済み出力では、次がすべて `OK` になっている。

- Kronecker積で独立に作ったdense Hamiltonian。
- MPOをdense化したHamiltonianとの一致。
- center site 2、内部bond dimension 2のmixed-canonical MPS。
- left environmentの更新と全期待値。
- right environmentの更新と全期待値。
- environmentから作った $H_{\mathrm{eff}}$。
- 明示的な $P^{\mathsf T}HP$ との一致。
- matrix-free作用と明示行列作用の一致。
- 1回の1-site局所更新。

### 12.2 保存値

更新前エネルギーは、

$$
E_{\mathrm{before}}
=
1.157791565571693
$$

である。局所有効問題の最小値と、中心coreだけを置換した後のfull stateのエネルギーは、

$$
E_{\mathrm{local\ min}}
=
E_{\mathrm{after}}
=
-2.805281614274409
$$

で一致した。

全Hilbert空間で厳密対角化した基底エネルギーは、

$$
E_{\mathrm{exact}}
=
-8.269344348226955
$$

であり、更新後状態のglobal residualは、

$$
\left\|H|\psi_{\mathrm{after}}\rangle
-E_{\mathrm{after}}|\psi_{\mathrm{after}}\rangle\right\|
=
3.2695724304911513
$$

である。

この結果から確認できる事実は、固定した左右環境が定める局所空間では中心coreを最小固有ベクトルへ更新できたことである。更新後状態が全Hamiltonianの基底状態または固有状態になった、とはいえない。

---

## 13. 当初案から実際のNotebookへ変更された点

Notebook 01の実物は、当初の会話で示された案をそのまま実装したものではない。実際には次の構成を採用している。

- NumPyではなくPyTorchと既存の `nn_compression` APIを使う。
- Heisenberg鎖ではなく横磁場Ising鎖を使う。
- 複素数ではなく実数・`float64`を使う。
- 非一様bond dimensionではなく、内部bond dimensionを2に固定する。
- 正準化をNotebook内で再実装せず、既存APIを再利用する。
- 中心を全位置で動かさず、固定した1サイトだけを更新する。
- matrix-freeな $H_{\mathrm{eff}}$ 作用の一致確認を追加する。
- 非正準形との数値比較は行わない。

この変更は、Notebook 01の学習対象を「MPOとenvironmentによる局所有効問題の構成」に絞るためのものである。

---

## 14. このNotebookで確認していないこと

Notebook 01を完了しても、次は未検証である。

- center siteを移動する1-site sweep。
- 複数bondを往復する2-site sweep。
- sweep中のleft／right environmentの保存・再利用・更新順序。
- 2-site tensorのSVD再分割とbond dimensionの増減。
- SVD打ち切り、discarded weight、切り捨て後のエネルギー変化。
- sweepの収束判定。
- 大規模問題での反復固有値solver。
- NN損失へ置き換えたDMRG-like学習。

物理DMRGからNNへそのまま移せるのは、environment、正準形、中心移動、SVD打ち切りといったTensor Networkの部品である。一方、物理HamiltonianのMPO、最小固有値問題、エネルギーの単調減少を、そのままNN学習へ持ち込むことはできない。NN側では目的関数と局所更新問題を別に定義する必要がある。

次のNotebookでは、同じHamiltonianを使い、2-site局所更新を複数bondへつなぐleft-to-right／right-to-left sweep、environmentの逐次更新、SVD打ち切り、discarded weight、収束判定を扱う。
