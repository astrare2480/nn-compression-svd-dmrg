---
title: TT-SVDの準最適性と誤差上界のPyTorch確認
tags:
  - TT
  - MPS
  - PyTorch
  - TT-SVD
  - quasi-optimality
---

# TT-SVDの準最適性と誤差上界のPyTorch確認

## このノートの位置づけ

理論導出は [[00_基礎理論/01_数学基礎/02_テンソル代数/47_TT-SVDの準最適性と誤差上界]]、手を動かす学習セルは [Notebook 11](../../notebooks/30_tt_mps/00_fundamentals/11_tt_svd_quasi_optimality_and_error_bound.ipynb) を参照する。ここでは、Notebookで確認した式、shape、代表的な数値結果、PyTorch実装上の注意をまとめる。

対象は、dense tensorから左から右へ構成する標準TT-SVDである。既存TTコアのrankを下げるTT-roundingとは区別する。

## 1. 検証条件

- dtype：`torch.float64`
- random seed：`11`
- warm-up行列：$M\in\mathbb R^{6\times4}$
- 指定特異値：$(5,2,0.5,0.1)$
- warm-upの保持rank：$r=2$
- dense tensor：$A\in\mathbb R^{4\times4\times4\times4}$
- 固定rank：$(r_1,r_2,r_3)=(2,4,2)$

## 2. 単一行列のtruncated SVD

保持rankが $r=2$ なので、捨てる特異値は $0.5$ と $0.1$ である。理論値は

$$
\|M-M_2\|_F^2
=
0.5^2+0.1^2
=
0.26.
$$

実行結果は次の通りだった。

```text
||M - M_r||_F^2 = 0.2599999999999998
sum_{l>r} sigma^2 = 0.2599999999999999
difference = 1.1102230246251565e-16
```

丸め誤差水準で

$$
\|M-M_r\|_F^2
=
\sum_{\ell>r}\sigma_\ell^2
$$

を確認できた。実装の要点は次の通りである。

```python
# reduced SVDを計算する。
U, S, Vh = torch.linalg.svd(M, full_matrices=False)

# 上位r成分だけでrank-r行列を再構成する。
r = 2
max_rank = S.numel()
assert 1 <= r <= max_rank
M_r = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]

# 再構成誤差の二乗と捨てた特異値の二乗和を比較する。
reconstruction_error_sq = torch.linalg.vector_norm(M - M_r).square()
discarded_sq = S[r:].square().sum()
```

`max_rank` はこの行列で保持できる最大rank、`r` は実際に保持するrankを表す。両者を単に `rank` と書き分けないより役割が明確になる。

`reconstruction_error_sq` は $\|M-M_r\|_F^2$ を保持する。`reconstruction_error` という名前にするなら、通常は二乗前の $\|M-M_r\|_F$ を入れる方が明確である。変数名は `reconstruction_error` と正しく綴り、値が二乗済みなら `_sq` を付けて区別する。

## 3. rank制限なしTT-SVD

$A\in\mathbb R^{4\times4\times4\times4}$ に対して、各SVDの全rankを保持し、TTコアから元のテンソルを再構成した。

```text
reconstruction_error: 3.4675948148250965e-14
```

これはfloat64の丸め誤差水準であり、rank制限なしTT-SVDが理論通り情報を失わずに再構成できている。

### 3.1 各SVDのshape、特異値、bond rank、core shape

この実験ではセットアップでwarm-up行列を生成した後に $A$ を生成している。seedだけでなく乱数を消費する順序も同じにすると、rank制限なしTT-SVDの各段階は次の値になる。

第1段階では、

$$
M_1
\in
\mathbb R^{4\times64},
$$

$$
\sigma(M_1)
=
(10.091549112942,\ 8.542518161548,\ 8.008357115962,\ 6.920341921924)
$$

であり、数値rankは $r_1=4$、第1コアは

$$
G^{(1)}
\in
\mathbb R^{1\times4\times4}
$$

となる。

第2段階では左bondと第2物理添字を行側へまとめるため、

$$
M_2
\in
\mathbb R^{(4\cdot4)\times(4\cdot4)}
=
\mathbb R^{16\times16}
$$

となる。全特異値は

$$
\begin{aligned}
\sigma(M_2)
=(&7.771993952184,\ 7.077945056663,\ 5.621349169028,\ 5.406311550472,\\
&5.251194375883,\ 4.896236469881,\ 4.283442133325,\ 3.707215350314,\\
&3.482791626905,\ 2.627596329469,\ 2.447759835567,\ 1.999806557158,\\
&1.136485279277,\ 0.887641203452,\ 0.740131432508,\ 0.467202278407)
\end{aligned}
$$

であり、数値rankは $r_2=16$、第2コアは

$$
G^{(2)}
\in
\mathbb R^{4\times4\times16}
$$

となる。

第3段階では、

$$
M_3
\in
\mathbb R^{(16\cdot4)\times4}
=
\mathbb R^{64\times4},
$$

$$
\sigma(M_3)
=
(10.161949916282,\ 9.027492343254,\ 7.524552596524,\ 6.742340852110)
$$

であり、数値rankは $r_3=4$ となる。したがって、残りのコアshapeは

$$
G^{(3)}
\in
\mathbb R^{16\times4\times4},
\qquad
G^{(4)}
\in
\mathbb R^{4\times4\times1}
$$

である。よって、rank制限なしのbond rank列とcore shape列は

$$
(r_1,r_2,r_3)
=
(4,16,4),
$$

$$
(1,4,4),
\quad
(4,4,16),
\quad
(16,4,4),
\quad
(4,4,1)
$$

となる。ここで `tt_svd_exact` が使うrankは、`torch.linalg.matrix_rank` の既定toleranceに基づく数値rankである。

## 4. 固定rank TT-SVDと基本誤差上界

固定rank $(r_1,r_2,r_3)=(2,4,2)$ でTT-SVDを実行した。ここではbondごとに異なる上限を与える `target_tt_ranks = [2, 4, 2]` を使う。全bondへ同じ上限を与えるscalarの `max_rank` とは役割が異なり、`max_rank` だけではこのrank列を表せない。

元のテンソル $A$ のunfolding shapeは

$$
A^{\langle1\rangle}\in\mathbb R^{4\times64},
\qquad
A^{\langle2\rangle}\in\mathbb R^{16\times16},
\qquad
A^{\langle3\rangle}\in\mathbb R^{64\times4}
$$

である。各unfoldingを元の $A$ から直接作り、

$$
\varepsilon_1^2
=
\sum_{\ell>2}\sigma_\ell(A^{\langle1\rangle})^2,
$$

$$
\varepsilon_2^2
=
\sum_{\ell>4}\sigma_\ell(A^{\langle2\rangle})^2,
$$

$$
\varepsilon_3^2
=
\sum_{\ell>2}\sigma_\ell(A^{\langle3\rangle})^2
$$

を計算した。

### 4.1 TT-SVD途中の行列と実際の局所打ち切り

固定rank TT-SVDが実際にSVDする中間行列は、元のunfoldingを3本独立にSVDしたものではない。第1段階の打ち切り結果を次段へ渡すため、shapeと特異値は次のように変化する。

第1段階では、

$$
M_1
\in
\mathbb R^{4\times64},
\qquad
\sigma(M_1)
=
(10.091549112942,\ 8.542518161548,\ 8.008357115962,\ 6.920341921924)
$$

から先頭2本を保持する。したがって、

$$
G^{(1)}
\in
\mathbb R^{1\times4\times2},
\qquad
\delta_1^2
=
8.008357115962^2+6.920341921924^2
=
112.024916013132
$$

となる。

第2段階の中間行列は、直前のbond rankが2なので、

$$
M_2
\in
\mathbb R^{(2\cdot4)\times(4\cdot4)}
=
\mathbb R^{8\times16}
$$

である。全特異値は

$$
\begin{aligned}
\sigma(M_2)
=(&7.464818969505,\ 6.324239088674,\ 5.190569498373,\ 4.107774957031,\\
&3.928986508236,\ 3.102343506028,\ 2.613136925994,\ 1.840835712594)
\end{aligned}
$$

であり、先頭4本を保持するため、

$$
G^{(2)}
\in
\mathbb R^{2\times4\times4},
$$

$$
\begin{aligned}
\delta_2^2
={}&
3.928986508236^2
+3.102343506028^2\\
&+2.613136925994^2
+1.840835712594^2\\
=&
35.278630926052
\end{aligned}
$$

となる。

第3段階では、

$$
M_3
\in
\mathbb R^{(4\cdot4)\times4}
=
\mathbb R^{16\times4},
$$

$$
\sigma(M_3)
=
(7.977606047821,\ 6.123703963833,\ 4.577031584883,\ 4.176623336301)
$$

から先頭2本を保持する。したがって、

$$
G^{(3)}
\in
\mathbb R^{4\times4\times2},
\qquad
G^{(4)}
\in
\mathbb R^{2\times4\times1},
$$

$$
\delta_3^2
=
4.577031584883^2+4.176623336301^2
=
38.393400622356
$$

となる。この実験では、TT-SVDで実際に捨てた局所誤差について、

$$
\begin{aligned}
\|A-A_{\mathrm{tt}}\|_F
&=
\left(
\delta_1^2+\delta_2^2+\delta_3^2
\right)^{1/2}\\
&=
\sqrt{
112.024916013132
+35.278630926052
+38.393400622356
}\\
&=
13.627066726245229
\end{aligned}
$$

が数値的に成り立つ。一方、次に計算する $\varepsilon_k$ は元の $A$ の各cut unfoldingから求めるため、$\delta_k$ と定義対象を区別する。

```text
reconstruction_actual_error: 13.627066726245229
epsilon_1^2 = 112.02491601313216
epsilon_2^2 = 115.50992877359565
epsilon_3^2 = 102.07805194389165
bound = 18.15524433133907
actual_error <= bound + eps ? True
```

すなわち、

$$
\|A-A_{\mathrm{tt}}\|_F
=
13.627066726245229
\le
18.15524433133907
=
\left(
\varepsilon_1^2+
\varepsilon_2^2+
\varepsilon_3^2
\right)^{1/2}.
$$

実測誤差と上界は等しい必要がない。この実験で直接確認したのは基本誤差上界である。大域最適TT近似 $A_\ast$ は計算していないため、

$$
\|A-A_{\mathrm{tt}}\|_F
\le
\sqrt{d-1}\,\|A-A_\ast\|_F
$$

そのものは理論ノートで導出する。

## 5. 等長埋め込みと縮約の確認

縦長行列をreduced QR分解し、

$$
U_{\mathrm{iso}}
\in
\mathbb R^{20\times5},
\qquad
U_{\mathrm{iso}}^TU_{\mathrm{iso}}=I_5
$$

を満たす列直交行列を作る。確認する式は

$$
\|U_{\mathrm{iso}}Z\|_F=\|Z\|_F,
\qquad
\|U_{\mathrm{iso}}^TY\|_F\le\|Y\|_F
$$

である。

```python
# reduced QRによって列直交行列を作る。
A_tall = torch.randn(20, 5, dtype=torch.float64)
U_iso, _ = torch.linalg.qr(A_tall, mode="reduced")

# このノートではU_isoを2次元行列として扱う。
assert U_iso.ndim == 2
assert U_iso.shape == (20, 5)

# U^T U = Iを確認する。
gram = U_iso.mT @ U_iso
I = torch.eye(U_iso.shape[1], dtype=U_iso.dtype, device=U_iso.device)
gram_error = torch.linalg.vector_norm(gram - I)

# Uによる埋め込みがFrobeniusノルムを保存するか確認する。
Z = torch.randn(U_iso.shape[1], 3, dtype=U_iso.dtype)
embedding_gap = (
    torch.linalg.vector_norm(U_iso @ Z)
    - torch.linalg.vector_norm(Z)
).abs()

# U^Tによる縮約がノルムを増やさないか確認する。
Y = torch.randn(U_iso.shape[0], 3, dtype=U_iso.dtype)
contracted_norm = torch.linalg.vector_norm(U_iso.mT @ Y)
original_norm = torch.linalg.vector_norm(Y)

print("||U^T U - I||_F =", gram_error.item())
print("||U Z||_F - ||Z||_F =", embedding_gap.item())
print("||U^T Y||_F =", contracted_norm.item())
print("||Y||_F =", original_norm.item())
print(
    "contraction inequality holds =",
    bool(contracted_norm <= original_norm + 1e-12),
)
```

PyTorchで行列転置を明示するときは、2次元行列にも多次元Tensorにも意味が明確な `.mT` を使う。`.T` は2次元行列では同じ結果になるが、多次元Tensorでは全次元を逆順に並べる動作になるため、行列転置の意図を表す用途では `.mT` の方が安全である。ここでは `U_iso.ndim == 2` も明示的に確認している。

## 6. 確認できたこと

1. 単一行列では、再構成誤差の二乗と捨てた特異値の二乗和が一致する。
2. rank制限なしTT-SVDは、float64の丸め誤差水準で元テンソルを再構成する。
3. 固定rank TT-SVDの実測誤差は、元の各unfoldingから求めた基本誤差上界以下だった。
4. 列直交行列による埋め込みはノルムを保存し、転置による縮約はノルムを増やさない。
5. 基本誤差上界の数値検証と、大域最適TT近似を必要とする準最適係数の直接検証は別である。
