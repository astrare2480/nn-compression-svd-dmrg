---
title: TT・MPS基礎理論 セルフRV
aliases:
  - TT MPS self review
  - TT MPS docs self RV
  - TT MPS最終監査
tags:
  - TT
  - MPS
  - review
  - audit
  - docs
---

# TT・MPS基礎理論 セルフRV

## 目的

`TT_MPS基礎理論.md` をもとに作成したTT/MPS基礎docsについて、初回docs化監査とは別に、完成後の状態を**独立したレビュアーのつもりで再点検**する。

確認軸は次の4つである。

1. 元資料から重要な論点が抜けていないか
2. 数式・論理に矛盾や飛躍がないか
3. docs間で同じ記号・概念の意味がずれていないか
4. 現行Notebook / src contractと実装説明が一致しているか

元資料は30,907行の長い対話ログであり、同じ疑問を別の角度から何度も確認している。そのため「全発言を複製する」のではなく、**最終的に確定した理解を落とさず、重複だけ統合する**ことをcoverageの基準とする。

---

## 1. セルフRV対象

### 数学・テンソル代数

```text
docs/00_基礎理論/01_数学基礎/02_テンソル代数/
├── 30_TT_MPSの定義.md
├── 31_TT-rankとunfolding.md
├── 32_TT-SVD.md
├── 33_TT-SVDの打ち切りと誤差.md
└── 34_基底変換とTT-rank不変性.md
```

### PyTorch / Python

```text
docs/00_基礎理論/05_PyTorch実装/
├── 27_TT_MPS基礎のPyTorch実装.md
├── 28_TT_MPS実装で使うPyTorch_Python操作メモ.md
└── 29_TT_cutとPyTorchのreshape_Kronecker順序.md
```

### 手法間・物理・ロードマップ

```text
docs/00_基礎理論/06_手法間のつながり/
├── 15_TuckerとTTの比較.md
├── 16_TT_MPSの物理解釈_縮約密度行列と局所写像.md
├── 17_TT_MPS学習ロードマップ.md
└── 18_TT_MPSの等長写像と射影.md
```

初回監査:

```text
docs/00_基礎理論/00_TT_MPS基礎理論_docs化監査.md
```

このセルフRVにより、学習内容は合計12docsへ整理された。

---

## 2. セルフRVで見つかった補強点

### Finding 1: $U^TU=I$ と $UU^T$ の物理解釈が薄かった

**Severity: coverage gap**

元資料では、PyTorchで

```python
U.T @ U
U @ U.T
```

を比較するだけでなく、数学・物理的に

$$
U^TU=I
$$

を等長写像、

$$
UU^T=P
$$

を列空間への直交射影として詳しく解釈していた。

特に、

$$
\begin{aligned}
\|Ux\|_2^2
&=(Ux)^T(Ux)\\
&=x^TU^TUx\\
&=x^Tx\\
&=\|x\|_2^2
\end{aligned}
$$

というノルム保存の途中式と、

$$
\begin{aligned}
P^2
&=(UU^T)(UU^T)\\
&=U(U^TU)U^T\\
&=UU^T\\
&=P
\end{aligned}
$$

という射影の証明は重要である。

また量子側では

$$
U^\dagger U=I,
\qquad
UU^\dagger=\Pi
$$

という長方形isometryと、正方unitaryを区別していた。

初回docsでは実装上の `U.T @ U` / `U @ U.T` は28へ入っていたが、この物理解釈が独立した形では弱かった。

**対応:**

```text
18_TT_MPSの等長写像と射影.md
```

を追加し、途中式・TT-SVDでの意味・量子/MPS側の随伴表記まで反映した。

**Status: fixed**

---

### Finding 2: TT第2cutとmode-2 unfoldingのコード上の違いが暗黙的だった

**Severity: coverage gap**

理論docsではTT cut unfoldingとTucker mode-n unfoldingを区別していたが、元資料ではさらに具体的に、

```python
X.reshape(n1 * n2, n3)
```

が

$$
(i_1,i_2)\mid i_3
$$

というTT第2cutであり、

$$
i_2\mid(i_1,i_3)
$$

ではないことを確認していた。

後者のmode-2 unfoldingを作るには、axis順を入れ替えて

```python
X.permute(1, 0, 2).reshape(
    n2,
    n1 * n3,
)
```

とする必要がある。

**対応:**

```text
29_TT_cutとPyTorchのreshape_Kronecker順序.md
```

へ、TT cut / mode unfoldingの表と具体的コードを追加した。

**Status: fixed**

---

### Finding 3: `reshape` と `torch.kron` の複合index順、$P$ が不要な理由を独立して残すべきだった

**Severity: coverage gap**

元資料後半では、

```python
X_cut2 = X.reshape(n1 * n2, n3)
B_cut2 = B.reshape(r1 * n2, n3)
L2 = torch.kron(U, torch.eye(n2))
```

について、複合indexの並び順を明示していた。

$$
\operatorname{row}_X
=
i_1n_2+i_2,
$$

$$
\operatorname{row}_B
=
\alpha_1n_2+i_2.
$$

一方、

$$
L_2=U\otimes I_{n_2}
$$

は

$$
(\alpha_1,i_2)
\longrightarrow
(i_1,i_2)
$$

に対応する。

したがって3者は同じPyTorch順で揃っており、コードでは置換行列 $P$ を作る必要がない。

別のblock順

$$
(i_2,i_1)
$$

を使えば

$$
I_{n_2}\otimes U
$$

が自然に現れ、PyTorch順とつなぐために置換行列が必要になる。

**対応:**

29へ、

- PyTorch reshape順
- block順
- $U\otimes I$ と $I\otimes U$
- permutation matrix $P$
- なぜNotebookコードでは $P$ 不要か

を途中式付きで追加した。

**Status: fixed**

---

## 3. 数学・論理セルフRV

### 3.1 TT coreの定義

30の基本式

$$
G^{(k)}
\in
\mathbb R^{r_{k-1}\times n_k\times r_k},
\qquad
r_0=r_d=1
$$

と、

$$
X_{i_1\cdots i_d}
=
\sum_{\alpha_1,\ldots,\alpha_{d-1}}
G^{(1)}_{1,i_1,\alpha_1}
\cdots
G^{(d)}_{\alpha_{d-1},i_d,1}
$$

は整合している。

`reshape(U,1,n_1,r_1)` はdummy boundary axisを追加するだけで、要素数を変えないという説明も元資料と一致する。

**Status: pass**

---

### 3.2 SVDとSchmidt分解

30では係数行列SVDを状態展開へ代入し、

$$
|\Psi\rangle
=
\sum_\alpha
\lambda_\alpha
|L_\alpha\rangle
\otimes
|R_\alpha\rangle
$$

へ到達する途中式を残している。

16ではさらに

$$
\rho_1
=
X^{\langle1\rangle}
\left(X^{\langle1\rangle}\right)^\dagger
=
U\Sigma^2U^\dagger
$$

を導き、

$$
\text{singular value}
\leftrightarrow
\text{Schmidt coefficient},
$$

$$
\sigma_\alpha^2
\leftrightarrow
\rho_1\text{ の固有値}
$$

までつないでいる。

**Status: pass**

---

### 3.3 TT-rank

31の

$$
\boxed{
r_k
=
\operatorname{rank}
\left(X^{\langle k\rangle}\right)
}
$$

と、

$$
X^{\langle k\rangle}
\in
\mathbb R^{
(n_1\cdots n_k)
\times
(n_{k+1}\cdots n_d)
}
$$

はTTのsequential cut定義として統一されている。

Tucker mode-n unfoldingとの違いは29でコード上も明示した。

**Status: pass after Finding 2 fix**

---

### 3.4 TT-SVDの逐次reshape

32では、第 $k$ 段階で

$$
M_k
\in
\mathbb R^{(r_{k-1}n_k)\times(n_{k+1}\cdots n_d)}
$$

を作り、

$$
M_k
=
U_k\Sigma_kV_k^T
$$

から

$$
G^{(k)}
=
\operatorname{reshape}
\left(U_k,r_{k-1},n_k,r_k\right)
$$

を取り出し、

$$
\Sigma_kV_k^T
$$

を次へ渡す流れで統一されている。

これは現行srcの

```python
mat = remainder.reshape(r_left * n_mode, -1)
...
remainder = (torch.diag(S) @ Vh).reshape(...)
```

と一致する。

**Status: pass**

---

### 3.5 `tt_unfold`とTT-SVD内部reshape

27 / 32は、

```text
tt_unfold(X,k)
→ 元Tensor Xのcutを観察

remainder.reshape(...)
→ 現在のTT-SVD remainderを次のSVD行列へ変換
```

を別の役割としている。

第2段階以降、特にtruncation後に両者を混同しない。

これは現行srcの設計と一致する。

**Status: pass**

---

### 3.6 第2cut rank invariance

34の核心

$$
X_{\mathrm{cut2}}
=
L_2B_{\mathrm{cut2}},
\qquad
L_2=U\otimes I_{n_2}
$$

と、

$$
L_2^TL_2
=
I
$$

から、

$$
\operatorname{rank}(X_{\mathrm{cut2}})
=
\operatorname{rank}(B_{\mathrm{cut2}})
$$

を両方向のrank不等式で証明する流れは整合している。

29でPyTorchの複合index順まで補足したため、理論式とコードの間にも飛躍が残らない。

**Status: pass after Finding 3 fix**

---

### 3.7 exact rank / truncation rank

33は

$$
r_k
=
\text{元表現に必要な厳密rank},
$$

$$
\widetilde r_k
=
\text{近似として残すrank}
$$

を明確に分離している。

$$
\widetilde r_k<r_k
$$

では、元では非ゼロである小さい特異値を捨てるため近似誤差が発生する、という説明も一貫している。

**Status: pass**

---

### 3.8 TT-SVDの誤差評価

元資料では途中で誤差直交性の説明を言い直しているため、最終docsでは途中の雑な説明をそのまま採用しない。

33では、

- core同士が直交するとは言わない
- 左interfaceの列直交性を使う
- 実際の局所残差を $\delta_k$
- 許容誤差予算を $\varepsilon_k$

と分けている。

安全に覚える式は

$$
\boxed{
\|X-\widehat X\|_F
\le
\sqrt{
\sum_{k=1}^{d-1}\varepsilon_k^2
}
}
$$

である。

実際に捨てた局所残差を射影構造の中で元空間へ持ち上げる整理では、

$$
\|X-\widehat X\|_F^2
=
\sum_k\delta_k^2
$$

という等号の形と区別している。

また、

$$
\varepsilon_k
=
\frac{\varepsilon}{\sqrt{d-1}}
$$

と誤差予算を配る考え方、TT-SVDは各局所SVDでは最良でも指定TT-rank集合全体での大域的最適化そのものではないという注意も残している。

**Status: pass**

---

### 3.9 truncation後のrank invariance

第1cutをtruncateした後は表現対象が

$$
X
\longrightarrow
\widehat X
$$

へ変わる。

したがって比較するのは

$$
\operatorname{rank}
\left(\widehat X_{\mathrm{cut2}}\right)
$$

と

$$
\operatorname{rank}
\left(\widehat B_{\mathrm{cut2}}\right)
$$

であり、元 $X$ のrankと同じであることは要求しない。

33 / 34 / 27の説明はこの点で一致している。

**Status: pass**

---

## 4. PyTorch / src整合セルフRV

現行srcのpublic APIは、

```python
tt_unfold(X, k)
tt_svd_exact(X)
tt_svd(X, max_rank)
tt_reconstruct(cores)
tt_num_parameters(cores)
```

である。

27の記述と照合した結果、以下は一致している。

### cut contract

```text
1 <= k < X.ndim
```

### exactのrank

```text
torch.linalg.matrix_rank のdefault toleranceによるnumerical rank
```

### zero tensor

```python
effective_rank = max(1, numerical_rank)
```

そのためzero tensorでは

```text
cut numerical rank = 0
TT bond dimension  = 1
```

となる特殊ケースがある。

### dtype contract

正式対応:

```text
torch.float32
torch.float64
```

入口で拒否:

```text
float16
bfloat16
integer
bool
complex
```

### `max_rank`

boolではない1以上の整数を要求する。

### core validation

- non-empty
- 各coreはTensorかつ3階
- 各dimensionは正
- 左端left bond = 1
- 右端right bond = 1
- 隣接bond一致
- dtype一致
- device一致

### reconstruction

```python
result = torch.tensordot(
    result,
    core,
    dims=([-1], [0]),
)
```

で左からbondを縮約し、

```python
.squeeze(0).squeeze(-1)
```

で境界bondだけを外す。

**Status: pass**

---

## 5. 元資料coverage再監査

### 定義・Schmidt・core

主な反映先:

```text
30
16
18
```

反映内容:

- SVDからSchmidt分解への途中式
- TT core
- bond / physical index
- boundary rank
- reduced density matrix
- Schmidt rank
- isometry / projection

**Coverage: complete for current scope**

---

### Tuckerとの比較

反映先:

```text
15
```

反映内容:

- central core vs chain
- multilinear rank vs sequential cut rank
- HOSVD独立SVD vs TT逐次SVD
- parameter scaling
- 物理との接続差

**Coverage: complete**

---

### rank / remainder / $U\otimes I$

反映先:

```text
31
34
29
```

反映内容:

- cut unfolding
- compound index
- $B$に$i_1$がない理由
- $B$に$i_3$が残る理由
- $X_{\mathrm{cut2}}=L_2B_{\mathrm{cut2}}$
- $L_2=U\otimes I$
- Kronecker delta
- rank invariance
- PyTorch index ordering
- permutation matrix

**Coverage: complete after Finding 2/3 fixes**

---

### truncation / error

反映先:

```text
33
34
27
```

反映内容:

- exact / truncated rank
- discarded singular values
- Frobenius local error
- TT global error bound
- interface orthogonality
- local residual vs error budget
- quasi-optimal / non-global-optimum caveat
- truncation後の $\widehat X$ / $\widehat B$

**Coverage: complete**

---

### PyTorch / Python基礎操作

反映先:

```text
27
28
29
```

反映内容:

- `torch.Size`
- `.shape`
- `.numel()`
- `.item()` / Python `int`
- `matrix_rank`
- reduced / truncated SVD
- `reshape(...,-1)`
- `permute`
- list / append / len / enumerate
- slice / list comprehension / dict access
- `tensordot`
- `squeeze(dim)`
- `torch.eye` / `torch.diag`
- `.T`
- stride / contiguous
- `torch.kron`
- `flatten().clone().view` workaround
- PyTorch reshape / Kronecker index order

**Coverage: complete after Finding 2/3 fixes**

---

### 学習ロードマップ

反映先:

```text
17
```

反映内容:

```text
TT基礎
→ gauge freedom
→ QR直交化
→ canonical form
→ TT-rounding
→ TT/MPS演算
→ TT-matrix / MPO
→ Fashion-MNIST
→ ALS / one-site
→ two-site DMRG
```

各理論の後に小実装を挟む方針、Fashion-MNISTを二段階で検証する方針も保存している。

**Coverage: complete as roadmap**

---

## 6. 意図的に本文へ入れていないもの

以下は元資料に存在するが、現在の理論内容そのものではないため、本文の情報として複製していない。

- 「次」「OK」「そうかも」のような会話進行だけの発言
- 学習にかかる日数のその場の見積もり
- 「このチャットを他チャットへ引き継げるか」といった会話運用
- 同じ説明をほぼ同じ内容で聞き直した箇所
- Notebookの生の実行出力だけを見出し化した行
- Perplexity回答の脚注一覧そのものの反復

これは重要内容の欠落ではなく、**会話ログから再利用可能な知識docsへ変換する際の重複・運用情報の除外**である。

一方、同じ質問でも後から新しい説明・注意・例が追加された場合は、その追加情報を統合している。

---

## 7. 現在範囲より後の内容

以下は詳細理論としてはまだdocs化していない。

- gauge freedom
- QR left/right orthogonalization
- left/right/mixed canonical form
- orthogonality center
- TT-rounding
- TT/MPS直接演算
- TT-matrix / MPO
- Fashion-MNISTでのTT層
- ALS / one-site optimization
- two-site DMRG

これは抜けではない。

元資料でも、今回の基礎実装範囲を

```text
TT-SVD
TT-rank
truncation
L2 = U ⊗ I
rank invariance
```

までで区切り、gauge freedom以降は次の学習範囲としている。

17には順番と到達目標だけをロードマップとして残している。

---

## 8. 最終セルフRV判定

### must fix

```text
0件
```

### 数式・理論の重大な矛盾

```text
0件
```

### srcとの重大な不一致

```text
0件
```

### coverage gap

```text
3件検出
→ すべて補完済み
```

補完:

```text
18_TT_MPSの等長写像と射影.md
29_TT_cutとPyTorchのreshape_Kronecker順序.md
```

### 最終判定

現在の学習範囲、すなわち

```text
TT/MPSの定義
TT-rank / cut unfolding
TT-SVD
Schmidt対応
truncation / error
U ⊗ I
basis変換 / rank invariance
PyTorchでの小実装
```

について、元資料からの重要な内容はdocsへ反映されたと判断する。

今後はこの基礎docsを固定し、次の独立トピックである **gauge freedom** へ進む。
