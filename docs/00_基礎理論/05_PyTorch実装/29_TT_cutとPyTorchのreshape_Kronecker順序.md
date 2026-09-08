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
