---
title: TT・MPS実装で使うPyTorch・Python操作メモ
aliases:
  - TT PyTorch操作メモ
  - TT Pythonメモ
  - TT Notebook補助メモ
tags:
  - TT
  - MPS
  - PyTorch
  - Python
  - implementation
---

# TT・MPS実装で使うPyTorch・Python操作メモ

## サマリー

TT/MPS基礎Notebookでは、理論そのものとは別に、PyTorch/Pythonの細かい配列操作を何度も確認した。この章はそれらを独立した実装メモとしてまとめる。

対象は主に、

- `torch.Size` / `.shape`
- `.numel()` / `.item()`
- `reshape`
- `list` / `append` / `len`
- slice / list comprehension / dictionary access
- `enumerate`
- `torch.tensordot`
- `squeeze(dim)`
- `torch.eye` / `torch.diag`
- `.T` / stride / `.contiguous()`

である。

TT-SVD本体のsrc contractは [[27_TT_MPS基礎のPyTorch実装]] を参照する。

---

## 1. `torch.Size` とshapeの保存

Tensorのshapeは

```python
X.shape
```

で取得する。

```python
X = torch.randn(2, 3, 4)
print(X.shape)
# torch.Size([2, 3, 4])
```

`torch.Size` はtupleのように扱えるので、

```python
n1, n2, n3 = X.shape
```

と分解できる。

普通のtupleとして保存したいなら

```python
shape = tuple(X.shape)
# (2, 3, 4)
```

listにしたいなら

```python
shape = list(X.shape)
# [2, 3, 4]
```

である。

---

## 2. `.numel()`

```python
X.numel()
```

はTensorの全要素数を返す。

$$
X\in\mathbb R^{n_1\times\cdots\times n_d}
$$

なら

$$
\operatorname{numel}(X)
=
n_1n_2\cdots n_d.
$$

TTのparameter ratioを計算するとき、dense側の要素数として使える。

```python
dense_params = X.numel()
```

---

## 3. `.item()` とPythonの整数

`torch.linalg.matrix_rank` の返り値は0階Tensorである。

```python
rank_tensor = torch.linalg.matrix_rank(X1)
print(rank_tensor)
# tensor(2)
```

Tensor内部の1個の値をPython scalarへ取り出すのが

```python
rank_tensor.item()
```

である。

```python
r1 = torch.linalg.matrix_rank(X1).item()
print(r1)
# 2
```

rankを明示的にPython `int` として扱うなら

```python
r1 = int(torch.linalg.matrix_rank(X1).item())
```

とする。

```text
tensor(2)
  ↓ .item()
2
  ↓ int(...)
Python int
```

---

## 4. shape上の最大rankと`matrix_rank`

行列

$$
A\in\mathbb R^{m\times n}
$$

が持ち得るrank上限は

$$
\min(m,n)
$$

である。

これは

```python
min(A.shape)
```

で分かるが、「実際に非ゼロと判定される特異方向の本数」とは別である。

実際のnumerical rankは

```python
torch.linalg.matrix_rank(A)
```

で求める。

さらにtruncated SVDで「何本残すか」という選択rankはまた別である。

```text
min(A.shape)
→ shape上の最大可能rank

torch.linalg.matrix_rank(A)
→ 実際のnumerical rank

rank引数
→ 近似として残す本数
```

---

## 5. `U` のshapeと列ベクトル

$$
X^{\langle1\rangle}
\in
\mathbb R^{n_1\times(n_2n_3)}
$$

をreduced SVDすると、

$$
q=\min(n_1,n_2n_3)
$$

として

$$
U\in\mathbb R^{n_1\times q}.
$$

したがって、`U` の各列

$$
u_\alpha=U_{:,\alpha}
$$

は

$$
\boxed{u_\alpha\in\mathbb R^{n_1}}
$$

であり、各左特異ベクトルは $n_1$ 個の成分を持つ。

ここで言葉に注意する。SVDの `U` の列は**左特異ベクトル**である。量子状態として読むと、これらはSchmidtベクトルであり、対応する縮約密度行列の固有ベクトルでもある。単に「SVDの固有ベクトル」と呼ぶと混乱しやすい。

---

## 6. `reshape` の引数の個数とTensorの階数

```python
U.reshape(1, n1, r1)
```

の出力shapeは

```text
(1, n1, r1)
```

である。

指定したaxis sizeが3個なので、結果は3階Tensorである。

```text
reshape(1, n1, r1)
        ↑   ↑   ↑
       軸0 軸1 軸2
```

先頭の `1` は「1個軸を増やす」ことを意味するが、`n1` と `r1` を含めて軸は全部で3本であり、4階にはならない。

---

## 7. `reshape(..., -1)`

```python
mat = remainder.reshape(r_left * n_mode, -1)
```

の `-1` は、「総要素数が一致するように、この次元を自動計算する」という指定である。

例えば

```text
remainder.shape = (2, 3, 4, 5)
```

なら、

```python
remainder.reshape(2 * 3, -1)
```

は

```text
(6, 20)
```

になる。

TT-SVDでは右側の残りmodeの積を毎回手で計算しなくてよいため便利である。

---

## 8. 連続軸なら`reshape`で順序が保たれる

通常のcontiguous Tensorで

```python
X.reshape(n1 * n2, n3)
```

とすると、元のaxis順 `(i1, i2, i3)` を変えずに、先頭の連続した `(i1,i2)` を1個のrow indexへまとめる。

一方、

```text
i2 | (i1, i3)
```

のようにaxis順自体を変えたい場合は、先に

```python
X.permute(...)
```

が必要である。

```text
連続した現在のaxisをまとめる
→ reshape

axis順を入れ替えてからまとめる
→ permute → reshape
```

---

## 9. Python listの作成と`append`

空listは

```python
cores = []
```

と作る。

要素を末尾へ加えるには

```python
cores.append(core)
```

を使う。

例えば

```python
values = []
values.append(10)
values.append(20)
values.append(30)
print(values)
# [10, 20, 30]
```

TT-SVDではTensorの階数 $d$ によってcore数が変わるため、listへ順番に追加するのが自然である。

---

## 10. `len`

listの要素数は

```python
len(cores)
```

で得る。

TT-SVDが$d$個のcoreを返すなら

```python
len(cores) == X.ndim
```

になる。

---

## 11. `enumerate`

listの「何番目か」と「中身」を同時に取りたいときは

```python
for i, core in enumerate(cores, start=1):
    print(i, core.shape)
```

とする。

`start=1` を指定すると、Python内部のindexとは別に表示番号を1から始められる。

```text
1 G1.shape
2 G2.shape
3 G3.shape
```

のようにNotebookでcore番号を見せるときに使いやすい。

---

## 12. slice `cores[:-1]`

```python
cores[:-1]
```

は「最後の要素を除く全部」である。

```python
cores = [G1, G2, G3]
print(cores[:-1])
# [G1, G2]
```

TTでは最後のcoreのright bondは境界rank

$$
r_d=1
$$

なので、内部bond rankだけを集めるなら

```python
core_ranks = [core.shape[2] for core in cores[:-1]]
```

とする。

---

## 13. list comprehension

```python
core_ranks = [core.shape[2] for core in cores[:-1]]
```

は、展開すると

```python
core_ranks = []
for core in cores[:-1]:
    core_ranks.append(core.shape[2])
```

と同じである。

学習中は展開形で書いて処理を理解し、意味が固定してからlist comprehensionへ戻してよい。

---

## 14. dictionaryから値を取る

例えば

```python
exact_ranks = [
    {"mode": 1, "rank": torch.tensor(2)},
    {"mode": 2, "rank": torch.tensor(3)},
]
```

の1要素

```python
item = exact_ranks[0]
```

に対して

```python
item["rank"]
```

でdictionaryの `"rank"` に対応する値を取得する。

rankだけPython整数として集めるなら

```python
unfold_ranks = [
    int(item["rank"].item())
    for item in exact_ranks
]
```

と書ける。

展開すると

```python
unfold_ranks = []
for item in exact_ranks:
    rank_tensor = item["rank"]
    rank_number = rank_tensor.item()
    rank_int = int(rank_number)
    unfold_ranks.append(rank_int)
```

である。

---

## 15. list同士の比較

```python
core_ranks == unfold_ranks
```

は、listの長さ・順序・各値が全て一致するとき `True` になる。

```python
[2, 3] == [2, 3]
# True

[2, 3] == [3, 2]
# False
```

Notebook 01では

```python
print("match:", core_ranks == unfold_ranks)
```

として、TT coreから読んだ内部rankと元Tensorのcut unfolding rankを照合した。

---

## 16. `torch.tensordot` の`dims`

```python
temp = torch.tensordot(G1, G2, dims=([2], [0]))
```

では、

- `G1` のaxis 2
- `G2` のaxis 0

を縮約する。

$$
G^{(1)}:(1,n_1,r_1),
\qquad
G^{(2)}:(r_1,n_2,r_2)
$$

なので、$r_1$ が和を取る内部indexになり、

$$
(1,n_1,r_1)
\times_{r_1}
(r_1,n_2,r_2)
\rightarrow
(1,n_1,n_2,r_2)
$$

となる。

行列積で「左の列数と右の行数を消す」のと同じ発想を、高階Tensorの任意axisへ拡張したものとして読む。

---

## 17. `squeeze(dim)`

TT coreを全て縮約した直後は、境界bondが残って

```text
(1, n1, n2, ..., nd, 1)
```

となる。

```python
X_hat = X_hat_with_boundary.squeeze(0).squeeze(-1)
```

は両端のsize 1 axisだけを削除する。

```text
(1, n1, ..., nd, 1)
  ↓ squeeze(0)
(n1, ..., nd, 1)
  ↓ squeeze(-1)
(n1, ..., nd)
```

`-1` は現在の最後のaxisを意味する。

### なぜ`squeeze()`だけにしないか

```python
X_hat_with_boundary.squeeze()
```

と引数なしで呼ぶと、**全てのsize 1 axis**を消す。

もしphysical modeに

```text
n_k = 1
```

が含まれていた場合、その正当なphysical axisまで消してしまう。

したがってTT再構成では

```python
.squeeze(0).squeeze(-1)
```

と境界axisを明示する方が安全である。

---

## 18. `torch.eye`

正方単位行列は

```python
I = torch.eye(3)
```

で作る。

Tensorとdtype/deviceを合わせるなら

```python
I = torch.eye(
    n,
    dtype=X.dtype,
    device=X.device,
)
```

とする。

`torch.eye(rows, cols)` と2つ指定すれば、正方でない「主対角に1を置いた行列」も作れる。

```python
A = torch.eye(2, 4)
# shape: (2, 4)
```

TTの $U^TU=I_r$ 確認では正方の

```python
I_r = torch.eye(r, dtype=U.dtype, device=U.device)
```

を使う。

---

## 19. `torch.eye` と `torch.diag` は別

SVDの特異値 `S` は1次元Tensorである。

```text
S.shape == (r,)
```

これを

$$
\Sigma
=
\operatorname{diag}(S)
$$

という対角行列へするのが

```python
Sigma = torch.diag(S)
```

である。

```python
torch.eye(r)
```

は対角成分が全部1の単位行列であり、特異値行列ではない。

---

## 20. `.T` は転置view

```python
A.T
```

は、基本的には同じstorageを別のaxis順で読むviewであり、毎回データを物理的にコピーして並べ替える操作ではない。

そのため、見た目の隣接要素とmemory上の隣接要素が一致しない場合がある。

```text
A
→ stride (例: 3, 1)

A.T
→ stride (例: 1, 3)
```

のようにstrideが入れ替わる。

---

## 21. `.contiguous()`

```python
B = A.T.contiguous()
```

とすると、転置後の論理的な要素順を維持したまま、標準的な連続memory layoutへ必要に応じて実体化する。

すでにcontiguousなら

```python
A.contiguous()
```

は実質的に追加コピーを必要としない場合がある。

`torch.kron` 等へ渡す前にlayoutを揃える目的で使うことがあるが、

```text
SVDのUは必ずnon-contiguous
```

とは限らない。

---

## 22. `.is_contiguous()` と`stride()`

layoutを確認するには

```python
U.is_contiguous()
U.stride()
```

を使う。

値・shapeが同じでもstrideが違うTensorは存在する。

```text
数学的なTensorの値
≠
memory layout
```

を分けて考える。

---

## 23. layoutを強制的に実体化する補助策

資料の実行環境では、truncated SVD後の細い `U_hat` と `torch.kron` の組み合わせでstride/layout由来の `RuntimeError` が出るケースがあった。

まず

```python
U_hat_safe = U_hat.contiguous()
```

を試す。

それでも問題が残る環境では、検証用に

```python
U_hat_safe = U_hat.flatten().clone().view(U_hat.shape)
```

として、新しいcontiguous storageへ値を実体化した。

確認は

```python
assert U_hat_safe.shape == U_hat.shape
assert torch.allclose(U_hat_safe, U_hat)
assert U_hat_safe.is_contiguous()
```

とする。

これは数学的な $U$ を変える操作ではなく、storage/layoutだけを標準化する回避策である。

---

## 24. `U.T @ U` と `U @ U.T`

細い

$$
U\in\mathbb R^{n\times r},
\qquad
r\le n
$$

に対して

```python
U.T @ U
```

はshape

$$
r\times r
$$

で、列直交なら

$$
U^TU=I_r.
$$

一方

```python
U @ U.T
```

はshape

$$
n\times n
$$

で、$r<n$ なら一般に

$$
UU^T\ne I_n.
$$

$UU^T$ は $U$ の列空間への射影である。

```text
U.T @ U
→ 採用した列同士が直交しているか

U @ U.T
→ 元空間のどの部分へ射影するか
```

という違いを持つ。

---

## 25. `torch.kron` とreshape順

```python
L2 = torch.kron(U, I_n2)
```

は

$$
L_2=U\otimes I_{n_2}
$$

を作る。

```python
X_cut2 = X.reshape(n1 * n2, n3)
B_cut2 = B.reshape(r1 * n2, n3)
```

という現在のaxis順と揃えると、

```python
X_cut2 ≈ L2 @ B_cut2
```

を直接比較できる。

ここで重要なのは

```text
reshapeもkronも「いつでも自動的に同じ順序になる」
```

わけではないことである。今回のindex順

```text
X: (i1, i2, i3)
B: (alpha1, i2, i3)
```

と、使っているKronecker因子順が一致しているため直接比較できる。

別のblock順を採用すれば `I ⊗ U` と置換行列が現れる。

---

## 26. Notebookでは短縮形を急がない

学習中は

```python
core_ranks = []
for core in cores[:-1]:
    right_rank = core.shape[2]
    core_ranks.append(right_rank)
```

のように処理を展開してよい。

意味が固定した後で

```python
core_ranks = [core.shape[2] for core in cores[:-1]]
```

へ短縮する。

このプロジェクトでは、Notebookは「なぜそのaxisを取るのか」を確認する場所、srcは再利用性・validation・重複排除を優先する場所、と役割を分ける。
