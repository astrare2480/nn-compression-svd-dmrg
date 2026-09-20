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

`dims=1` という整数の省略形は、左Tensorの**最後の1軸**と右Tensorの**最初の1軸**を縮約する。この第1・第2コアでは、それぞれaxis 2とaxis 0が $r_1$ なので、次の二つは同じ結果を返す。

```python
import torch

# 小さい第1・第2コアを用意する: (1, n1, r1) と (r1, n2, r2)。
G1 = torch.arange(4, dtype=torch.float64).reshape(1, 2, 2)
G2 = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)

# 最後の1軸と最初の1軸が同じbond r1である場合に限り、dims=1と書ける。
G12_short = torch.tensordot(G1, G2, dims=1)
G12_explicit = torch.tensordot(G1, G2, dims=([2], [0]))
assert torch.allclose(G12_short, G12_explicit)
# 両方ともshapeは(1, n1, n2, r2)。

# bだけを和で消し、残す軸をa,i,j,cの順に指定する。
G12_einsum = torch.einsum("aib,bjc->aijc", G1, G2)
assert torch.allclose(G12_explicit, G12_einsum)
```

収縮したい軸がこの配置にない場合、`dims=1` で別の軸を消してはいけない。軸番号を明示するか、添字を文字で指定する `einsum` を使う。同じ計算は上の `G12_einsum` のように書ける。`"aib,bjc->aijc"` では、二つの入力にある `b` が出力にないので和で消え、`a,i,j,c` は記した順の軸として残る。成分では

$$
\mathrm{G12}(a,i,j,c)=\sum_{b=1}^{r_1}G_1(a,i,b)G_2(b,j,c)
$$

である。`tensordot` は指定した収縮軸を消し、両入力の残りの軸を順に並べる。`einsum` は `->` の右側で残す添字とその順序を明示でき、3コア以上を一式で表すときにも使える。読むときは入力の各軸に文字を対応させ、出力に書かない添字を特定し、最後に出力shapeを確認する。3コアの再構成では境界のサイズ1の添字も出力から省くが、その意味は [[27_TT_MPS基礎のPyTorch実装]] で扱う。

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

QRの $Q$ が `(m,r)` なら、列Gram `Q.T @ Q` のshapeは `(r,r)` なので、比較する単位行列は `torch.eye(Q.shape[1], dtype=Q.dtype, device=Q.device)` とする。`torch.eye(Q.shape)` のようにshape全体を一つの引数に渡さず、また行数 `Q.shape[0]` と列数を取り違えない。

### eyeの出力を全要素で確認する

`torch.eye(3)` の出力を全要素で書くと

$$
\operatorname{eye}(3)=
\begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix},\qquad
\operatorname{eye}(2,4)=
\begin{pmatrix}1&0&0&0\\0&1&0&0\end{pmatrix}.
$$

後者は単位行列 $I_4$ ではなく、2行4列の主対角部分へ1を置いた行列である。
全要素は整数値でも、Tensorのdtypeは呼び出しで指定した型になる。
`dtype=torch.int64` は整数型の例、
`dtype=torch.float64` は倍精度の例である。
計算相手 `X` へ合わせるときは、既存のコードどおりdtypeとdeviceの両方を渡す。

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

### 特異値5・2・1を、diagの全成分へ対応させる

`S = torch.tensor([5.0, 2.0, 1.0])` を使う。

$$
\Sigma=\operatorname{diag}(5,2,1)
=\begin{pmatrix}5&0&0\\0&2&0\\0&0&1\end{pmatrix},\qquad
I_3=\begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix}.
$$

`torch.diag(S) @ Vh` の第1行は `Vh` の第1行の5倍、第2行は2倍、
第3行は1倍である。`torch.eye(3) @ Vh` なら全行を元のまま残し、
特異値の係数を掛けたremainderにはならない。
`S[:, None] * Vh` が同じremainderを作る理由も、各行へ `S` の対応成分を掛けるためである。
これは対角化APIの元例であり、[[33_TT-SVDの打ち切りと誤差]] のrank 2打ち切り例と数値を混同しない。

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

### 0〜5の値を、転置とcontiguousのメモリ位置まで追う

値が変わらないことと格納順が変わることを、小さい配列で区別する。
6要素を表示すると

$$
A=\begin{pmatrix}0&1&2\\3&4&5\end{pmatrix},\qquad
B=A^T=\begin{pmatrix}0&3\\1&4\\2&5\end{pmatrix},\qquad
C=\operatorname{contiguous}(B)
=\begin{pmatrix}0&3\\1&4\\2&5\end{pmatrix}.
$$

元の `A` はメモリ上で `0, 1, 2, 3, 4, 5` の順に並び、
0始まりの位置は `A[i,j]` に対して $3i+j$ である。
`B[i,j]=A[j,i]` は同じstorageの位置 $i+3j$ を読む。
第0行の `B[0,0], B[0,1]` は位置0と3、
第1行は位置1と4、第2行は位置2と5から読むので、各行の2要素は隣接しない。
ここでtransposeは値を `0, 3, 1, 4, 2, 5` へ物理的に格納し直してはいない。

`C` はこのnon-contiguousな `B` を連続layoutへコピーするので、
位置は $2i+j$、実際の格納順は `0, 3, 1, 4, 2, 5` になる。
`B` と `C` のshapeと各添字の値は同じである。

```python
import torch

# 6個の値を用い、すべて2階Tensorとして比較する。
A = torch.tensor([[0, 1, 2], [3, 4, 5]])
B = A.T
C = B.contiguous()

assert A.stride() == (3, 1)
assert B.stride() == (1, 3) and not B.is_contiguous()
assert C.stride() == (2, 1) and C.is_contiguous()
assert torch.equal(B, C)
assert A.untyped_storage().data_ptr() == B.untyped_storage().data_ptr()
assert B.untyped_storage().data_ptr() != C.untyped_storage().data_ptr()
assert C.view(-1).tolist() == [0, 3, 1, 4, 2, 5]
```

この例の `B.view(-1)` はstrideを保った1次元viewとして表せず失敗する。
`B.reshape(-1)` は必要ならコピーし、同じ論理順 `[0, 3, 1, 4, 2, 5]` を作れる。
一方、すでに連続なTensorへの `contiguous()` は通常追加コピーをしない。
転置で「別の角度から見る」とは、添字からstorageの位置への読み方を変えることを指す。
このメモリlayoutの話と、[[29_TT_cutとPyTorchのreshape_Kronecker順序]] の
physical/bondの複合添字を入れ替える話は、別の確認として扱う。

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

truncated SVD後の細い `U_hat` と `torch.kron` の組み合わせでは、環境によってstride/layout由来の `RuntimeError` が出る場合がある。

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

### 細い1列の例を、`flatten` → `clone` → `view` の各段階で追う

layoutを実体化する操作を、一段階ずつ確認する。
以下は**1列sliceのlayout**を再現する人工例である。

$$
T=\begin{pmatrix}1/3&2/3&2/3\\4&5&6\end{pmatrix},\qquad
U_{\mathrm{hat}}=T^T[:,0:1]
=\begin{pmatrix}1/3\\2/3\\2/3\end{pmatrix},\qquad
U_{\mathrm{hat}}^TU_{\mathrm{hat}}
=\begin{pmatrix}1/9+4/9+4/9\end{pmatrix}
=\begin{pmatrix}1\end{pmatrix}.
$$

`T` のstrideは $(3,1)$、`T.T` は $(1,3)$ である。
この1列sliceのshapeは $(3,1)$、strideは $(1,3)$ になる。
サイズ1の列軸には隣の列が存在せず、そのstrideが3でも `is_contiguous()` は `True` になる。
そのため `contiguous()` が同じTensorを返してstride $(1,3)$ が残ることがある。
「連続と判定された」と「全strideが期待する標準値である」は完全には同じではない。

実体化の各段階は

$$
\begin{aligned}
U_{\mathrm{hat}}&:(3,1),\quad \operatorname{stride}=(1,3),\\
f=\operatorname{flatten}(U_{\mathrm{hat}})
&=\begin{pmatrix}1/3&2/3&2/3\end{pmatrix}\quad\text{（1次元・3要素）},\\
c=\operatorname{clone}(f)&\quad\text{（同じ3値・新しいstorage）},\\
U_{\mathrm{safe}}=\operatorname{view}(c,3,1)
&=\begin{pmatrix}1/3\\2/3\\2/3\end{pmatrix},\quad
\operatorname{stride}=(1,1).
\end{aligned}
$$

flattenは論理順で1次元化し、必要ならコピーする。cloneは新しいstorageを明示的に作り、
最後のviewはその連続な3値を元shapeへ戻す。
新しいstorageを作るコストはあるが、値・shape・dtype・deviceは保持する。
この行列でのKronecker積も、数学上は実体化の前後で同じである。

$$
U_{\mathrm{safe}}\otimes I_2
=\begin{pmatrix}
1/3&0\\0&1/3\\2/3&0\\0&2/3\\2/3&0\\0&2/3
\end{pmatrix},\qquad
(U_{\mathrm{safe}}\otimes I_2)^T(U_{\mathrm{safe}}\otimes I_2)
=(1/9+4/9+4/9)I_2=I_2.
$$

```python
import torch

# SVDとは独立に、転置後の1列sliceを用意する。
T = torch.tensor([[1 / 3, 2 / 3, 2 / 3], [4.0, 5.0, 6.0]], dtype=torch.float64)
U_hat = T.T[:, :1]
flat = U_hat.flatten()
copied = flat.clone()
U_safe = copied.view(U_hat.shape)
assert U_hat.stride() == (1, 3) and U_hat.is_contiguous()
assert U_hat.contiguous() is U_hat
assert U_safe.stride() == (1, 1) and U_safe.is_contiguous()
assert torch.equal(U_safe, U_hat)
assert U_safe.dtype == U_hat.dtype and U_safe.device == U_hat.device
assert U_safe.untyped_storage().data_ptr() != U_hat.untyped_storage().data_ptr()
I2 = torch.eye(2, dtype=U_hat.dtype, device=U_hat.device)

# 直接kronの成否は環境依存なので、失敗自体を必須条件にはしない。
try:
    L_direct = torch.kron(U_hat, I2)
except RuntimeError as error:
    print("direct kron:", str(error))
else:
    assert torch.equal(L_direct, torch.kron(U_safe, I2))
L_safe = torch.kron(U_safe, I2)
assert torch.allclose(U_safe.T @ U_safe, torch.ones(1, 1, dtype=U_hat.dtype))
assert torch.allclose(L_safe.T @ L_safe, I2)
```

直接kronの成否はversion/deviceに依存する。エラーが出ること自体を、この例の成功条件にはしない。
通常はまずshape・stride・値を確認し、必要な場合だけこの補助策を使う。
`contiguous` の同一Tensor返却とview/copyの扱いは
[PyTorch公式contiguous](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html) と
[Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html) を参照する。

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

QRの $Q$ が $(m,r)$ で $r<m$ なら、`Q @ Q.T` は $(m,m)$ だが単位行列とは限らない。表示された各要素が0・1でなくても、列が正規直交なら射影として正しい。shapeだけで判断せず、対称性・冪等性・traceを分けて確認する。

```python
import torch

# 4次元の中の2次元部分空間を、reduced QRの列で表す。
A = torch.tensor(
    [[1.0, 1.0], [0.0, 1.0], [1.0, -1.0], [0.0, 0.0]],
    dtype=torch.float64,
)
Q, _ = torch.linalg.qr(A, mode="reduced")
P = Q @ Q.T
I_cols = torch.eye(Q.shape[1], dtype=Q.dtype, device=Q.device)

assert torch.allclose(Q.T @ Q, I_cols)
assert torch.allclose(P.T, P)
assert torch.allclose(P @ P, P)
assert torch.allclose(torch.trace(P), torch.tensor(2.0, dtype=P.dtype))
print(P)
```

この入力では $A$ の二列が互いに直交し、長さの二乗が2と3なので、QRで各列の符号が変わっても $P$ の全要素は

$$
P=\begin{pmatrix}
5/6&1/3&1/6&0\\
1/3&1/3&-1/3&0\\
1/6&-1/3&5/6&0\\
0&0&0&0
\end{pmatrix}.
$$

対角和は $5/6+1/3+5/6+0=2$ である。対角成分が0または1だけでなく、非対角成分が非零でも、4次元空間の中の2次元部分空間への射影として正しい。

`torch.trace(P)` が列数2に近いことは射影先の次元の確認であり、それだけで射影と判定する条件ではない。$Q^TQ=I$ を満たすことと $P^2=P$ も確認する。理論上の理由、非正規化列を使った反例、全要素の計算は [[35_TT_MPSの等長写像と射影]] に置く。

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

---

## 27. QRの追補：入力の行・列と新しい基底

以下は左右QRで使う操作メモである。既存のTT-SVD public APIへ新しいQR APIが追加されたという意味ではない。

`Q, R_qr = torch.linalg.qr(A, mode="reduced")` で、入力が `(m,n)` なら `q=min(m,n)`、`Q.shape==(m,q)`、`R_qr.shape==(q,n)` となる。本追補の `R_qr` は理論docs [[36_TT_MPSのGauge自由度と左QR直交化]] の三角因子 $T$ に対応する。

$$
A(a,b)=\sum_{\gamma=1}^{q}Q(a,\gamma)R_{\mathrm{qr}}(\gamma,b).
$$

行 $a$ は `Q`、列 $b$ は `R_qr` に残り、$\gamma$ は新しい基底である。QR後の列を元のボンド添字と同一視しない。また、`mode="reduced"` だけでnumerical rankを選ぶわけではない。rank欠損入力の数値挙動・微分には注意が必要である。

左ではコアを `(r_left*n_mode, r_right)` に、右では `(r_left, n_mode*r_right)` に行列化してから**その転置**をQRする。元のボンドサイズが維持できる次元条件と、横長のときの更新後shapeは理論章を参照する。出力を常に古いrankのままreshapeしてはいけない。

### `reduced`・`complete`・`r` の出力を要素まで比べる

入力が $(m,n)$、$q=\min(m,n)$ のとき、`reduced` は $Q:(m,q)$、$R_{\mathrm{qr}}:(q,n)$、`complete` は $Q:(m,m)$、$R_{\mathrm{qr}}:(m,n)$ を返す。`r` は $Q$ が空で、$R_{\mathrm{qr}}:(q,n)$ だけを計算する。第1コアを $Q$ へ置き換える演習に `r` は使えない。

$m=3,n=2$ の小例を

$$
A=\begin{pmatrix}1&0\\0&1\\0&0\end{pmatrix}
$$

とする。符号の選び方を一つ固定すれば、reduced QRとcomplete QRはそれぞれ

$$
\begin{aligned}
Q_{\mathrm{red}}&=\begin{pmatrix}1&0\\0&1\\0&0\end{pmatrix},
&R_{\mathrm{red}}&=\begin{pmatrix}1&0\\0&1\end{pmatrix},
&Q_{\mathrm{red}}R_{\mathrm{red}}&=A,\\
Q_{\mathrm{full}}&=\begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix},
&R_{\mathrm{full}}&=\begin{pmatrix}1&0\\0&1\\0&0\end{pmatrix},
&Q_{\mathrm{full}}R_{\mathrm{full}}&=A.
\end{aligned}
$$

completeの第3列は入力の二つの列を表すのに不要な補完方向である。$(1,3,2)$ の第1TTコアへ戻すには6要素の $Q_{\mathrm{red}}$ がそのまま対応し、9要素の $Q_{\mathrm{full}}$ はそのままreshapeできない。なおPyTorchのQRは列符号が異なる正しい組を返すことがあるため、手書きの $Q$ との値の一致ではなくshape、$QR=A$、Gramを検査する。横長の $m\le n$ ではreducedとcompleteのshapeは一致する。

```python
import torch

# 3行2列の入力で三つのmodeのshapeと再構成を比べる。
A = torch.tensor([[1., 0.], [0., 1.], [0., 0.]], dtype=torch.float64)
Q_red, R_red = torch.linalg.qr(A, mode="reduced")
Q_full, R_full = torch.linalg.qr(A, mode="complete")
Q_empty, R_only = torch.linalg.qr(A, mode="r")

assert Q_red.shape == (3, 2) and R_red.shape == (2, 2)
assert Q_full.shape == (3, 3) and R_full.shape == (3, 2)
assert Q_empty.numel() == 0 and R_only.shape == (2, 2)
assert torch.allclose(Q_red @ R_red, A)
assert torch.allclose(Q_full @ R_full, A)
assert torch.allclose(Q_red.T @ Q_red, torch.eye(2, dtype=A.dtype))
assert Q_red.reshape(1, 3, 2).shape == (1, 3, 2)
```

## 28. `unsqueeze` は転置の代わりではない

`unsqueeze(dim)` は指定位置へsize 1の軸を追加し、要素の値を変えない。行列の行・列を交換する `.T` とは別の操作である。

配列操作だけを示す小行列を

$$
B=\begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix},\qquad
B^T=\begin{pmatrix}1&3&5\\2&4&6\end{pmatrix}
$$

とすると、`B.T.unsqueeze(-1)` の唯一の最終軸sliceは $B^T$ で、shapeは `(2,3,1)`。`B.unsqueeze(-1)` のsliceは $B$ のままで `(3,2,1)` となる。

```python
import torch

# 軸交換と境界軸の追加を、要素番号で区別する。
B = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64)
B_with_boundary = B.T.unsqueeze(-1)
assert B_with_boundary.shape == (2, 3, 1)
assert B_with_boundary[1, 2, 0].item() == B[2, 1].item() == 6.0
assert torch.equal(B_with_boundary.squeeze(-1), B.T)
assert B.unsqueeze(-1).shape == (3, 2, 1)
```

第1コアでは `Q1.unsqueeze(0)` で `(1,n1,q1)`、右端コアでは `Q3.T.unsqueeze(-1)` で `(q2,n3,1)` とする。`n3==q2` なら転置忘れでもshapeが同じになるため、shapeだけでなく縮約後の値を検証する。

`.T` はここでは**2階行列**にだけ使う。3階コアの軸交換には `transpose(dim0,dim1)` または `permute(...)` を明示する。複素行列の共役転置は `.mH` であり、実数の `.T` と区別する。`squeeze()` の引数を省くと、境界以外のsize 1 physical/bond axisも消し得る。

## 29. 三角因子の吸収：`tensordot` の残る軸

第1QRの吸収は `torch.tensordot(R1_qr, G2, dims=([1], [0]))`。入力を `(beta1,alpha1)` と `(alpha1,i2,alpha2)` と読めば、消えるのは `alpha1`、出力順は `(beta1,i2,alpha2)` になる。

右端QRの吸収は `torch.tensordot(G2, R3_qr.T, dims=([2], [0]))`。入力は `(alpha1,i2,alpha2)` と `(alpha2,beta2)` なので、出力順は `(alpha1,i2,beta2)` になる。

`tensordot` の結果は、最初の入力の非縮約軸、次の入力の非縮約軸の順で並ぶ。単に二つの入力を逆にすると、同じshapeや添字順になるとは限らない。

```python
import torch

# 右吸収の古いbondはG2のaxis 2。新しいbondは出力のaxis 2へ残る。
G2 = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2)
R3_qr = torch.tensor([[1., 2.], [0., 1.]], dtype=G2.dtype)
updated = torch.tensordot(G2, R3_qr.T, dims=([2], [0]))
same = torch.einsum("aib,bc->aic", G2, R3_qr.T)
assert updated.shape == (2, 3, 2)
assert torch.equal(updated, same)
# 元sliceの行[0,1]とR3_qr.Tの積は[2,1]。
assert torch.equal(updated[0, 0], torch.tensor([2., 1.], dtype=G2.dtype))
```

理論上の全要素計算と全テンソル不変性は [[38_TT_MPSの右QR直交化と右ブロック]] に置く。

### Gauge変換の二つの `einsum` を対で確認する

左コアだけへ可逆行列 $M$ を掛けると全体は変わる。右隣へ $M^{-1}$ を同時に掛ける必要がある。コードの文字 `b` は古いbond、`c` は新しいbondと読む。

$$
\begin{aligned}
G_1'(a,i,c)&=\sum_bG_1(a,i,b)M(b,c),\\
G_2'(c,j,d)&=\sum_bM^{-1}(c,b)G_2(b,j,d),\\
\sum_cG_1'(a,i,c)G_2'(c,j,d)
&=\sum_bG_1(a,i,b)G_2(b,j,d).
\end{aligned}
$$

`G_1'` と `G_2'` の値は変わるが、縮約後の全要素は変わらない。次の例は [[36_TT_MPSのGauge自由度と左QR直交化]] の手計算とは別の小行列を使う。

```python
import torch

# 境界軸を保った2サイトの全要素を指定する。
G1 = torch.tensor([[[1., 0.], [2., 1.]]], dtype=torch.float64)
G2 = torch.tensor([[[1.], [0.]], [[0.], [1.]]], dtype=G1.dtype)
M = torch.tensor([[1., 2.], [0., 1.]], dtype=G1.dtype)
M_inverse = torch.linalg.inv(M)

# bを古いbond、cを新しいbondとして、隣接二コアを対で更新する。
G1_new = torch.einsum("aib,bc->aic", G1, M)
G2_new = torch.einsum("cb,bjd->cjd", M_inverse, G2)
before = torch.einsum("aib,bjd->aijd", G1, G2)
after = torch.einsum("aic,cjd->aijd", G1_new, G2_new)

assert G1_new.shape == G1.shape and G2_new.shape == G2.shape
assert not torch.equal(G1_new, G1)
assert torch.allclose(after, before)
assert torch.equal(
    before[0, :, :, 0],
    torch.tensor([[1., 0.], [2., 1.]], dtype=G1.dtype),
)
```

## 30. 左右ブロックの配列・行列・Gramを区別する

`torch.einsum("aib,bjc->ijc", G1_left, G2_left)` では、内部bond `b` とsize 1の左境界 `a` を消し、`(i1,i2,alpha2)` が残る。次に `reshape(n1*n2,r2)` で $(i_1,i_2)$ を行にする。実数の左Gramは `L2_block.T @ L2_block` である。

3階TTの右ブロックは `R2_block = G3_right.squeeze(-1)`。shapeは `(r2,n3)`、右Gramは `R2_block @ R2_block.T` である。すでに右QRした第3コアから境界を外すだけなので、ここで三角因子を再吸収しない。

`R2_block` と `R2_qr` は別の対象である。また、以前の `L2 = torch.kron(U, I_n2)` は2コア左ブロックではなく、`(n1*n2,r1*n2)` の基底拡張行列である。命名を `L2_expand` / `L2_block` のように分けると取り違えを防げる。詳しくは [[37_TT_MPSの左ブロックと直交性の導出]] を参照する。

## 31. 直交性・局所不変性・全体不変性を別々に測る

Gramと単位行列の差は、`torch.linalg.matrix_norm(gram-I, ord="fro").item()` で数値化する。`I` はGramの大きさで作り、dtype/deviceをそろえる。

例えば2階の誤差行列 $E=\begin{pmatrix}3&4\\0&12\end{pmatrix}$ なら、要素を二乗して足し、平方根を取る。

$$
\|E\|_F=\sqrt{3^2+4^2+0^2+12^2}=\sqrt{169}=13.
$$

```python
import torch

# 2階行列のFrobeniusノルムを明示して計算する。
E = torch.tensor([[3.0, 4.0], [0.0, 12.0]], dtype=torch.float64)
fro = torch.linalg.matrix_norm(E, ord="fro")
assert torch.allclose(fro, torch.tensor(13.0, dtype=E.dtype))
```

2階行列に対しては従来の `torch.norm(E)` も同じ値を返すが、このAPIは非推奨である。行列のFrobeniusノルムであることを明示するため、ここでは `torch.linalg.matrix_norm(E, ord="fro")` を使う。ベクトルや高階Tensorのノルムを、2階行列のFrobeniusノルムと無条件に同一視しない。

$$
e_{\mathrm{orth}}=\|\mathrm{Gram}-I\|_F,\qquad
e_{\mathrm{local}}=\|C_{\mathrm{before}}-C_{\mathrm{after}}\|_F,\qquad
e_{\mathrm{whole}}=\|X_{\mathrm{before}}-X_{\mathrm{after}}\|_F.
$$

全体相対誤差は $X_{\mathrm{before}}\ne0$ のとき $e_{\mathrm{whole}}/\|X_{\mathrm{before}}\|_F$ とし、零テンソルでは絶対誤差を見る。float64の小例では丸め誤差程度の一致を期待するが、特定の $10^{-16}$ を全環境の保証値にはしない。入力のscaleに応じた絶対・相対許容値を使う。

QRの列符号は非一意なので、手計算と `Q,R_qr` の符号が違っても誤りとは限らない。Gramと因子の積、吸収後の再構成を検証する。

## 参考資料：QR追補

- [PyTorch：torch.linalg.qr](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html)
- [PyTorch：torch.einsum](https://docs.pytorch.org/docs/stable/generated/torch.functional.einsum.html)
- [PyTorch：torch.tensordot](https://docs.pytorch.org/docs/stable/generated/torch.tensordot.html)
- [PyTorch：torch.norm](https://docs.pytorch.org/docs/stable/generated/torch.norm.html)：非推奨の注意と代替API。
- [PyTorch：torch.unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html)
- [PyTorch：torch.squeeze](https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html)
