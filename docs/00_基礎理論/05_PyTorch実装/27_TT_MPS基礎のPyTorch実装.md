---
title: TT・MPS基礎のPyTorch実装
aliases:
  - TT-SVD PyTorch
  - TT src実装
  - TT Notebook実装
tags:
  - TT
  - MPS
  - PyTorch
  - src
  - implementation
---

# TT・MPS基礎のPyTorch実装

## サマリー

TT/MPS基礎編では、理論を先に小さいNotebookで手実装して確認し、その後再利用可能な処理を `src` へ抽出した。

```text
学習Notebook
→ shape・添字・SVD・縮約を手で追う

src
→ TT cut unfolding
→ exact / truncated TT-SVD
→ dense reconstruction
→ parameter count
→ validation

src版Notebook
→ 同じ理論現象をpublic APIで再確認
```

現在の主な実装は

```text
src/nn_compression/compression/tt.py
src/nn_compression/compression/tt_validation.py
```

である。

公開APIは

```python
tt_unfold(X, k)
tt_svd_exact(X)
tt_svd(X, max_rank)
tt_reconstruct(cores)
tt_num_parameters(cores)
```

である。

---

## 1. 学習Notebookとsrcの役割

元の学習Notebookは

```text
notebooks/30_tt_mps/00_fundamentals/
├── 00_tt_svd_3way_basics.ipynb
├── 01_tt_rank_unfolding.ipynb
├── 02_truncation_error_tradeoff.ipynb
└── 03_basis_transform_rank_invariance.ipynb
```

に残す。

- 00：3階TT-SVDを2回のSVDとして手で追う
- 01：一般$d$階、TT cut rank、再構成
- 02：`max_rank`、保存量、誤差trade-off
- 03：$X^{\langle2\rangle}=(U\otimes I)B$ とrank不変性

src化後は別に

```text
notebooks/30_tt_mps/00_fundamentals_src/
```

を作り、public APIを使って同じ現象を再確認する。

元Notebookは学習履歴、src版Notebookは実用API利用例として分離する。

---

## 2. `shape`、`numel`、Python値への変換

PyTorch Tensorのshapeは

```python
X.shape
```

で `torch.Size` として得る。

```python
X = torch.randn(2, 3, 4)
X.shape
# torch.Size([2, 3, 4])
```

各次元へ分解するなら

```python
n1, n2, n3 = X.shape
```

全要素数は

```python
X.numel()
```

で得る。

$$
\operatorname{numel}(X)
=
\prod_k n_k.
$$

`torch.linalg.matrix_rank` は0階Tensorを返すので、Python整数として使うには

```python
r = torch.linalg.matrix_rank(X1).item()
```

とする。

```text
tensor(2)  -- .item() -->  2
```

srcではさらに `int(...)` を明示してPython `int` にする。

---

## 3. 「最大rank」「実際のrank」「残すrank」を分ける

行列

```python
X1.shape == (m, n)
```

に対して、shape上取り得る最大rankは

$$
\min(m,n).
$$

しかし実際のrankは

```python
torch.linalg.matrix_rank(X1)
```

で決まる。

さらに圧縮では「何本だけ残すか」というユーザー指定rankが別にある。

```text
shape上の最大rank
≠ 実際のnumerical rank
≠ 圧縮で残すrank
```

exact TT-SVDでは

```python
r = int(torch.linalg.matrix_rank(mat).item())
```

を使う。

rank上限付き版では概念的に

```python
r = min(max_rank, numerical_rank)
```

とする。

現在srcではzero tensor対応のため

```python
effective_rank = max(1, numerical_rank)
r = effective_rank if max_rank is None else min(max_rank, effective_rank)
```

としている。

---

## 4. `truncated_svd`

既存srcのwrapperは概念的に

```python
def truncated_svd(matrix, rank):
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    return U[:, :rank], S[:rank], Vh[:rank, :]
```

である。

返り値は

```text
U_r  : (m, rank)
S_r  : (rank,)
Vh_r : (rank, n)
```

で、

$$
A
\approx
U_r\operatorname{diag}(S_r)V_{h,r}
$$

と再構成する。

TT-SVDでは `rank` にshapeそのものを渡すのではなく、numerical rankまたは選択したtruncation rankを渡す。

---

## 5. reduced SVDと`full_matrices`

`torch.linalg.svd(A, full_matrices=False)` では、

$$
q=\min(m,n)
$$

として

$$
U\in\mathbb R^{m\times q},
$$

$$
S\in\mathbb R^q,
$$

$$
V_h\in\mathbb R^{q\times n}
$$

を返す。

TT-SVDで必要なのは最終的にrank $r\le q$ の列だけなので、full SVDの不要な基底を保持しないreduced SVDが自然である。

---

## 6. `tt_unfold`: TT cutのreshape

現在srcは

```python
def tt_unfold(X, k):
    left_dim = math.prod(X.shape[:k])
    right_dim = math.prod(X.shape[k:])
    return X.reshape(left_dim, right_dim)
```

という形である。

4階Tensorなら

```text
tt_unfold(X, 1) -> (n1, n2*n3*n4)
tt_unfold(X, 2) -> (n1*n2, n3*n4)
tt_unfold(X, 3) -> (n1*n2*n3, n4)
```

となる。

`tensor.operations.unfold(X, mode)` はTuckerのmode-n unfoldingであり、1つのmodeと残りを分ける。`tt_unfold` は先頭 $k$ modeと残りを分けるため意味が違う。

---

## 7. なぜ連続した軸なら`reshape`だけでよいか

PyTorchの通常のcontiguous Tensorで、先頭から連続する軸

```text
(i1, i2, ..., ik)
```

をまとめるなら、その軸順を変えずにflattenするだけなので `reshape` でよい。

```python
X.reshape(n1 * n2, n3)
```

は、元のaxis順

```text
(i1, i2, i3)
```

を保ったまま $(i_1,i_2)$ を1個のrow indexとしてまとめる。

一方、

```text
i2 | (i1, i3)
```

のように元のaxis順を入れ替えてからまとめたい場合は、先に `permute` が必要になる。

```python
X.permute(...).reshape(...)
```

したがって「reshapeだけでよいか」は、まとめたい軸が現在の並びで連続しているかに依存する。

---

## 8. TT-SVD内部では`tt_unfold`を使わない

srcのTT-SVD本体は

```python
mat = remainder.reshape(r_left * n_mode, -1)
```

を使う。

これは意図的である。

- `tt_unfold(X, k)`：元テンソル $X$ のcut rankを観察する関数
- `remainder.reshape(...)`：逐次TT-SVDで現在残っているremainderを次のSVD行列へ変える処理

第1cutでは対応するが、第2段階以降は入力Tensor自体が異なる。

特にtruncation後のremainderは元 $X$ ではなく近似側の情報なので、`tt_unfold(X,k)` をTT-SVD内部へ無理に流用しない。

---

## 9. `tt_svd_exact` / `tt_svd`

公開APIはexact版とtruncated版を分ける。

```python
cores = tt_svd_exact(X)
```

は各段階のnumerical rankを保持する。

```python
cores = tt_svd(X, max_rank=4)
```

は各段階のbond rankに上限を付ける。

内部の逐次ループはprivate helper `_tt_svd_sweep` で共有する。

中心ロジックは

```python
for mode in range(d - 1):
    n_mode = shape[mode]
    mat = remainder.reshape(r_left * n_mode, -1)

    numerical_rank = int(torch.linalg.matrix_rank(mat).item())
    effective_rank = max(1, numerical_rank)
    r = effective_rank if max_rank is None else min(max_rank, effective_rank)

    U, S, Vh = truncated_svd(mat, r)
    cores.append(U.reshape(r_left, n_mode, r))

    remainder = (torch.diag(S) @ Vh).reshape(
        r, *shape[mode + 1:]
    )
    r_left = r
```

である。

---

## 9A. $\Sigma V^T$ のPyTorch表現

`torch.linalg.svd` の `S` はshape `(r,)` の1次元Tensorである。そのため

```python
S @ Vh
```

は「対角行列 $\Sigma$ と $V^T$ の積」を意味しない。

現在のsrcでは明示的に

```python
torch.diag(S) @ Vh
```

を使う。

同じ数値はbroadcastingを使って

```python
S[:, None] * Vh
```

とも書ける。

成分では

$$
\left[S[:,None]*V_h\right]_{a,j}
=
S_a(V_h)_{a,j}
=
\left[\operatorname{diag}(S)V_h\right]_{a,j}.
$$

shapeは

```text
S[:, None] : (r, 1)
Vh         : (r, rest_dim)
result     : (r, rest_dim)
```

である。

---

## 10. zero tensorの扱い

零テンソルでは

$$
\operatorname{rank}(M_k)=0
$$

になる。

しかしTT coreのbond dimensionは正の整数として扱うため、srcでは

```python
effective_rank = max(1, numerical_rank)
```

として最低bond dimension 1を使う。

したがってzero tensorでは

```text
cut unfolding numerical rank = 0
bond dimension             = 1
```

となり、通常の

$$
\text{cut rank}=\text{bond dimension}
$$

という照合の特殊ケースになる。

これは実装contractであり、数学的なrank定義を「零テンソルのrankは1」と変えているわけではない。

---

## 11. dtype contract

現在のTT-SVD public APIが正式対応するdtypeは

```text
torch.float32
torch.float64
```

のみである。

TT専用

```python
validate_tt_dtype(X)
```

で入口を検証し、

```text
float16
bfloat16
integer
bool
complex
```

を `TypeError` で拒否する。

これは現在の内部処理

```python
torch.linalg.matrix_rank
torch.linalg.svd
```

をCPUで安定して使うためのproject contractである。Tucker側の `validate_real_dtype` は変更せず、TT専用のcontractとして分離している。

---

## 12. `max_rank` contract

`tt_svd(X, max_rank)` では、`max_rank` はboolではない1以上の整数とする。

```text
max_rank = 0      NG
max_rank < 0      NG
True / False      NG
non-integer       NG
```

0や負数を勝手に1へclampしない。

一方、`max_rank` が実際のrankより大きい場合は、単なる上限として許可する。

---

## 13. TT core validation

`tt_reconstruct` と `tt_num_parameters` は、入力core列に対して共通の

```python
validate_tt_cores(cores)
```

を使う。

確認するcontractは

- core列が空でない
- 各要素が `torch.Tensor`
- 各coreが3階
- 各dimensionが正
- 左端coreのleft bondが1
- 右端coreのright bondが1
- 隣接coreのbond dimensionが一致
- core間dtype一致
- core間device一致

である。

例えば

```text
G1: (1, n1, r1)
G2: (r1, n2, r2)
G3: (r2, n3, 1)
```

を要求する。

---

## 14. TT再構成：`torch.tensordot`

3階の場合、まず

```python
temp = torch.tensordot(G1, G2, dims=([2], [0]))
```

とする。

縮約するのは

```text
G1の右bond axis = 2
G2の左bond axis = 0
```

である。

shapeは

$$
(1,n_1,r_1)
\times_{r_1}
(r_1,n_2,r_2)
\longrightarrow
(1,n_1,n_2,r_2).
$$

次に

```python
X_hat_with_boundary = torch.tensordot(
    temp, G3, dims=([3], [0])
)
```

で

$$
(1,n_1,n_2,r_2)
\times_{r_2}
(r_2,n_3,1)
\longrightarrow
(1,n_1,n_2,n_3,1).
$$

最後に

```python
X_hat = X_hat_with_boundary.squeeze(0).squeeze(-1)
```

で境界size 1だけを除き、

$$
(n_1,n_2,n_3)
$$

へ戻す。

一般版 `tt_reconstruct` は常に現在結果の最後axisと次coreの最初axisを縮約する。

```python
result = cores[0]
for core in cores[1:]:
    result = torch.tensordot(result, core, dims=([-1], [0]))
return result.squeeze(0).squeeze(-1)
```

---

## 15. `torch.eye` と $U^TU$, $UU^T$

単位行列は

```python
I = torch.eye(n, dtype=U.dtype, device=U.device)
```

と作る。

SVDの細い $U\in\mathbb R^{n\times r}$ では

$$
U^TU=I_r
$$

だが、一般に

$$
UU^T\ne I_n.
$$

コードでは

```python
UtU = U.T @ U
UUt = U @ U.T
```

で確認できる。

$U^TU=I$ は「係数空間から元空間へ埋め込んでも情報を潰さない」こと、$UU^T$ は元空間から $U$ の列空間へのprojectionを表す。

---

## 15A. `einsum` で3コアを一度に縮約する

3サイトなら、同じ縮約を `einsum` で

```python
X_hat = torch.einsum("aib,bjc,ckd->ijk", G1, G2, G3)
```

とも書ける。

対応は

| 文字 | 数学的添字 | 役割 |
| --- | --- | --- |
| `a` | $r_0=1$ | 左境界bond |
| `i` | $i_1$ | 第1physical index |
| `b` | $\alpha_1$ | 第1bond |
| `j` | $i_2$ | 第2physical index |
| `c` | $\alpha_2$ | 第2bond |
| `k` | $i_3$ | 第3physical index |
| `d` | $r_3=1$ | 右境界bond |

入力側で繰り返し現れ、出力 `ijk` に残らない `a,b,c,d` が縮約される。

学習用00 Notebookでは途中shapeが見える `tensordot` を優先し、数式との対応が固まった後なら `einsum` で簡潔に書ける。

---

## 16. `torch.kron` と $U\otimes I$

03 Notebookでは

```python
I_n2 = torch.eye(n2, dtype=U.dtype, device=U.device)
L2 = torch.kron(U.contiguous(), I_n2)
```

として

$$
L_2=U\otimes I_{n_2}
$$

を作る。

PyTorch側の

```python
X_cut2 = X.reshape(n1 * n2, n3)
B_cut2 = B.reshape(r1 * n2, n3)
```

という行順と合わせると、

```python
X_cut2 ≈ L2 @ B_cut2
```

を直接確認できる。

資料中では途中で $I\otimes U$ のblock順も検討した。最終整理は次の通り。

```text
PyTorch reshape順に合わせる
→ U ⊗ I

別のblock順で並べる
→ I ⊗ U
→ permutation matrixで順序を対応させる
```

実装内で行順が最初から揃っていればpermutation matrixを追加する必要はない。

---

## 17. contiguous / non-contiguous / stride

Tensorのview操作では、見た目のshapeと実際のmemory layoutを分けて考える。

例えば

```python
A.T
```

は通常、データを物理的に並べ替えるのではなくstrideを入れ替えたviewを返す。

```text
A.T
→ 同じstorageを別の軸順として読む
→ 見た目で隣り合う値がmemory上では飛び飛びの場合がある
```

一方

```python
A.T.contiguous()
```

は、その見た目の順序に沿った連続memory layoutを必要に応じて作る。

`U` が必ずnon-contiguousとは限らないため、

```text
「truncated_svdのUは非連続だからcontiguousが必要」
```

とは断定しない。

より正確には、

```python
# kronに渡すUを連続メモリ配置にそろえる。
# すでにcontiguousなら実質的に追加コピーをしない。
L2 = torch.kron(U.contiguous(), I_n2)
```

と理解する。

資料の環境では、SVD由来の細いTensorで `torch.kron` がlayout/stride由来のRuntimeErrorになる場合があり、そのときは「値を変えず標準contiguous layoutへ実体化する」回避を採った。これは理論上の $U\otimes I$ とは別の実装上の問題である。

### `contiguous()` でも環境依存エラーが残る場合

資料の検証環境では、truncated SVD後の細い `U_hat` を `torch.kron` へ直接渡すとstride/layout由来の `RuntimeError` が出るケースがあった。

通常は

```python
U_hat_safe = U_hat.contiguous()
```

を最初に試す。

それでも問題が残る場合、検証時には

```python
U_hat_safe = U_hat.flatten().clone().view(U_hat.shape)
```

で論理的な要素順を保ったまま新しいcontiguous storageへ実体化した。

この操作では

```python
U_hat_safe.shape == U_hat.shape
torch.allclose(U_hat_safe, U_hat)
U_hat_safe.is_contiguous()
```

を確認する。変えているのは値ではなくmemory layoutである。

---

## 18. list・`append`・`len`・slice

一般$d$階TT-SVDではcore数が可変なのでPython listを使う。

```python
cores = []
cores.append(core)
```

個数は

```python
len(cores)
```

で取る。

最後のcoreを除いてbond rankを読むなら

```python
core_ranks = [core.shape[-1] for core in cores[:-1]]
```

とする。

`cores[:-1]` は最後の要素を除いたlistである。

---

## 19. `tt_num_parameters`

TT core列の総パラメータ数は

```python
tt_num_parameters(cores)
```

で計算する。

内部では

```python
sum(core.numel() for core in cores)
```

であり、数学式

$$
P_{\mathrm{TT}}
=
\sum_k r_{k-1}n_kr_k
$$

と一致する。

---

## 20. float64を学習Notebookで使う理由

基礎検証Notebookでは

```python
torch.set_default_dtype(torch.float64)
```

を使い、exact reconstructionやorthogonalityの誤差を

$$
10^{-14}\sim10^{-16}
$$

程度まで確認する。

目的は高速化ではなく、数式の等式が丸め誤差の範囲で成立することを見やすくすることである。

テストでdefault dtypeを一時的に変更する場合は、元のdtypeを保存して復元しglobal stateを汚さない。

```python
old_dtype = torch.get_default_dtype()
try:
    torch.set_default_dtype(torch.float64)
    ...
finally:
    torch.set_default_dtype(old_dtype)
```

---

## 21. numerical rankの注意

`torch.linalg.matrix_rank` はdefault toleranceを使う。

したがって理論上非常に小さい非ゼロ特異値を持つ行列では、shape・dtype・scaleによって数値rankが変わり得る。

このためテストで

```text
元Xのcut numerical rank
== TT core bond rank
```

をthresholdぎりぎりの人工例まで絶対contractとして扱わない。

通常の明確なrankを持つ検証例では一致を確認し、`tt_svd_exact` の「exact」はnumerical rankを打ち切らないという意味に限定する。

---

## 22. 00〜03で確認したこと

### 00：3階exact TT-SVD

- $X\in\mathbb R^{2\times3\times4}$
- 第1cut
- 2回のSVD
- core shape
- `tensordot` reconstruction
- relative Frobenius errorがmachine precision程度

### 01：一般$d$階とTT-rank

- `tt_unfold`
- exact cut ranks
- general TT-SVD
- core bond ranksとcut ranksの照合
- parameter count

### 02：truncation trade-off

`max_rank=[1,2,4,8]` をsweepし、

| max_rank | bond_ranks | tt_params | parameter ratio | relative error |
| ---: | --- | ---: | ---: | ---: |
| 1 | `[1,1,1]` | 16 | 0.0625 | 0.942147 |
| 2 | `[2,2,2]` | 48 | 0.1875 | 0.850899 |
| 4 | `[4,4,4]` | 160 | 0.6250 | 0.614882 |
| 8 | `[4,8,4]` | 288 | 1.1250 | 0.294857 |

を確認した。

### 03：基底変換とrank不変性

- `U.T @ U ≈ I`
- `L2 = U ⊗ I_n2`
- `L2.T @ L2 ≈ I`
- `X_cut2 ≈ L2 @ B_cut2`
- exact時のrank一致
- truncation後は $\widehat X$ と $\widehat B$ のrankが一致し、元 $X$ のrankとは異なり得る

---

## 22A. 03 Notebookで修正したtruncation検証

rank不変性Notebookでは、truncation後の比較対象を取り違えないことが重要である。

誤った比較は、名前だけ `X_hat_cut2` にして中身を元の `X.reshape(...)` にすることだった。正しくは、まず

```python
U_hat, S_hat, Vh_hat = truncated_svd(X1, r)
B_hat = S_hat[:, None] * Vh_hat
X1_hat = U_hat @ B_hat
X_hat = X1_hat.reshape(n1, n2, n3)
```

として近似Tensor $\widehat X$ 自体を作る。

その後

```python
X_hat_cut2 = X_hat.reshape(n1 * n2, n3)
B_hat_cut2 = B_hat.reshape(r * n2, n3)
L2_hat = torch.kron(U_hat.contiguous(), torch.eye(n2, dtype=U_hat.dtype))
```

とし、

```python
rank(X_hat_cut2) == rank(B_hat_cut2)
```

と

```python
rank(X_cut2) vs rank(X_hat_cut2)
```

を分けて確認する。

前者はtruncation後もbasis変換がrankを保存すること、後者はtruncationそのものが元Tensorのrankを変え得ることを確認する。

またrelation確認は行列そのものをprintするより

```python
relation_error = torch.linalg.matrix_norm(
    X_hat_cut2 - L2_hat @ B_hat_cut2,
    ord="fro",
).item()
```

のようにFrobenius normで数値化した方が読みやすい。

---

## 23. 現行src contractのまとめ

```text
tt_unfold
→ 1 <= k < d
→ 元XのTT cut unfolding

tt_svd_exact
→ float32 / float64
→ numerical rankを保持
→ zero rank時のbondは最低1

tt_svd
→ max_rank >= 1
→ 各段階のbond rankに上限

tt_reconstruct
→ validな3階TT core列を左から縮約

tt_num_parameters
→ validなcore列の総numel
```

この章はPyTorch上の実装contractを固定する。数学的な導出は [[30_TT_MPSの定義]] 〜 [[34_基底変換とTT-rank不変性]] に分離する。
