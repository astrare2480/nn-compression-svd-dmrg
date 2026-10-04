---
title: PyTorchのvector_norm matrix_normとtorch.linalg
tags:
  - PyTorch
  - vector-norm
  - matrix-norm
  - singular-value
  - numerical-linear-algebra
---

# PyTorchの`vector_norm`・`matrix_norm`と`torch.linalg`

このノートでは、誤差上界の数式に現れるnormとPyTorch APIを対応させる。2階tensorに対して同じ数値を返す場合でも、vector norm、Frobenius norm、spectral normは意味が異なる。

## 1. 要素ごとの絶対値はnormではない

行列

$$
A=(a_{ij})
$$

に対する`A.abs()`は、

$$
|A|=(|a_{ij}|)
$$

を返す。入力と同じshapeを持つ行列であり、1個のスカラーへ集約するnormではない。

~~~python
import torch

a = torch.tensor(
    [[3.0, -5.0], [2.0, 4.0]],
    dtype=torch.float64,
)

# 各要素の絶対値を返す。shapeは(2, 2)のままである。
abs_a = a.abs()

assert abs_a.shape == a.shape
~~~

ReLUの要素不等式

$$
|\Delta H_{ij}|
\le
|\Delta Z_{ij}|
$$

を検査するときは`abs()`が必要である。一方、全体誤差 $\|\Delta H\|_F$ を求めるときはnorm APIを使う。

## 2. `torch.linalg.vector_norm`

`torch.linalg.vector_norm`は、指定した軸上の要素をvectorと見てnormを計算する。`dim`を指定しない場合、入力の全要素を1本のvectorへ並べたものとして扱う。

$$
A=
\begin{pmatrix}
3&-4\\
0&12
\end{pmatrix}
$$

を全要素のvector

$$
\operatorname{vec}(A)=(3,-4,0,12)
$$

と見れば、

$$
\|\operatorname{vec}(A)\|_1
=
3+4+0+12
=
19,
$$

$$
\|\operatorname{vec}(A)\|_2
=
\sqrt{3^2+(-4)^2+0^2+12^2}
=
13,
$$

$$
\|\operatorname{vec}(A)\|_\infty
=
\max(3,4,0,12)
=
12.
$$

~~~python
import torch
from torch.testing import assert_close

a = torch.tensor(
    [[3.0, -4.0], [0.0, 12.0]],
    dtype=torch.float64,
)

# dimを指定しないため、全要素を1本のvectorとして扱う。
l1 = torch.linalg.vector_norm(a, ord=1)
l2 = torch.linalg.vector_norm(a, ord=2)
linf = torch.linalg.vector_norm(a, ord=float("inf"))

assert_close(l1, torch.tensor(19.0, dtype=a.dtype))
assert_close(l2, torch.tensor(13.0, dtype=a.dtype))
assert_close(linf, torch.tensor(12.0, dtype=a.dtype))

# dim=1なら、各行を別々のvectorとして2-normを求める。
row_l2 = torch.linalg.vector_norm(a, ord=2, dim=1)
assert_close(
    row_l2,
    torch.tensor([5.0, 12.0], dtype=a.dtype),
)
~~~

よく使う`ord`は次である。

- `ord=1`：絶対値の総和
- `ord=2`：Euclidean norm
- `ord=float("inf")`：最大絶対値
- `ord=0`：非零要素数
- 一般の`ord=p`：$p$-norm

## 3. `torch.linalg.matrix_norm`

`torch.linalg.matrix_norm`は、入力の最後の2軸を行列として扱う。行列normを数式の意味どおりに書き分けたいときに使う。

主な`ord`は次である。

| `ord` | 行列normの意味 |
| --- | --- |
| `"fro"` | Frobenius norm |
| `"nuc"` | nuclear norm、特異値の総和 |
| `2` | spectral norm、最大特異値 |
| `-2` | 最小特異値 |
| `1` | 列ごとの絶対値和の最大 |
| `-1` | 列ごとの絶対値和の最小 |
| `float("inf")` | 行ごとの絶対値和の最大 |
| `-float("inf")` | 行ごとの絶対値和の最小 |

### 3.1 Frobenius norm

$$
\|A\|_F
=
\sqrt{\sum_i\sum_j|a_{ij}|^2}.
$$

~~~python
# 行列全要素の二乗和平方根を計算する。
fro = torch.linalg.matrix_norm(a, ord="fro")
assert_close(fro, torch.tensor(13.0, dtype=a.dtype))
~~~

2階tensorでは、

$$
\|\operatorname{vec}(A)\|_2
=
\|A\|_F
$$

なので、`vector_norm(A, ord=2)`と`matrix_norm(A, ord="fro")`は同じ数値になる。しかし、前者は全要素をvectorとして測り、後者は行列のFrobenius normを測るという意味をコード上で表す。

Frobenius normは要素の二乗和から計算されるため、`matrix_norm(a.abs(), ord="fro")`としても同じ数値になるが、先に`.abs()`を付ける必要はない。`matrix_norm(a, ord="fro")`だけで負の要素も正しく扱われる。

例えば、

$$
C=
\begin{pmatrix}
3&-5\\
2&4
\end{pmatrix}
$$

では、

$$
\|\operatorname{vec}(C)\|_2
=
\|C\|_F
=
\sqrt{3^2+(-5)^2+2^2+4^2}
=
\sqrt{54}.
$$

### 3.2 spectral norm

行列の2-normは最大特異値である。

$$
\|A\|_2
=
\sigma_{\max}(A).
$$

~~~python
# 最大特異値を返す。Frobenius normとは一般に異なる。
spectral = torch.linalg.matrix_norm(a, ord=2)

# 特異値を直接求め、最大値と一致することを確認する。
singular_values = torch.linalg.svdvals(a)
assert_close(spectral, singular_values.max())
~~~

誤差上界

$$
\|X\Delta W^{\mathsf T}\|_F
\le
\|X\|_F\|\Delta W\|_2
$$

の $\|\Delta W\|_2$ は、このspectral normである。

### 3.3 行列1-normと無限大norm

$$
D=
\begin{pmatrix}
1&-2\\
3&4
\end{pmatrix}
$$

とする。列ごとの絶対値和は

$$
|1|+|3|=4,
\qquad
|-2|+|4|=6
$$

なので、行列1-normは

$$
\|D\|_1=6
$$

である。行ごとの絶対値和は

$$
|1|+|-2|=3,
\qquad
|3|+|4|=7
$$

なので、行列無限大normは

$$
\|D\|_\infty=7
$$

である。

一方、全要素をvectorとして見た1-normは

$$
\|\operatorname{vec}(D)\|_1
=
1+2+3+4
=
10
$$

であり、行列1-normとは異なる。

~~~python
d = torch.tensor(
    [[1.0, -2.0], [3.0, 4.0]],
    dtype=torch.float64,
)

# 行列1-normは最大列和である。
matrix_l1 = torch.linalg.matrix_norm(d, ord=1)

# 行列無限大normは最大行和である。
matrix_linf = torch.linalg.matrix_norm(d, ord=float("inf"))

# vector 1-normは全要素の絶対値和である。
vector_l1 = torch.linalg.vector_norm(d, ord=1)

assert_close(matrix_l1, torch.tensor(6.0, dtype=d.dtype))
assert_close(matrix_linf, torch.tensor(7.0, dtype=d.dtype))
assert_close(vector_l1, torch.tensor(10.0, dtype=d.dtype))
~~~

## 4. nuclear norm

nuclear normは全特異値の和である。

$$
\|A\|_\ast
=
\sum_k\sigma_k(A).
$$

~~~python
# nuclear normと特異値和を別々に計算して照合する。
nuclear = torch.linalg.matrix_norm(a, ord="nuc")
singular_values = torch.linalg.svdvals(a)
assert_close(nuclear, singular_values.sum())
~~~

nuclear normは低rank化の理論で現れることがあるが、ReLUのFrobenius誤差伝播や今回のLinear出力誤差上界には使わない。

## 5. `torch.linalg.norm`との違い

`torch.linalg.norm`はvector normとmatrix normの両方を扱う汎用APIである。ただし、入力次元、`ord`、`dim`の組合せで意味が変わる。

教材では、数式の意味をコードに残すため、次のように明示的なAPIを優先する。

~~~python
# 行列全体のFrobenius normであることを明示する。
delta_h_fro = torch.linalg.matrix_norm(delta_h, ord="fro")

# 行列のspectral normであることを明示する。
delta_w_spectral = torch.linalg.matrix_norm(delta_w, ord=2)

# sampleごとのhidden vector 2-normならdimを明示する。
sample_errors = torch.linalg.vector_norm(delta_h, ord=2, dim=1)
~~~

## 6. 特異値・rank・条件数に関するAPI

### 6.1 `torch.linalg.svdvals`

特異値だけが必要な場合に使う。特異値 $\sigma_k$ から、

$$
\|A\|_2=\max_k\sigma_k,
$$

$$
\|A\|_F
=
\sqrt{\sum_k\sigma_k^2},
$$

$$
\|A\|_\ast
=
\sum_k\sigma_k
$$

を再構成できる。

~~~python
singular_values = torch.linalg.svdvals(a)

# 特異値から3種類のnormを組み立てる。
spectral_from_s = singular_values.max()
fro_from_s = torch.linalg.vector_norm(singular_values, ord=2)
nuclear_from_s = singular_values.sum()

assert_close(
    spectral_from_s,
    torch.linalg.matrix_norm(a, ord=2),
)
assert_close(
    fro_from_s,
    torch.linalg.matrix_norm(a, ord="fro"),
)
assert_close(
    nuclear_from_s,
    torch.linalg.matrix_norm(a, ord="nuc"),
)
~~~

### 6.2 `torch.linalg.matrix_rank`

数値的に非零とみなす特異値の個数を返す。TT cut unfoldingのrank確認にも使えるが、許容誤差によって数値rankが変わり得るため、厳密な代数rankと無条件に同一視しない。

### 6.3 `torch.linalg.cond`

条件数は、可逆な正方行列なら典型的に

$$
\kappa_2(A)
=
\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}
$$

である。数値計算の感度を調べる量だが、ReLUの1-Lipschitz不等式を検査するためには不要である。

### 6.4 `torch.linalg.pinv`

Moore–Penrose擬似逆行列を求める。最小二乗問題やrank落ち行列で有用だが、今回の誤差伝播では逆行列を作らないため不要である。

## 7. ReLU誤差伝播で使う最小API

今回のtoy検証に必要な操作は、次で十分である。

~~~python
# 要素ごとの誤差を作る。
abs_delta_h = delta_h.abs()
abs_delta_z = delta_z.abs()

# 浮動小数点の丸めを許して、全要素の不等式を判定する。
elementwise_holds = abs_delta_h <= abs_delta_z + inequality_atol
all_elements_hold = torch.all(elementwise_holds)

# batch全体のFrobenius誤差を計算する。
delta_h_fro = torch.linalg.matrix_norm(delta_h, ord="fro")
delta_z_fro = torch.linalg.matrix_norm(delta_z, ord="fro")

# Linear出力誤差上界に必要な2種類のnormを計算する。
x_fro = torch.linalg.matrix_norm(x, ord="fro")
delta_w_spectral = torch.linalg.matrix_norm(delta_w, ord=2)
~~~

`torch.all`の返り値はshapeが空のscalar Tensorである。Pythonの`bool`として表示や分岐に使うときは、

~~~python
# Tensorの真偽値をPython boolへ変換する。
all_elements_hold_python = bool(all_elements_hold.item())
~~~

とする。

## 8. 数値許容誤差の意味

理論上は

$$
|\Delta H_{ij}|
\le
|\Delta Z_{ij}|
$$

である。コードで

~~~python
inequality_atol = 1e-12
elementwise_holds = (
    delta_h.abs()
    <= delta_z.abs() + inequality_atol
)
~~~

とする場合、`inequality_atol`は理論式へ追加された誤差項ではない。有限精度演算による丸め差で正しい判定が反転することを避ける、検査用の数値許容幅である。

## 9. 使い分けの結論

- 各要素の絶対誤差：`tensor.abs()`
- 全要素を1本のvectorとして測る：`torch.linalg.vector_norm`
- 行列のFrobenius norm：`torch.linalg.matrix_norm(..., ord="fro")`
- 行列のspectral norm：`torch.linalg.matrix_norm(..., ord=2)`
- 特異値列：`torch.linalg.svdvals`
- 数値rank：`torch.linalg.matrix_rank`
- 条件数：`torch.linalg.cond`

同じ数値が出る場合でも、教材コードでは数式上の対象に合わせてAPIを選ぶ。ReLUの実装検証は、[[08_TT_MPS基礎実装検証/15_ReLU_hidden_activation誤差伝播のPyTorch確認]]を参照する。
