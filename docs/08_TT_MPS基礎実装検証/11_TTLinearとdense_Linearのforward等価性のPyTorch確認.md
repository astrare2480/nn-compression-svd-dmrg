---
title: TTLinearとdense Linearのforward等価性のPyTorch確認
tags:
  - TT-matrix
  - TT-Linear
  - PyTorch
  - nn.Linear
  - equivalence
  - validation
---

# TTLinearとdense Linearのforward等価性のPyTorch確認

## このノートの位置づけ

このノートは、[Notebook 15](../../notebooks/30_tt_mps/00_fundamentals/15_ttlinear_dense_forward_equivalence_learning.ipynb) で行った、truncationなし2-core TT-SVDとdense `nn.Linear`のforward等価性検証を整理する。

理論上の等価性、転置、添字、縮約順序、biasの意味は [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/58_TTLinearとdense_Linearのforward等価性]] を参照する。TT-matrix一般の実装は、[[08_TT_MPS基礎実装検証/10_TT-matrix_Dense_Reconstruction_TT-Linear_ForwardのPyTorch確認]] と [[08_TT_MPS基礎実装検証/45_TT-matrix線形層のPyTorch直接forward]] に分けている。

ここで検証したのは、CPU・`float64`の小さいrandom `nn.Linear(6, 8)`である。学習済みFashion-MNIST MLPの層を置き換えた実験ではない。

## 1. 数値条件

保存済みNotebookでは次を使用した。

```text
PyTorch version: 2.11.0+cu128
device: CPU
default dtype: torch.float64
torch.manual_seed(0)

batch size B = 2
in_features n = 6
out_features m = 8

out_modes = (m1, m2) = (2, 4)
in_modes  = (n1, n2) = (2, 3)
```

したがって、

$$
m=m_1m_2=2\cdot4=8,
$$

$$
n=n_1n_2=2\cdot3=6.
$$

shapeは

```text
x             : (2, 6)
linear.weight : (8, 6)
linear.bias   : (8,)
y             : (2, 8)
```

である。

## 2. dense `nn.Linear`を三通りで比較する

`F`は`torch.nn.functional`の別名である。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

B = 2
n = 6
m = 8

# weight.shapeは(out_features, in_features)になる。
linear = nn.Linear(
    in_features=n,
    out_features=m,
    bias=True,
    dtype=torch.float64,
    device="cpu",
)

x_flat = torch.randn(B, n, dtype=torch.float64)

# Module、functional API、行列積を同じ入力で比較する。
y_module = linear(x_flat)
y_functional = F.linear(
    x_flat,
    linear.weight,
    linear.bias,
)
y_manual = x_flat @ linear.weight.T + linear.bias

torch.testing.assert_close(
    y_module,
    y_functional,
    rtol=1e-12,
    atol=1e-12,
)
torch.testing.assert_close(
    y_module,
    y_manual,
    rtol=1e-12,
    atol=1e-12,
)
```

保存済み結果は

```text
OK: nn.Linear == F.linear == x @ weight.T + bias
```

である。さらに、

```text
y_error (L2): 0.0
```

が保存されている。

## 3. dense forwardの一要素をloopで確認する

対象をbatchの2番目、出力の4番目に固定する。

```python
# Pythonは0始まりなので、beta=1は2番目、i_out=3は4番目を指す。
beta = 1
i_out = 3

# bias parameterを直接更新しないようにcloneする。
y_loop = linear.bias[i_out].clone()

# Y[beta, i] = b[i] + sum_j X[beta, j] W[i, j] を一項ずつ計算する。
for j_in in range(n):
    y_loop = (
        y_loop
        + x_flat[beta, j_in]
        * linear.weight[i_out, j_in]
    )

torch.testing.assert_close(
    y_loop,
    y_module[beta, i_out],
    rtol=1e-12,
    atol=1e-12,
)
```

保存済み結果は

```text
loop result:   1.363470070814456
module result: 1.363470070814456
abs error:     0.0
```

である。ここではTTをまだ使わず、TTLinearが再現すべきdense側の積和を固定している。

## 4. weightをinterleaved順へtensorizeする

元のweightは

$$
W:(m,n)=(8,6)
$$

である。まず出力modeをまとめたgrouped順へreshapeする。

$$
(8,6)
\longrightarrow
(m_1,m_2,n_1,n_2)
=(2,4,2,3).
$$

次に、同じsiteの出力・入力modeを隣接させる。

$$
(m_1,m_2,n_1,n_2)
\longrightarrow
(m_1,n_1,m_2,n_2).
$$

```python
m1, m2 = 2, 4
n1, n2 = 2, 3

# grouped order: (m1, m2, n1, n2)
weight_4d = linear.weight.reshape(
    m1,
    m2,
    n1,
    n2,
)

# interleaved order: (m1, n1, m2, n2)
weight_interleaved = weight_4d.permute(0, 2, 1, 3)

assert linear.weight.shape == (m, n)
assert weight_4d.shape == (m1, m2, n1, n2)
assert weight_interleaved.shape == (m1, n1, m2, n2)
```

`reshape`と`permute`の意味をshapeだけで判定せず、要素で確認する。

```python
i1_idx, i2_idx = 1, 2
j1_idx, j2_idx = 0, 1

# mode indexを元のdense row/columnへ戻す。
i_flat = i1_idx * m2 + i2_idx
j_flat = j1_idx * n2 + j2_idx

w_dense = linear.weight[i_flat, j_flat]
w_interleaved = weight_interleaved[
    i1_idx,
    j1_idx,
    i2_idx,
    j2_idx,
]

torch.testing.assert_close(w_dense, w_interleaved)
```

保存済み結果は

```text
W[i_flat, j_flat] = -0.015652710424751803
A[i1, j1, i2, j2] = -0.015652710424751803
```

である。

## 5. `divmod`でflat indexをmode indexへ戻す

重み一要素を検証するときは、flat indexからmode indexへの逆変換が必要になる。

```python
i_flat = 5
j_flat = 2

# i_flat = i1_idx * m2 + i2_idx の逆変換。
i1_idx, i2_idx = divmod(i_flat, m2)

# j_flat = j1_idx * n2 + j2_idx の逆変換。
j1_idx, j2_idx = divmod(j_flat, n2)

assert i_flat == i1_idx * m2 + i2_idx
assert j_flat == j1_idx * n2 + j2_idx
```

`divmod(a, b)`は `(a // b, a % b)`、すなわち商と余りを返す。この例では

$$
5=1\cdot4+1,
\qquad
2=0\cdot3+2
$$

なので、

$$
(i_1,i_2)=(1,1),
\qquad
(j_1,j_2)=(0,2)
$$

である。

## 6. truncationなし2-core TT-SVD

site 1のphysical sizeは

$$
q_1=m_1n_1=2\cdot2=4,
$$

site 2は

$$
q_2=m_2n_2=4\cdot3=12
$$

である。interleaved tensorを

$$
A\in\mathbb R^{4\times12}
$$

へreshapeし、thin SVDを行う。

```python
# site 1 | site 2のunfoldingを作る。
A = weight_interleaved.reshape(
    m1 * n1,
    m2 * n2,
)

# correctness確認なのでrank truncationは行わない。
U, S, Vh = torch.linalg.svd(
    A,
    full_matrices=False,
)
r1 = int(S.shape[0])

# G1へU、G2へSigma V^Tを入れるgaugeを使う。
core1 = U.reshape(1, m1, n1, r1)
core2 = (S[:, None] * Vh).reshape(
    r1,
    m2,
    n2,
    1,
)
```

保存済みshapeは

```text
A.shape:      (4, 12)
U.shape:      (4, 4)
S.shape:      (4,)
Vh.shape:     (4, 12)
r1:           4
core1.shape:  (1, 2, 2, 4)
core2.shape:  (4, 4, 3, 1)
```

である。$r_1=4$ は第1site側の最大次元4まで保持した結果であり、この段階ではrankを圧縮していない。

## 7. コアからdense weightを復元する

2コアを

```text
core1: a b c d = (r0, m1, n1, r1)
core2: d e f g = (r1, m2, n2, r2)
```

と読む。共有bond `d` を縮約し、境界rank `a`、`g`はsize 1なので出力へ残さない。

```python
# bond r1を縮約してinterleaved順(m1, n1, m2, n2)を作る。
weight_interleaved_reconstructed = torch.einsum(
    "abcd,defg->bcef",
    core1,
    core2,
)

# grouped順(m1, m2, n1, n2)へ戻す。
weight_grouped_reconstructed = (
    weight_interleaved_reconstructed.permute(0, 2, 1, 3)
)

# 元のdense weight shape(m, n)へ戻す。
weight_reconstructed = weight_grouped_reconstructed.reshape(m, n)

torch.testing.assert_close(
    weight_reconstructed,
    linear.weight,
    rtol=1e-12,
    atol=1e-12,
)
```

保存済み結果は

```text
weight_error (L2): 1.0062667213961117e-15
max_abs_error_weight: 4.996003610813204e-16
```

である。

## 8. 一要素をbond indexの和で復元する

`divmod`で得たmode indexを使い、重み一要素を明示的に計算する。

```python
# scalar tensorとして初期化し、dtype/deviceをコアへ合わせる。
weight_loop = torch.zeros(
    (),
    dtype=core1.dtype,
    device=core1.device,
)

# W[i,j] = sum_alpha G1[0,i1,j1,alpha] G2[alpha,i2,j2,0]
for alpha in range(r1):
    weight_loop = weight_loop + (
        core1[0, i1_idx, j1_idx, alpha]
        * core2[alpha, i2_idx, j2_idx, 0]
    )

torch.testing.assert_close(
    weight_loop,
    weight_reconstructed[i_flat, j_flat],
    rtol=1e-12,
    atol=1e-12,
)
```

この検証は、`einsum`全体の一致だけでなく、bond和の一項一項がどの重み要素を作るかを固定する。

## 9. dense重みを作らないdirect contraction

元の入力を上書きせず、flat入力とtensorized入力を分ける。

```python
# 外部interfaceの入力を保持する。
x_flat = x_flat.reshape(B, n)

# TTLinear内部だけで入力modeへ分ける。
x_tensor = x_flat.reshape(B, n1, n2)

# size 1の境界rank軸だけを落とす。
core1_no_boundary = core1[0]       # (m1, n1, r1)
core2_no_boundary = core2[..., 0]  # (r1, m2, n2)
```

まず $j_2$ を縮約する。

```python
# x_tensor:          a b c = (B, n1, n2)
# core2_no_boundary: d e c = (r1, m2, n2)
# tt2:               a b d e = (B, n1, r1, m2)
tt2 = torch.einsum(
    "abc,dec->abde",
    x_tensor,
    core2_no_boundary,
)
```

次に $j_1$ と $r_1$ を縮約する。

```python
# tt2:               a b c d = (B, n1, r1, m2)
# core1_no_boundary: e b c   = (m1, n1, r1)
# tt1:               a d e   = (B, m2, m1)
tt1 = torch.einsum(
    "abcd,ebc->ade",
    tt2,
    core1_no_boundary,
)
```

最後にoutput modeの順を元へ戻し、biasを加える。

```python
# dense weightのrow規約(i1, i2)へ戻す。
output_tensor = tt1.permute(0, 2, 1)  # (B, m1, m2)

# 通常のLinearと同じ(B, m)へ戻す。
output_flat = output_tensor.reshape(B, m)

# biasはTT化せず、batchへbroadcastして加える。
y_tt = output_flat + linear.bias
```

shape遷移は

$$
(B,n_1,n_2)
\longrightarrow
(B,n_1,r_1,m_2)
\longrightarrow
(B,m_2,m_1)
\longrightarrow
(B,m_1,m_2)
\longrightarrow
(B,m)
$$

である。

### `einsum`、`tensordot`、`permute`の関係

この2-core例では、出力labelまで明示できる`einsum`を使った。`torch.tensordot`で同じ縮約を行う場合、出力軸は原則として

$$
\text{左operandの未縮約軸}
+
\text{右operandの未縮約軸}
$$

の順に並ぶ。縮約した軸だけが両operandから消え、それ以外の軸サイズは変化しない。

`permute`は縮約値を正しくするために常に必要なわけではない。次回の`dims`で対象axisを明示すれば、`tensordot`直後の軸順のまま計算できる。一方、反復ごとに軸順を

```text
(batch, 未処理input modes, 現在bond, 処理済みoutput modes)
```

のような一定規約へ揃えると、同じ形の縮約を繰り返しやすい。最後には必ずoutput modesを

$$
(m_1,m_2,\ldots,m_d)
$$

へ揃えてからflattenする。

## 10. 三段階のcorrectness検証

検証を一つにまとめず、原因を切り分けられる順で行う。

```python
rtol = 1e-12
atol = 1e-12

# 1. PyTorch dense Linear内部の規約を確認する。
torch.testing.assert_close(
    y_module,
    y_functional,
    rtol=rtol,
    atol=atol,
)

# 2. TT-SVDとdense reconstructionを確認する。
torch.testing.assert_close(
    weight_reconstructed,
    linear.weight,
    rtol=rtol,
    atol=atol,
)

# 3. denseを作らないTT forwardを確認する。
torch.testing.assert_close(
    y_tt,
    y_module,
    rtol=rtol,
    atol=atol,
)
torch.testing.assert_close(
    y_tt,
    y_functional,
    rtol=rtol,
    atol=atol,
)
```

保存済み結果は次である。

| 検証対象 | 保存済み結果 |
| --- | ---: |
| `nn.Linear(x)` vs `F.linear(x, W, b)` | L2誤差 $0$ |
| reconstructed weight vs `linear.weight` | L2誤差 $1.0062667213961117\times10^{-15}$ |
| reconstructed weight vs `linear.weight` | 最大絶対誤差 $4.996003610813204\times10^{-16}$ |
| TT direct contraction vs dense forward | 最大絶対誤差 $1.1102230246251565\times10^{-15}$ |
| TT rank | $r_1=4$ |
| core shapes | $(1,2,2,4)$、$(4,4,3,1)$ |

これらは`float64`の丸め誤差の範囲で、

$$
W_{\mathrm{TT}}\approx W,
$$

$$
Y_{\mathrm{TT}}\approx XW^{\mathsf T}+b
$$

を確認した結果である。

## 11. parameter数はbiasと分けて数える

dense重みのparameter数は

$$
N_{\mathrm{dense,weight}}
=8\cdot6
=48.
$$

TTコアは

$$
\begin{aligned}
N_{\mathrm{TT,core}}
&=1\cdot2\cdot2\cdot4
+4\cdot4\cdot3\cdot1\\
&=16+48\\
&=64.
\end{aligned}
$$

biasは両者で共通に8要素なので、

$$
N_{\mathrm{dense,total}}=48+8=56,
$$

$$
N_{\mathrm{TT,total}}=64+8=72.
$$

この小例のtruncationなしTTはdenseより大きい。これはcorrectness検証として異常ではない。完全表現rankを保持しているためであり、圧縮評価にはrank truncationを伴う別実験が必要である。

## 12. 現在のNotebook 15に残る実装上の注意

現在のNotebookを静的に確認すると、等価性の中心計算は実装されている一方、教材・実行記録として次が残っている。

1. Tensorizationセルは `weight_interleaved.shape` を表示しているのに、labelが `linear.weight.shape` になっている。三つのshapeを別々に表示する方がよい。
2. direct contractionセルで `x = x.reshape(...)` と元の変数を上書きしている。`x_flat`と`x_tensor`へ分けると、外部interfaceと内部tensorizationを混同しない。
3. 一要素bond和の現在のソースには`assert_close`があるが、保存出力には後から加えた成功表示がなく、`execution_count`も`null`である。現在のソースを実行した証拠としては扱えない。
4. direct contractionセルも`execution_count`が`null`なのに保存出力がある。後段の実行番号は連続しておらず、カーネルを再起動したRun Allの証明ではない。
5. 現在のparameter表示はweight 48とcore 64だけで、biasを含むtotal 56と72を表示していない。
6. `dense_to_tt_matrix_tensor`のvalidationには重複があり、`assert condition, print(...)`も残る。`print`の戻り値は`None`なので、再利用関数では具体的な例外メッセージを持つvalidationへ直す必要がある。

学習用Notebookなので、これらの修正をdocs側から完成コードとして埋め戻す必要はない。Notebookでは、ユーザーがshape、要素対応、SVD、縮約式を自分で実装するTODOを残す。

## 13. 確認済みと未実施を分ける

この段階で確認済みなのは次である。

- `nn.Linear`、`F.linear`、明示行列積の一致
- dense forward一要素のloop展開
- `(m,n)`から`(m_1,n_1,m_2,n_2)`への要素対応
- truncationなし2-core TT-SVD
- TTコアからのdense weight復元
- bond indexを明示した重み一要素の復元
- dense重みを作らない2-core direct contraction
- dense forwardとTT forwardの保存済み数値一致

未実施または未証明なのは次である。

- 現在のNotebook全セルを新しいカーネルでRestart/Run Allしたこと
- rank truncation後の重み誤差と出力誤差
- backwardとgradientの一致
- optimizerを使ったTTコアのfine-tuning
- 既存の再利用可能な`TTLinear`クラスへコアを設定し、その`forward`を呼んだこと
- 学習済みFashion-MNIST MLPの対象層をTTLinearへ置換したこと
- モデル全体のlogits、loss、accuracyの比較
- parameter削減、速度、メモリの実測

したがって、今回の到達点は「小規模2-core TTLinearのforward correctness」であり、「Fashion-MNIST MLPのTT圧縮完了」ではない。

次の段階で行った学習済みFashion-MNIST MLPの`fc1`置換とlogits比較は、[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/00_TTLinear一層置換とlogits等価性]]を参照する。rank打ち切りによる一般的なLinear出力誤差は、[[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]]へ分けている。

## 14. 学習済みMLPへ接続するときの検証順

次段階では、次の順を崩さない。

```text
学習済みdense層を選ぶ
→ weight / bias / 入出力shapeを固定
→ mode分解を決める
→ truncationなしTT-SVDで層単体の等価性を確認
→ rankを制限して重み誤差と層出力誤差を測る
→ モデル全体のlogitsとaccuracyを測る
→ 必要ならfine-tuning
→ parameter数・速度・メモリを別々に評価
```

truncationなしの検証を先に通すことで、mode順序やdirect contractionのバグを、rank近似誤差と混同せずに済む。
