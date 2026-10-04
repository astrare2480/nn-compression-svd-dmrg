---
title: 第2Linearのlogits誤差伝播と合成上界のPyTorch確認
tags:
  - TTLinear
  - PyTorch
  - logits
  - spectral-norm
  - error-bound
  - validation
---

# 第2Linearのlogits誤差伝播と合成上界のPyTorch確認

## 1. このノートの責務

このノートは、[03_second_linear_logits_error_propagation.ipynb](../../notebooks/30_tt_mps/10_fashion_mnist_mlp/03_second_linear_logits_error_propagation.ipynb) で使うPyTorch操作、shape、独立検証の組み方、実装上の注意を整理する。

配置を次のように分ける。

- 一般式と途中式：[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]
- spectral normと最大特異値：[[00_基礎理論/01_数学基礎/01_線形代数/07_spectral_normと最大特異値_supとmax]]
- PyTorch API、shape、assert、実装上の注意：このノート
- 実験設定、保存済み数値、グラフの読み取り、結果の考察：[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/00_第2Linearのlogits誤差伝播_設定結果考察]]

## 2. 変数とshape

batch sizeを $B$、入力次元を $d$、hidden次元を $h$、output/class次元を $c$ とする。

| 変数 | 数式 | shape | 軸の意味 |
| --- | --- | ---: | --- |
| `x` | $X$ | `(B, d)` | batch、input feature |
| `w1`, `w1_tilde` | $W_1,\widetilde W_1$ | `(h, d)` | hidden unit、input feature |
| `delta_w1` | $\Delta W_1$ | `(h, d)` | hidden unit、input feature |
| `z1`, `z1_tilde` | $Z_1,\widetilde Z_1$ | `(B, h)` | batch、hidden unit |
| `h_dense`, `h_tilde` | $H,\widetilde H$ | `(B, h)` | batch、hidden unit |
| `w2` | $W_2$ | `(c, h)` | output/class、hidden unit |
| `logits`, `logits_tilde` | $L,\widetilde L$ | `(B, c)` | batch、output/class |

PyTorchのLinear weightは`(out_features, in_features)`なので、batch-firstの明示的行列積は

~~~python
# input: (B, in_features)
# weight: (out_features, in_features)
# output: (B, out_features)
output = input_batch @ weight.T + bias
~~~

と書く。

## 3. 誤差の符号を固定する

誤差はcompressed minus denseで統一する。

$$
\Delta W_1
=
\widetilde W_1-W_1,
$$

$$
\Delta Z_1
=
\widetilde Z_1-Z_1,
$$

$$
\Delta H
=
\widetilde H-H,
$$

$$
\Delta L
=
\widetilde L-L.
$$

toy設定で

$$
\widetilde W_1
=
W_1+\Delta W_1
$$

としていても、実際のTT近似へ差し替えたときの符号ミスを防ぐため、定義をコードで確認する。

~~~python
# compressed - dense の符号規約をコードでも固定する。
delta_w1_from_definition = w1_tilde - w1

assert_close(
    delta_w1,
    delta_w1_from_definition,
    rtol=rtol,
    atol=atol,
)
~~~

## 4. 第1 Linearの恒等式を独立に検証する

実forwardの差と理論式の右辺を別々に計算する。

~~~python
# dense側と近似側を実際にforwardする。
z1 = x @ w1.T + b1
z1_tilde = x @ w1_tilde.T + b1

# 実forwardの差からDelta_Z1を求める。
delta_z1 = z1_tilde - z1

# 恒等式の右辺は独立に計算する。
delta_z1_from_formula = x @ delta_w1.T

assert_close(
    delta_z1,
    delta_z1_from_formula,
    rtol=rtol,
    atol=atol,
)
~~~

`delta_z1`自体を`x @ delta_w1.T`から作ると、同じ式を同じ式と比較するだけになり、forward検証にならない。

## 5. ReLU後の誤差を計算する

~~~python
# ReLUをdense側と近似側へ個別に適用する。
h_dense = torch.relu(z1)
h_tilde = torch.relu(z1_tilde)

# compressed - denseでhidden activation誤差を作る。
delta_h = h_tilde - h_dense

# ReLUの1-Lipschitz性を要素ごとに確認する。
assert torch.all(
    delta_h.abs()
    <= delta_z1.abs() + inequality_atol
)
~~~

要素不等式

$$
|\Delta H_{n,j}|
\le
|\Delta Z_{1,n,j}|
$$

と、Frobenius normの不等式

$$
\|\Delta H\|_F
\le
\|\Delta Z_1\|_F
$$

を区別する。

## 6. 第2 Linearの恒等式を独立に検証する

dense側と近似側で、同じ $W_2,b_2$ を用いる。

~~~python
# dense側のlogitsを実forwardから計算する。
logits = h_dense @ w2.T + b2

# 近似側も同じw2とb2を使用する。
logits_tilde = h_tilde @ w2.T + b2

# compressed - denseで実測logits誤差を作る。
delta_l = logits_tilde - logits

# 理論恒等式の右辺は別計算する。
delta_l_from_formula = delta_h @ w2.T

assert logits.shape == (B, c)
assert logits_tilde.shape == (B, c)
assert delta_l.shape == (B, c)
assert delta_l_from_formula.shape == (B, c)

assert_close(
    delta_l,
    delta_l_from_formula,
    rtol=rtol,
    atol=atol,
)
~~~

この検証は、

$$
\Delta L
=
\Delta H W_2^{\mathsf T}
$$

の数値照合である。

## 7. spectral normと最大特異値を計算する

~~~python
# ord=2は行列のspectral normを返す。
w2_spectral = torch.linalg.matrix_norm(
    w2,
    ord=2,
)

# 特異値列の長さはmin(c, h)になる。
w2_singular_values = torch.linalg.svdvals(w2)

assert w2_singular_values.shape == (min(c, h),)
assert_close(
    w2_spectral,
    w2_singular_values.max(),
    rtol=rtol,
    atol=atol,
)
~~~

変数名は`w2_fro`ではなく`w2_spectral`とする。この値は

$$
\|W_2\|_2
=
\sigma_{\max}(W_2)
$$

であり、Frobenius normではない。

## 8. 第2 Linearの上界を確認する

実測誤差は理論式から作り直さず、実forward差`delta_l`から測る。

~~~python
# 実forwardで得たlogits差のFrobenius normを測る。
delta_l_fro = torch.linalg.matrix_norm(
    delta_l,
    ord="fro",
)

# hidden activation誤差のFrobenius normを測る。
delta_h_fro = torch.linalg.matrix_norm(
    delta_h,
    ord="fro",
)

# 第2 Linearのworst-case上界を計算する。
logits_upper_bound = delta_h_fro * w2_spectral

assert delta_l_fro <= (
    logits_upper_bound + inequality_atol
)
~~~

## 9. 第1 Linearからの合成上界を確認する

~~~python
# 行列XのFrobenius normを計算する。
x_fro = torch.linalg.matrix_norm(
    x,
    ord="fro",
)

# 第1 Linearの重み誤差はspectral normで測る。
delta_w1_spectral = torch.linalg.matrix_norm(
    delta_w1,
    ord=2,
)

# 段階ごとの上界を作る。
z1_upper_bound = x_fro * delta_w1_spectral
full_upper_bound = z1_upper_bound * w2_spectral

assert delta_z1_fro <= (
    z1_upper_bound + inequality_atol
)
assert delta_h_fro <= (
    delta_z1_fro + inequality_atol
)
assert delta_l_fro <= (
    logits_upper_bound + inequality_atol
)
assert delta_l_fro <= (
    full_upper_bound + inequality_atol
)
~~~

確認対象は、

$$
\|\Delta L\|_F
\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
$$

である。

## 10. `vector_norm`と`matrix_norm`

2階tensor $A$ に対して、

$$
\|\operatorname{vec}(A)\|_2
=
\|A\|_F
$$

なので、`torch.linalg.vector_norm(A)`と`torch.linalg.matrix_norm(A, ord="fro")`は同じ数値になる。

それでも、行列のFrobenius normを実装していることを明示するには、

~~~python
# 数式のFrobenius normとAPIを対応させる。
matrix_fro = torch.linalg.matrix_norm(
    matrix,
    ord="fro",
)
~~~

と書く方が読みやすい。

~~~python
# これはFrobenius normではなくspectral normである。
matrix_spectral = torch.linalg.matrix_norm(
    matrix,
    ord=2,
)
~~~

詳細は、[[08_TT_MPS基礎実装検証/30_PyTorchのvector_norm_matrix_normとtorch_linalg]]を参照する。

## 11. scale sweepを実装するときの注意

### 11.1 比較ごとに誤差方向を作り直さない

~~~python
# 誤差方向はloopの外で1回だけ生成する。
perturbation_direction = torch.randn(
    h,
    d,
    dtype=dtype,
    device=device,
)

for scale in scales:
    # scaleだけを変えて、公平に比較する。
    delta_w1_scale = scale * perturbation_direction
~~~

loop内でrandom方向を再生成すると、scaleの効果と方向の効果を分離できない。

### 11.2 scale 0の比率を0にしない

scale 0では実測値と上界がともに0なので、比率は $0/0$ となり未定義である。

~~~python
import math


if full_bound_scale.item() == 0.0:
    # 0 / 0は0ではなく未定義なのでNaNを記録する。
    actual_bound_ratio_scale = float("nan")
else:
    actual_bound_ratio_scale = (
        delta_l_scale_fro / full_bound_scale
    ).item()

# NaNでないscaleだけ上界比を検査する。
if not math.isnan(actual_bound_ratio_scale):
    assert actual_bound_ratio_scale <= (
        1.0 + inequality_atol
    )
~~~

### 11.3 ReLU maskの変化を測る

scaleを大きくしたときに線形な比例関係が崩れた場合、pre-activationが0を横切った可能性を調べる。

~~~python
# dense側と近似側でReLUの通過状態が異なる要素を数える。
mask_changes = torch.count_nonzero(
    (z1_tilde_scale > 0)
    != (z1 > 0)
).item()
~~~

## 12. よくある実装ミス

### 12.1 符号を逆にする

$$
\Delta W_1
=
\widetilde W_1-W_1
$$

で統一する。$W_1-\widetilde W_1$へ逆転させると、恒等式の符号も全て逆になる。

### 12.2 転置を忘れる

~~~python
# 正しいbatch-first Linear。
logits = hidden @ w2.T + b2
~~~

偶然shapeが正方形だと転置漏れを検出できない場合があるため、toy例では $d,h,c$ を異なる値にする。

### 12.3 第2 Linearまで変える

$$
\Delta L
=
\Delta H W_2^{\mathsf T}
$$

は、dense側と近似側で同じ $W_2,b_2$ を使う場合の式である。第2 Linearも圧縮する場合は、重み誤差と交差項を含む別の式になる。

### 12.4 actual errorを理論式から作る

~~~python
# actual errorは実forwardの差から作る。
delta_l = logits_tilde - logits

# 理論式の右辺は別の変数として計算する。
delta_l_from_formula = delta_h @ w2.T
~~~

この2つを分けないと、独立な検証にならない。

## 13. OpenMP回避設定

現行Notebookは`torch`のimport前に、

~~~python
import os

# OpenMP runtime重複時の限定的な回避策である。
# 数学や一般的なPyTorch実装の必須設定ではない。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
~~~

としている。

これはOpenMP runtime重複エラーの一時的な回避策であり、恒久解決ではない。エラーがない環境では設定しない。すでに`torch`をimportした後に設定を変更しても、最初のruntime読込みには間に合わないため、必要な場合はkernelを再起動してimport前に設定する。

## 14. 検証の最小セット

小さいtoy例で一度確認するものは次である。

1. $\Delta W_1=\widetilde W_1-W_1$。
2. $\Delta Z_1=X\Delta W_1^{\mathsf T}$。
3. $|\Delta H_{n,j}|\le|\Delta Z_{1,n,j}|$。
4. $\Delta L=\Delta H W_2^{\mathsf T}$。
5. $\|W_2\|_2=\sigma_{\max}(W_2)$。
6. 第2 Linear上界と合成上界。

実際のTT-rank sweepでは、各rankについて少なくとも合成上界をassertし、実測値、上界、比率を記録する。spectral normと最大特異値のAPI一致は、学習のために一度確認すればよい。

## 15. 現行Notebookの静的監査

現行Notebookでは`TODO 1`から`TODO 6`までの実装欄にコードが保存されている。今回の教材整理でNotebookへ解答を追加したわけではない。

主なcode cellにはexecution countが残っているが、scale 0の比率について現行ソースと保存出力が一致しない。このため、execution countだけを根拠に現行ソースのFresh Run All済みとは判定しない。

保存済み数値とその考察は、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/00_第2Linearのlogits誤差伝播_設定結果考察]]を参照する。

