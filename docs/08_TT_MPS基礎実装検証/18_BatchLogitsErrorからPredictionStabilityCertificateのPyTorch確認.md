---
title: Batch Logits ErrorからPrediction Stability CertificateのPyTorch確認
tags:
  - PyTorch
  - TTLinear
  - logits
  - norm
  - classification-margin
  - certificate
---

# Batch Logits ErrorからPrediction Stability CertificateのPyTorch確認

## 1. このノートの役割

2層MLPの第1 Linearへ固定した重み摂動を与え、次のchainをPyTorchで計算する。

$$
\|\delta l_n\|_\infty
\le
\|\delta l_n\|_2
\le
\|\Delta L\|_F
\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2.
$$

さらに、

$$
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
<
\frac{\gamma_n}{2}
$$

を満たすとき、sample $n$ のpredictionが変わらないことを確認する。

一般式と途中式は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/66_BatchLogitsErrorからSample-wisePredictionStabilityへ]]を参照する。このノートでは、PyTorchのshape、行抽出、norm API、scalar boolean、assertに焦点を置く。

## 2. 変数とshape

| 変数 | 意味 | shape |
| --- | --- | ---: |
| `X` | batch入力 | `(B, d)` |
| `w1` | dense第1 Linear重み | `(h, d)` |
| `delta_w1` | 第1 Linearの重み誤差 | `(h, d)` |
| `w1_tilde` | 近似第1 Linear重み | `(h, d)` |
| `w2` | 第2 Linear重み | `(c, h)` |
| `logits` | dense batch logits $L$ | `(B, c)` |
| `logits_tilde` | compressed batch logits $\widetilde L$ | `(B, c)` |
| `delta_logits` | batch logits error $\Delta L$ | `(B, c)` |
| `delta_l_n` | sample $n$ のlogits error | `(c,)` |
| 各norm・margin | scalar Tensor | `()` |

PyTorchの`(c,)` Tensorには、行・列の向きがない。数学では、

$$
\Delta L
=
\begin{pmatrix}
\delta l_1^T\\
\vdots\\
\delta l_B^T
\end{pmatrix}
$$

と書くが、コードでは、

~~~python
delta_l_n = delta_logits[sample_index]
~~~

で同じ $c$ 個の成分を取り出せる。

## 3. 実行条件と固定toy行列

~~~python
import torch
from torch.testing import assert_close

# 小さい誤差を安定して比較するためfloat64を使う。
torch.manual_seed(0)
dtype = torch.float64
device = torch.device("cpu")
rtol = 1e-10
atol = 1e-12
eps = 1e-12

B, d, h, c = 3, 2, 2, 3

# 各sampleを行に並べたbatch入力。
X = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=dtype,
    device=device,
)

# Dense modelの第1 Linear。
w1 = torch.tensor(
    [
        [2.0, 0.0],
        [0.0, 2.0],
    ],
    dtype=dtype,
    device=device,
)
b1 = torch.tensor([1.0, 1.0], dtype=dtype, device=device)

# TT-rank打ち切りそのものではなく、近似誤差を模した固定摂動。
delta_w1 = torch.tensor(
    [
        [0.010, -0.005],
        [0.004, 0.008],
    ],
    dtype=dtype,
    device=device,
)
w1_tilde = w1 + delta_w1

# Dense modelと近似modelで共有する第2 Linear。
w2 = torch.tensor(
    [
        [2.0, 0.0],
        [0.0, 1.5],
        [-1.0, -1.0],
    ],
    dtype=dtype,
    device=device,
)
b2 = torch.zeros(c, dtype=dtype, device=device)

assert X.shape == (B, d)
assert w1.shape == delta_w1.shape == w1_tilde.shape == (h, d)
assert b1.shape == (h,)
assert w2.shape == (c, h)
assert b2.shape == (c,)
~~~

この設定は、norm chainとcertificateを手で追えるようにしたtoy例である。`delta_w1`は実際のTT-SVDまたはTT-roundingから得た値ではない。

## 4. Denseと近似modelのforward

PyTorchの`torch.nn.functional.linear`は、

$$
Y
=
XW^T+b
$$

を計算する。biasはbatch軸へbroadcastされる。

~~~python
import torch.nn.functional as F

# Dense model。
z1 = F.linear(X, w1, b1)
hidden = torch.relu(z1)
logits = F.linear(hidden, w2, b2)

# 第1 Linearだけを近似したmodel。
z1_tilde = F.linear(X, w1_tilde, b1)
hidden_tilde = torch.relu(z1_tilde)
logits_tilde = F.linear(hidden_tilde, w2, b2)

delta_z1 = z1_tilde - z1
delta_hidden = hidden_tilde - hidden
delta_logits = logits_tilde - logits

assert z1.shape == z1_tilde.shape == (B, h)
assert hidden.shape == hidden_tilde.shape == (B, h)
assert logits.shape == logits_tilde.shape == (B, c)
assert delta_logits.shape == (B, c)
~~~

第1 Linearのbiasは共通なので、

$$
\Delta Z_1
=
X\Delta W_1^T
$$

が成立する。また、第2 Linearのbiasも共通なので、

$$
\Delta L
=
\Delta H W_2^T
$$

が成立する。

~~~python
# 恒等式を、最終boundとは独立に確認する。
assert_close(delta_z1, X @ delta_w1.T, rtol=rtol, atol=atol)
assert_close(delta_logits, delta_hidden @ w2.T, rtol=rtol, atol=atol)
~~~

## 5. Sample行の抽出

0始まりの`sample_index = 2`は、第3 sampleを選ぶ。

~~~python
sample_index = 2

l_n = logits[sample_index]
l_tilde_n = logits_tilde[sample_index]
delta_l_n = delta_logits[sample_index]

assert l_n.shape == l_tilde_n.shape == delta_l_n.shape == (c,)
assert_close(delta_l_n, l_tilde_n - l_n, rtol=rtol, atol=atol)
~~~

数学では第3行が $\delta l_3^T$ である。PyTorchでは、

~~~python
assert_close(delta_l_n, delta_logits[2], rtol=rtol, atol=atol)
~~~

で十分である。`delta_l_n.T`としても1階Tensorのshapeは変わらない。行列の向きを必要とするときだけ、`unsqueeze(0)`または`unsqueeze(1)`で2階Tensorへ変換する。

~~~python
delta_row = delta_l_n.unsqueeze(0)       # shape: (1, c)
delta_column = delta_l_n.unsqueeze(1)    # shape: (c, 1)

assert delta_row.shape == (1, c)
assert delta_column.shape == (c, 1)
~~~

## 6. Norm APIを使い分ける

### 6.1 Sample vectorのinfinity norm

$$
\|\delta l_n\|_\infty
=
\max_k|\delta l_{nk}|
$$

は、

~~~python
sample_inf = torch.linalg.vector_norm(delta_l_n, ord=float("inf"))
sample_inf_direct = torch.max(torch.abs(delta_l_n))

assert sample_inf.ndim == 0
assert_close(sample_inf, sample_inf_direct, rtol=rtol, atol=atol)
~~~

で計算できる。

### 6.2 Sample vectorの2-norm

$$
\|\delta l_n\|_2
=
\left(
\sum_k|\delta l_{nk}|^2
\right)^{1/2}
$$

は、

~~~python
sample_l2 = torch.linalg.vector_norm(delta_l_n, ord=2)
sample_l2_direct = torch.sqrt(torch.sum(delta_l_n**2))

assert sample_l2.ndim == 0
assert_close(sample_l2, sample_l2_direct, rtol=rtol, atol=atol)
~~~

で計算できる。

### 6.3 Batch matrixのFrobenius norm

$$
\|\Delta L\|_F
=
\left(
\sum_{i,k}|\Delta L_{ik}|^2
\right)^{1/2}
$$

は、

~~~python
batch_fro = torch.linalg.matrix_norm(delta_logits, ord="fro")
batch_fro_vectorized = torch.linalg.vector_norm(delta_logits.reshape(-1), ord=2)

assert batch_fro.ndim == 0
assert_close(batch_fro, batch_fro_vectorized, rtol=rtol, atol=atol)
~~~

で計算できる。`reshape(-1)`は値を1列に並べ替えているだけで、

$$
\|\Delta L\|_F
=
\|\operatorname{vec}(\Delta L)\|_2
$$

をコードにしたものである。

### 6.4 重み行列のspectral norm

$$
\|\Delta W_1\|_2
=
\sigma_{\max}(\Delta W_1),
\qquad
\|W_2\|_2
=
\sigma_{\max}(W_2)
$$

は、

~~~python
delta_w1_spectral = torch.linalg.matrix_norm(delta_w1, ord=2)
w2_spectral = torch.linalg.matrix_norm(w2, ord=2)

# 最大特異値との一致も確認する。
assert_close(
    delta_w1_spectral,
    torch.linalg.svdvals(delta_w1)[0],
    rtol=rtol,
    atol=atol,
)
assert_close(
    w2_spectral,
    torch.linalg.svdvals(w2)[0],
    rtol=rtol,
    atol=atol,
)
~~~

で計算できる。`torch.linalg.vector_norm(matrix)`は行列を全要素vectorのように扱うため、spectral normの代わりにはならない。

## 7. 理論上界と不等式chain

入力にはFrobenius norm、重み行列にはspectral normを使う。

~~~python
x_fro = torch.linalg.matrix_norm(X, ord="fro")
full_upper = x_fro * delta_w1_spectral * w2_spectral

assert x_fro.ndim == 0
assert full_upper.ndim == 0
~~~

不等式chainは、scalar boolean Tensorを`&`で結合できる。

~~~python
chain_holds = (
    (sample_inf <= sample_l2 + eps)
    & (sample_l2 <= batch_fro + eps)
    & (batch_fro <= full_upper + eps)
)

assert chain_holds.ndim == 0
assert bool(chain_holds)
~~~

`eps`は理論上の不等式を緩めるための定数ではなく、浮動小数点丸めによるごく小さい比較差を吸収する検証用toleranceである。理論式は、

$$
\|\delta l_n\|_\infty
\le
\|\delta l_n\|_2
\le
\|\Delta L\|_F
\le
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
$$

のままである。

表示は各量の意味が分かるように分ける。

~~~python
print(
    f"{sample_inf.item():.8f} <= "
    f"{sample_l2.item():.8f} <= "
    f"{batch_fro.item():.8f} <= "
    f"{full_upper.item():.8f}"
)
~~~

## 8. Classification marginの計算

dense sample logitsを、

$$
l_n
=
\begin{pmatrix}
(l_n)_0\\
\vdots\\
(l_n)_{c-1}
\end{pmatrix}
$$

とする。top classは、

~~~python
y_star = torch.argmax(l_n)
top_logit = l_n[y_star]

assert y_star.ndim == 0
assert top_logit.ndim == 0
~~~

で得る。strongest competitorを求めるときは、元の`l_n`をin-place変更しないように`clone()`する。

~~~python
competitor_logits = l_n.clone()
competitor_logits[y_star] = -torch.inf

competitor_index = torch.argmax(competitor_logits)
competitor_logit = competitor_logits[competitor_index]
gamma_n = top_logit - competitor_logit

assert competitor_index.ndim == 0
assert gamma_n.ndim == 0
assert gamma_n > 0
~~~

`y_star`はclass indexであり、`top_logit`は実数scoreである。両者を混同しない。

## 9. Certificateと実際のpredictionを分けて判定する

最も上流の十分条件は、

$$
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
<
\frac{\gamma_n}{2}
$$

である。

~~~python
margin_threshold = gamma_n / 2
certificate = full_upper < margin_threshold

y_tilde = torch.argmax(l_tilde_n)
prediction_stable = y_tilde == y_star

assert certificate.ndim == 0
assert prediction_stable.ndim == 0

# このtoy設定ではcertificateが成立する。
assert bool(certificate)

# Certificateから期待される実測prediction一致も独立に確認する。
assert bool(prediction_stable)
~~~

`certificate`と`prediction_stable`は別の判定である。

- `certificate`はnorm上界だけで成立する保証。
- `prediction_stable`はこの具体的なforward結果に対する実測。

理論は、

$$
\text{certificate=True}
\Longrightarrow
\text{prediction\_stable=True}
$$

を与える。逆向きは一般には成り立たない。

~~~python
print(f"batch Frobenius error = {batch_fro.item():.8f}")
print(f"full theoretical upper = {full_upper.item():.8f}")
print(f"gamma_n / 2           = {margin_threshold.item():.8f}")
print(f"certificate           = {bool(certificate)}")
print(f"dense prediction      = {int(y_star)}")
print(f"compressed prediction = {int(y_tilde)}")
~~~

## 10. 一つの関数へまとめる場合

学習段階では各量を個別に計算して意味を確認する。再利用時には、判定を次のようにまとめられる。

~~~python
def sample_stability_certificate(
    logits: torch.Tensor,
    logits_tilde: torch.Tensor,
    full_upper: torch.Tensor,
    sample_index: int,
) -> dict[str, torch.Tensor | bool]:
    """上流のlogits誤差上界から1 sampleの予測安定性を判定する。"""
    if logits.ndim != 2 or logits_tilde.ndim != 2:
        raise ValueError("logits and logits_tilde must have shape (B, c)")
    if logits.shape != logits_tilde.shape:
        raise ValueError("logits and logits_tilde must have the same shape")
    if logits.shape[1] < 2:
        raise ValueError("at least two classes are required to define a margin")
    if not 0 <= sample_index < logits.shape[0]:
        raise IndexError("sample_index is outside the batch")
    if full_upper.ndim != 0:
        raise ValueError("full_upper must be a scalar Tensor")
    if full_upper < 0:
        raise ValueError("full_upper must be nonnegative")

    l_n = logits[sample_index]
    l_tilde_n = logits_tilde[sample_index]

    y_star = torch.argmax(l_n)
    competitor_logits = l_n.clone()
    competitor_logits[y_star] = -torch.inf
    gamma_n = l_n[y_star] - torch.max(competitor_logits)

    certificate = full_upper < gamma_n / 2
    prediction_stable = torch.argmax(l_tilde_n) == y_star

    return {
        "y_star": y_star,
        "gamma_n": gamma_n,
        "threshold": gamma_n / 2,
        "certificate": bool(certificate),
        "prediction_stable": bool(prediction_stable),
    }
~~~

この関数へ渡す`full_upper`は、実測した$\|\Delta L\|_F$でも、さらに上流の$\|X\|_F\|\Delta W_1\|_2\|W_2\|_2$でもよい。ただし、どちらを渡したかを結果記録で明示する。

## 11. 実装上の注意

### 11.1 `&`と`and`

0次元boolean Tensor同士は`&`で要素単位に結合できる。Pythonの`and`を使う場合は、それぞれを`bool(...)`へ変換する方が意図が明確である。

~~~python
chain_holds_python = (
    bool(sample_inf <= sample_l2 + eps)
    and bool(sample_l2 <= batch_fro + eps)
    and bool(batch_fro <= full_upper + eps)
)
assert chain_holds_python
~~~

### 11.2 `clone()`の理由

~~~python
competitor_logits = l_n.clone()
competitor_logits[y_star] = -torch.inf
~~~

とする。`competitor_logits = l_n`だけでは同じstorageを参照するため、top classの除外操作が元のlogitsを破壊し得る。

### 11.3 環境回避設定を数学と混ぜない

環境によってはOpenMP runtimeの重複を避ける設定が必要になる場合がある。しかし、それはローカル実行環境の回避策であり、norm chainやcertificateの数学的条件ではない。必要性を確認せず、一般的な教材コードの理論要件として扱わない。

### 11.4 TT圧縮結果ではない

このtoy例の`delta_w1`は手入力である。従って、コードが通っても、特定のTT-rank、TT-SVD、TT-roundingの性能を検証したことにはならない。

## 12. 現行Notebookとの対応と確認範囲

対応する学習Notebookは、[05_logits_bound_to_margin_certificate.ipynb](../../notebooks/30_tt_mps/10_fashion_mnist_mlp/05_logits_bound_to_margin_certificate.ipynb)である。

現行ファイルでは、上記toy値に対する保存済み出力とsourceを確認できる。一方、code cellの`execution_count`には欠番と`null`が混在している。そのため、次を区別する。

- 現行source、保存済み出力、assertの記載は確認した。
- 教材化時にNotebookを新しいkernelでRestart Kernel、Run Allしたわけではない。
- 保存済み出力が、現行sourceを先頭から最後まで同一実行で生成したことまでは証明しない。

固定toy行列の全要素、保存済み数値、結果の解釈は、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/02_BatchLogitsErrorからPredictionStabilityCertificate_設定結果考察]]を参照する。
