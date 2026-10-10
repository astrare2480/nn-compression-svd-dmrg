---
title: Minimum MarginとBatch-wide Prediction StabilityのPyTorch確認
tags:
  - PyTorch
  - TTLinear
  - logits
  - minimum-margin
  - batch-wide-certificate
---

# Minimum MarginとBatch-wide Prediction StabilityのPyTorch確認

## 1. このノートの役割

Dense logitsとcompressed logitsのbatchから、次をPyTorchで計算する。

$$
y^*,
\qquad
\gamma,
\qquad
\gamma_{\min},
\qquad
n_{\mathrm{bottleneck}},
\qquad
\widetilde y.
$$

さらに、共通上界、

$$
U
=
\|X\|_F
\|\Delta W_1\|_2
\|W_2\|_2
$$

について、

$$
U
<
\frac{\gamma_{\min}}{2}
$$

を判定し、全sampleでpredictionが一致することを実測でも確認する。

一般式と途中式は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/67_Batch内全予測不変の十分条件]]を参照する。

## 2. 変数とshape

| Python | 数式 | shape | 軸・意味 |
| --- | --- | ---: | --- |
| `logits` | $L$ | `(B, c)` | axis 0がsample、axis 1がclass |
| `logits_tilde` | $\widetilde L$ | `(B, c)` | axis 0がsample、axis 1がclass |
| `y_star` | $y^*$ | `(B,)` | dense predicted class index |
| `top2_values` | 各sampleの上位2 logit | `(B, 2)` | axis 1の0列が1位、1列が2位 |
| `top2_indices` | 上位2 logitのclass index | `(B, 2)` | 値が元のclass index |
| `gamma` | $\gamma$ | `(B,)` | sample-wise margin |
| `gamma_min` | $\gamma_{\min}$ | `()` | batch内の最小margin |
| `bottleneck_index` | $n_{\mathrm{bottleneck}}$ | `()` | minimum marginを持つsample index |
| `full_upper_bound` | $U$ | `()` | 全sample共通のlogits error上界 |
| `batch_certificate` | $U<\gamma_{\min}/2$ | `()` | batch-wide十分条件の真偽値 |
| `sample_certificates` | $[U<\gamma_n/2]_n$ | `(B,)` | sample-wise条件の真偽値 |
| `y_tilde` | $\widetilde y$ | `(B,)` | compressed predicted class index |
| `all_predictions_match` | $\widetilde y=y^*$ | `()` | prediction vector全体の一致 |

## 3. 固定toy MLPを計算する

前段と同じ固定値を使う。

~~~python
import torch
from torch.testing import assert_close

torch.manual_seed(0)

dtype = torch.float64
device = torch.device("cpu")
rtol = 1e-10
atol = 1e-12
eps = 1e-12

B, d, h, c = 3, 2, 2, 3

# 行がsample、列が入力feature。
x = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=dtype,
    device=device,
)

w1 = torch.tensor(
    [
        [2.0, 0.0],
        [0.0, 2.0],
    ],
    dtype=dtype,
    device=device,
)
b1 = torch.tensor([1.0, 1.0], dtype=dtype, device=device)

# 実TT分解ではなく、重み近似誤差を模した固定値。
delta_w1 = torch.tensor(
    [
        [0.010, -0.005],
        [0.004, 0.008],
    ],
    dtype=dtype,
    device=device,
)
w1_tilde = w1 + delta_w1

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

z1 = x @ w1.T + b1
h_dense = torch.relu(z1)
logits = h_dense @ w2.T + b2

z1_tilde = x @ w1_tilde.T + b1
h_tilde = torch.relu(z1_tilde)
logits_tilde = h_tilde @ w2.T + b2

delta_l = logits_tilde - logits

x_fro = torch.linalg.matrix_norm(x, ord="fro")
delta_w1_spectral = torch.linalg.matrix_norm(delta_w1, ord=2)
w2_spectral = torch.linalg.matrix_norm(w2, ord=2)

full_upper_bound = x_fro * delta_w1_spectral * w2_spectral
delta_l_fro = torch.linalg.matrix_norm(delta_l, ord="fro")

assert logits.shape == logits_tilde.shape == delta_l.shape == (B, c)
assert full_upper_bound.shape == torch.Size([])
assert delta_l_fro <= full_upper_bound + eps
~~~

## 4. `.detach().cpu().numpy().round(4)`の意味

確認用に、

~~~python
print(logits.detach().cpu().numpy().round(4))
~~~

と書ける。処理は左から順に次の意味を持つ。

- `logits`：shape `(B, c)`のdense logits。
- `.detach()`：autogradの計算graphから切り離す。
- `.cpu()`：CPU Tensorへ移す。既にCPUなら値は変わらない。
- `.numpy()`：NumPy配列へ変換する。
- `.round(4)`：表示する値を小数点以下4桁へ丸める。
- `print(...)`：配列を表示する。

`round(4)`は元の`logits`を変更しない。また、固定小数点形式を強制しないため、`6.0000`ではなく`6.`と表示されることがある。

## 5. `torch.argmax(logits, dim=1)`

### 5.1 `dim=1`の意味

`logits`のshapeは`(B, c)`である。

~~~text
                    class軸: dim=1
                  class 0  class 1  class 2
sample 0             6.0      1.5     -4.0
sample 1             2.0      4.5     -4.0
sample 2             6.0      4.5     -6.0
   ↑
sample軸: dim=0
~~~

各sampleを固定してclass方向の最大値の位置を求めるので、`dim=1`を指定する。

~~~python
# 最大logitの値ではなく、class indexを返す。
y_star = torch.argmax(logits, dim=1)

assert y_star.shape == (B,)
assert y_star.dtype == torch.int64
~~~

結果は、

~~~text
tensor([0, 1, 0])
~~~

である。

`argmax(logits, dim=1)`だけで書くには、`from torch import argmax`が必要である。どのlibraryの関数かを明確にするため、教材では`torch.argmax`と書く方が分かりやすい。

## 6. `torch.topk`の値とindex

`torch.topk`は上位値と元のindexを両方返す。

~~~python
# 各sampleのclass軸から、大きい順に上位2個を取得する。
top2 = torch.topk(logits, k=2, dim=1)
top2_values = top2.values
top2_indices = top2.indices

assert top2_values.shape == (B, 2)
assert top2_indices.shape == (B, 2)
~~~

値は、

~~~text
tensor([[6.0000, 1.5000],
        [4.5000, 2.0000],
        [6.0000, 4.5000]], dtype=torch.float64)
~~~

であり、class indexは、

~~~text
tensor([[0, 1],
        [1, 0],
        [0, 1]])
~~~

である。

`top2_values[:, 0]`は1位のlogit値、`top2_values[:, 1]`は2位のlogit値である。一方、`top2_indices`の要素はclass番号である。

## 7. Margin vectorを計算する

~~~python
# 各sampleについて、1位と2位のlogit差を取る。
gamma = top2_values[:, 0] - top2_values[:, 1]

assert gamma.shape == (B,)
assert torch.all(gamma > 0.0)
~~~

結果は、

$$
\gamma
=
\begin{pmatrix}
4.5\\
2.5\\
1.5
\end{pmatrix}.
$$

`argmax`が返すのはclass index、`topk(...).values`が返すのはlogit値である。marginの計算には値を使う。

## 8. Minimumの値とindexを同時に取得する

### 8.1 推奨する書き方

~~~python
# dim=0はgammaのsample軸。
gamma_min, bottleneck_index = torch.min(gamma, dim=0)

assert gamma_min.shape == torch.Size([])
assert bottleneck_index.shape == torch.Size([])
assert bottleneck_index.dtype == torch.int64
assert_close(gamma_min, gamma[bottleneck_index], rtol=rtol, atol=atol)
~~~

`torch.min(gamma, dim=0)`は、最小値とそのindexをこの順序で返す。左辺の2変数へ分けて受け取る処理はPythonのアンパックである。

同じ処理を明示的に書くと、

~~~python
minimum_result = torch.min(gamma, dim=0)
gamma_min = minimum_result.values
bottleneck_index = minimum_result.indices
~~~

である。

### 8.2 別々に求める書き方

~~~python
gamma_min_separate = torch.min(gamma)
bottleneck_index_separate = torch.argmin(gamma)

assert_close(gamma_min_separate, gamma_min, rtol=rtol, atol=atol)
assert torch.equal(bottleneck_index_separate, bottleneck_index)
~~~

`dim`を指定しない`torch.min(gamma)`は最小値だけを返す。indexも必要なら`torch.argmin`を別に呼ぶか、`dim=0`を指定した形式を使う。

この例では、

~~~text
gamma_min        = tensor(1.5000, dtype=torch.float64)
bottleneck_index = tensor(2)
~~~

である。indexは0始まりなので、`2`は第3 sampleを指す。

## 9. Minimum thresholdが全sample以下であること

vector比較、

~~~python
minimum_threshold_comparison = gamma_min / 2 <= gamma / 2
~~~

のshapeは`(B,)`である。全要素がTrueかを、

~~~python
minimum_threshold_is_smallest = torch.all(
    minimum_threshold_comparison
)

assert minimum_threshold_comparison.shape == (B,)
assert minimum_threshold_is_smallest.shape == torch.Size([])
assert bool(minimum_threshold_is_smallest)
~~~

で確認できる。

## 10. Batch-wide certificateの正しい実装

理論上の条件は、

$$
U
<
\frac{\gamma_{\min}}{2}
$$

である。従って、対応するコードは、

~~~python
# 両辺がscalarなので、torch.allは不要。
batch_certificate = full_upper_bound < gamma_min / 2

assert batch_certificate.shape == torch.Size([])
assert bool(batch_certificate)
~~~

である。strict inequalityを使うのは、等号ではtop classとcompetitorのtieを排除できないためである。

### 10.1 間違えやすい式

次の式は今回のcertificateではない。

~~~python
# 誤り: 今回の十分条件とは異なる。
wrong_certificate = full_upper_bound / 2 <= gamma_min / 2
~~~

数式では、

$$
\frac{U}{2}
\le
\frac{\gamma_{\min}}{2}
\quad
\Longleftrightarrow
\quad
U
\le
\gamma_{\min}
$$

であり、必要な、

$$
U
<
\frac{\gamma_{\min}}{2}
$$

より弱い条件になっている。

例えば、

$$
\gamma_{\min}=1.5,
\qquad
U=1.0
$$

なら、

$$
1.0
\le
1.5
$$

はTrueだが、

$$
1.0
<
0.75
$$

はFalseである。

### 10.2 Scalarに対する`torch.all`

~~~python
batch_certificate = torch.all(full_upper_bound < gamma_min / 2)
~~~

も結果は正しい。ただし、括弧内が既にscalar boolean Tensorなので、`torch.all`は冗長である。

## 11. Sample-wise certificatesとbroadcasting

`full_upper_bound`はscalar、`gamma / 2`はshape`(B,)`である。比較するとscalarが各sampleへbroadcastされる。

~~~python
sample_certificates = full_upper_bound < gamma / 2

assert sample_certificates.shape == (B,)
assert sample_certificates.dtype == torch.bool
assert torch.all(sample_certificates)
~~~

このvectorは、

$$
\begin{pmatrix}
U<\gamma_1/2\\
U<\gamma_2/2\\
\vdots\\
U<\gamma_B/2
\end{pmatrix}
$$

の真偽値を並べたものである。

## 12. Denseとcompressedのprediction vectorを比較する

~~~python
# Compressed logitsの各行でargmaxを取る。
y_tilde = torch.argmax(logits_tilde, dim=1)

# 要素比較の結果を1個のscalar boolへまとめる。
all_predictions_match = torch.all(y_tilde == y_star)

assert y_tilde.shape == (B,)
assert y_tilde.dtype == torch.int64
assert all_predictions_match.shape == torch.Size([])
assert bool(all_predictions_match)

# 完全一致を直接確認する場合はこちらでもよい。
assert torch.equal(y_tilde, y_star)
~~~

`batch_certificate`と`all_predictions_match`は別の量である。

- `batch_certificate`：norm上界とminimum marginだけから得る理論保証。
- `all_predictions_match`：この具体的なforward結果における実測一致。

正しい論理は、

$$
\text{batch\_certificate=True}
\quad\Longrightarrow\quad
\text{all\_predictions\_match=True}
$$

である。逆向きは一般には成り立たない。

## 13. 一続きの検証コード

~~~python
# Dense prediction vector。
y_star = torch.argmax(logits, dim=1)

# Sampleごとのtop-1、top-2からmargin vectorを作る。
top2 = torch.topk(logits, k=2, dim=1)
top2_values = top2.values
top2_indices = top2.indices
gamma = top2_values[:, 0] - top2_values[:, 1]

# Batch内のminimum marginと、そのsample index。
gamma_min, bottleneck_index = torch.min(gamma, dim=0)

# Batch-wide十分条件。scalar同士なのでtorch.allは不要。
batch_certificate = full_upper_bound < gamma_min / 2

# 各sampleの条件をbroadcastingで確認する。
sample_certificates = full_upper_bound < gamma / 2

# Compressed prediction vectorと実測一致。
y_tilde = torch.argmax(logits_tilde, dim=1)
all_predictions_match = torch.equal(y_tilde, y_star)

assert torch.all(gamma_min / 2 <= gamma / 2)
assert bool(batch_certificate)
assert torch.all(sample_certificates)
assert all_predictions_match

print(f"||Delta L||_F       = {delta_l_fro.item():.8f}")
print(f"U                    = {full_upper_bound.item():.8f}")
print("y_star               =", y_star.tolist())
print("top2 values           =", top2_values.tolist())
print("top2 class indices    =", top2_indices.tolist())
print("gamma                 =", gamma.tolist())
print(f"gamma_min             = {gamma_min.item():.8f}")
print(f"gamma_min / 2         = {(gamma_min / 2).item():.8f}")
print(f"bottleneck sample idx = {int(bottleneck_index)}")
print("sample certificates   =", sample_certificates.tolist())
print("batch certificate     =", bool(batch_certificate))
print("y_tilde               =", y_tilde.tolist())
print("all predictions match =", all_predictions_match)
~~~

## 14. Tieに関する実装上の注意

`torch.argmax`は最大値が複数ある場合、最初のindexを返す。一方、top-1とtop-2が同じ値なら、

$$
\gamma_n=0
$$

である。

今回のstrictなcertificateでは、そのsampleを含むbatchに正の共通stability radiusを与えられない。そこでtoy検証では、

~~~python
assert torch.all(gamma > 0.0)
~~~

によりdense top classの一意性を確認する。

## 15. 現行Notebookとの対応と確認範囲

対応する学習Notebookは、[06_minimum_margin_batchwide_stability.ipynb](../../notebooks/30_tt_mps/10_fashion_mnist_mlp/06_minimum_margin_batchwide_stability.ipynb)である。

現行sourceでは、レビューで問題になったcertificateは、

~~~python
batch_certificate = torch.all(full_upper_bound < gamma_min / 2)
~~~

へ修正されている。`torch.all`は冗長だが、比較式そのものは正しい。

全code cellに`execution_count`はあるが、17から46まで非連続な順序で保存されている。従って、次を区別する。

- 現行source、保存済み出力、assertの記載は確認した。
- Certificateの比較式が現行sourceで訂正済みであることを確認した。
- 教材化時に新しいkernelでRestart Kernel、Run Allしたわけではない。
- 非連続な`execution_count`は、現行sourceを先頭から同一実行した証明ではない。

また、Notebook末尾は次をTT-rank sweepとしているが、この学習段階で確定した次テーマはground-truth labelを導入し、prediction agreementとaccuracyの違いを扱うことである。今回のbatch-wide certificateの結論には影響しないが、学習順序としては区別する。

固定toy MLPの保存済み数値、全sampleのthreshold、結果の考察は、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/03_MinimumMarginとBatch-widePredictionStability_設定結果考察]]を参照する。
