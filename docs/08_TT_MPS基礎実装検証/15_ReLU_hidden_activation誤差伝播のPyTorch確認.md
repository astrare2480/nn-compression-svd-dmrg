---
title: ReLU hidden activation誤差伝播のPyTorch確認
tags:
  - TTLinear
  - ReLU
  - PyTorch
  - hidden-activation
  - error-bound
---

# ReLU hidden activation誤差伝播のPyTorch確認

## このノートの位置づけ

このノートは、[02_relu_hidden_activation_error_todo.ipynb](../../notebooks/30_tt_mps/10_fashion_mnist_mlp/02_relu_hidden_activation_error_todo.ipynb) で扱ったReLUの要素不等式、Frobenius誤差、LinearからReLUまでの誤差上界を整理する。

数学の導出は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/61_Lipschitz連続性とReLUの1-Lipschitz性]]と[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/63_ReLUによるhidden_activation誤差伝播]]を参照する。PyTorchのnorm APIは、[[08_TT_MPS基礎実装検証/30_PyTorchのvector_norm_matrix_normとtorch_linalg]]へ分ける。

## 1. 証拠レベル

現行Notebookには、一部のcellについてexecution countと保存出力が残っている。しかし、最後の別形式上界cellは`execution_count=null`であり、Notebook全体を新しいkernelで上からFresh Run Allした証拠はない。

このノートでは、次を区別する。

- コードを読んで確認した内容：静的確認
- Notebookに残る表示：保存済み出力
- 今回行っていないこと：Fresh Run All

Notebookそのものはこの教材化では変更していない。

## 2. 実行条件

Notebookの設定は次である。

~~~text
torch.manual_seed(0)
dtype: torch.float64
device: CPU
rtol: 1e-12
atol: 1e-12
inequality_atol: 1e-12
~~~

`inequality_atol`は、理論上の不等式へ追加する項ではない。浮動小数点演算の丸め差を許すための検査用許容幅である。

## 3. 実験1：4種類の符号ケース

Notebookでは、

$$
Z=
\begin{pmatrix}
2&-2&3&-4\\
1&-5&-1&4
\end{pmatrix},
\qquad
\widetilde Z=
\begin{pmatrix}
5&-7&-2&2\\
3&-1&2&1
\end{pmatrix}
$$

を`torch.float64`の2階tensorとして作る。すべてのtensorはshape `(2, 4)` を持つ。

中心となる計算は次である。

~~~python
# ReLUを要素ごとに適用する。
h = torch.relu(z)
h_tilde = torch.relu(z_tilde)

# ReLU前後の差を作る。
delta_z = z_tilde - z
delta_h = h_tilde - h

# 全要素でReLUの1-Lipschitz不等式を検査する。
elementwise_holds = (
    delta_h.abs()
    <= delta_z.abs() + inequality_atol
)
assert torch.all(elementwise_holds)
~~~

### 3.1 `abs()`が返すもの

`delta_h.abs()`は各要素の絶対値を持つshape `(2, 4)` のtensorである。Frobenius normのような1個のスカラーではない。

$$
|\Delta H|
=
\begin{pmatrix}
3&0&3&2\\
2&0&2&3
\end{pmatrix}.
$$

同様に、

$$
|\Delta Z|
=
\begin{pmatrix}
3&5&5&6\\
2&4&3&3
\end{pmatrix}.
$$

### 3.2 比較演算が返すもの

~~~python
# 左右を要素ごとに比較するため、結果もshape (2, 4)のbool tensorになる。
elementwise_holds = (
    delta_h.abs()
    <= delta_z.abs() + inequality_atol
)
~~~

この結果は、概念的には

~~~text
[[True, True, True, True],
 [True, True, True, True]]
~~~

となる。`torch.all(elementwise_holds)`で全要素を1個のscalar Boolean Tensorへ集約する。Pythonの`bool`が必要なら`.item()`で値を取り出す。

~~~python
# 全要素がTrueかをscalar Tensorとして集約する。
all_holds_tensor = torch.all(elementwise_holds)

# 表示やPythonの分岐に使う場合はPython boolへ変換する。
all_holds = bool(all_holds_tensor.item())
assert all_holds
~~~

### 3.3 判定を段階に分ける

一行の`assert`へまとめる前に、途中tensorを分けると検査内容が見やすい。

~~~python
# 1. 各要素の絶対誤差を作る。
abs_delta_h = delta_h.abs()
abs_delta_z = delta_z.abs()

# 2. 要素ごとの不等式を判定する。
elementwise_holds = (
    abs_delta_h
    <= abs_delta_z + inequality_atol
)

# 3. 全要素がTrueかを確認する。
all_elements_hold = torch.all(elementwise_holds)
assert bool(all_elements_hold.item())
~~~

## 4. 保存済みの全要素表示

Notebookの符号分類cellには、全8要素の保存出力がある。

~~~text
(0, 0): Z= 2.0, Z_tilde= 5.0, both nonnegative,
        |delta_H|=3.0, |delta_Z|=3.0, True
(0, 1): Z=-2.0, Z_tilde=-7.0, both negative,
        |delta_H|=0.0, |delta_Z|=5.0, True
(0, 2): Z= 3.0, Z_tilde=-2.0, dense nonnegative / approximate negative,
        |delta_H|=3.0, |delta_Z|=5.0, True
(0, 3): Z=-4.0, Z_tilde= 2.0, dense negative / approximate nonnegative,
        |delta_H|=2.0, |delta_Z|=6.0, True
(1, 0): Z= 1.0, Z_tilde= 3.0, both nonnegative,
        |delta_H|=2.0, |delta_Z|=2.0, True
(1, 1): Z=-5.0, Z_tilde=-1.0, both negative,
        |delta_H|=0.0, |delta_Z|=4.0, True
(1, 2): Z=-1.0, Z_tilde= 2.0, dense negative / approximate nonnegative,
        |delta_H|=2.0, |delta_Z|=3.0, True
(1, 3): Z= 4.0, Z_tilde= 1.0, both nonnegative,
        |delta_H|=3.0, |delta_Z|=3.0, True
~~~

したがって、4種類の符号ケースをすべて含み、全要素で

$$
|\Delta H_{ij}|
\le
|\Delta Z_{ij}|
$$

が成立することを保存出力から確認できる。

## 5. 実験2：Frobeniusノルム

コードは、

~~~python
# ReLU前後のbatch全体誤差をFrobenius normで測る。
delta_z_fro = torch.linalg.matrix_norm(
    delta_z,
    ord="fro",
)
delta_h_fro = torch.linalg.matrix_norm(
    delta_h,
    ord="fro",
)

# ReLUがFrobenius誤差を増幅しないことを検査する。
assert float(delta_h_fro) <= (
    float(delta_z_fro) + inequality_atol
)
~~~

となっている。`matrix_norm(..., ord="fro")`の返り値はshape `()`のscalar Tensorである。

具体行列から手計算すると、

$$
\|\Delta H\|_F=\sqrt{39}\approx6.2450,
\qquad
\|\Delta Z\|_F=\sqrt{133}\approx11.5326.
$$

現行cellにはexecution countがあるが、値を`print`していないため、この2値そのものは保存出力には残っていない。不等式を検査する`assert`が例外なしで完了した実行状態は残っている。

scalar Tensor同士はそのまま比較できるため、`float(delta_h_fro)`と`float(delta_z_fro)`への変換は必須ではない。変換しても数値判定は誤りではないが、PyTorch Tensorのまま

~~~python
# scalar TensorのままFrobenius不等式を検査する。
assert delta_h_fro <= delta_z_fro + inequality_atol
~~~

と書く方が、他のtensor計算と統一しやすい。また、手計算値との対応を保存するなら、2個のnormを明示的に`print`する。

## 6. 実験3：LinearからReLUまで

randomな小さいLinear例のshapeは、

~~~text
batch size B = 4
input dimension n = 6
hidden dimension h = 5

X.shape       = (4, 6)
W.shape       = (5, 6)
W_tilde.shape = (5, 6)
b.shape       = (5,)
~~~

である。近似重みは、

~~~python
# 非零の誤差を作るため、小さいrandom摂動を重みに加える。
small_perturbation = 0.05 * torch.randn(
    hidden_features,
    in_features,
    dtype=dtype,
    device=device,
)
w_tilde = w + small_perturbation
~~~

として作る。

### 6.1 誤差恒等式

~~~python
# dense重みと近似重みのpre-activationを計算する。
z = x @ w.T + b
z_tilde = x @ w_tilde.T + b

# 重み誤差と実測pre-activation誤差を計算する。
delta_w = w_tilde - w
delta_z = z_tilde - z

# 数式Delta_Z = X Delta_W^Tを別経路で計算する。
delta_z_from_formula = x @ delta_w.T

# 2経路の結果をfloat64の許容誤差内で照合する。
torch.testing.assert_close(
    delta_z,
    delta_z_from_formula,
    rtol=rtol,
    atol=atol,
)
~~~

これは、

$$
\Delta Z
=
X\Delta W^{\mathsf T}
$$

を、実測差と数式からの再計算で照合している。

### 6.2 ReLU後の誤差

~~~python
# ReLU後のhidden activationを作る。
h = torch.relu(z)
h_tilde = torch.relu(z_tilde)
delta_h = h_tilde - h

# 全sample・全hidden unitで要素不等式を検査する。
assert torch.all(
    delta_h.abs()
    <= delta_z.abs() + inequality_atol
)
~~~

### 6.3 1本目のLinear上界

~~~python
# XはFrobenius norm、Delta_Wはspectral normで測る。
x_fro = torch.linalg.matrix_norm(x, ord="fro")
delta_w_spectral = torch.linalg.matrix_norm(
    delta_w,
    ord=2,
)

# ReLU前後の誤差はFrobenius normで測る。
delta_z_fro = torch.linalg.matrix_norm(
    delta_z,
    ord="fro",
)
delta_h_fro = torch.linalg.matrix_norm(
    delta_h,
    ord="fro",
)

# Linear出力誤差上界を計算する。
linear_bound = x_fro * delta_w_spectral

assert delta_h_fro <= delta_z_fro + inequality_atol
assert delta_z_fro <= linear_bound + inequality_atol
assert delta_h_fro <= linear_bound + inequality_atol
~~~

このコードが検査する連鎖は、

$$
\boxed{
\|\Delta H\|_F
\le
\|\Delta Z\|_F
\le
\|X\|_F\|\Delta W\|_2
}
$$

である。

## 7. 保存済み数値結果

1本目の上界cellには次の保存出力がある。

~~~text
||Delta_H||_F         = 0.337250011229
||Delta_Z||_F         = 0.519605165542
||X||_F               = 5.091422875106
||Delta_W||_2         = 0.175429273041
||X||_F ||Delta_W||_2 = 0.893184613722
~~~

したがって、保存出力上では

$$
0.337250011229
\le
0.519605165542
\le
0.893184613722
$$

が成り立つ。

この結果は、ReLUがこのbatchのFrobenius誤差を縮小したことと、Linear出力誤差がspectral normを使う上界以下であることを示す。

## 8. 2本目のLinear上界

別形式は、

$$
\boxed{
\|\Delta H\|_F
\le
\|\Delta Z\|_F
\le
\|X\|_2\|\Delta W\|_F
}
$$

である。対応コードは、

~~~python
# Xはspectral norm、Delta_WはFrobenius normで測る。
x_spectral = torch.linalg.matrix_norm(x, ord=2)
delta_w_fro = torch.linalg.matrix_norm(
    delta_w,
    ord="fro",
)

# 別形式のLinear出力誤差上界を計算する。
linear_bound_alt = x_spectral * delta_w_fro

assert delta_h_fro <= delta_z_fro + inequality_atol
assert delta_z_fro <= linear_bound_alt + inequality_atol
assert delta_h_fro <= linear_bound_alt + inequality_atol
~~~

である。

ただし、現行Notebookではこのcellの`execution_count`が`null`で、保存出力もない。コードの式対応は静的に確認できるが、このcellを現行ソースで実行済みとは扱わない。

## 9. `vector_norm`から`matrix_norm`への整理

2階tensor $A$ に対して、

$$
\|\operatorname{vec}(A)\|_2
=
\|A\|_F
$$

なので、`torch.linalg.vector_norm(A)`と`torch.linalg.matrix_norm(A, ord="fro")`は同じ数値になる。

それでも、今回の数式は行列のFrobenius normを使っているため、教材コードでは

~~~python
# 行列のFrobenius normであることをAPI名とordで明示する。
delta_h_fro = torch.linalg.matrix_norm(
    delta_h,
    ord="fro",
)
~~~

と書く方が意味を追いやすい。`matrix_norm(delta_w, ord=2)`はspectral normなので、Frobenius normとは別物である。

詳細なAPI比較は、[[08_TT_MPS基礎実装検証/30_PyTorchのvector_norm_matrix_normとtorch_linalg]]を参照する。

## 10. `KMP_DUPLICATE_LIB_OK`について

現行Notebookの初期化cellには、次の順で記述されている。

~~~python
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
~~~

この環境変数はOpenMP runtimeの重複ロード問題を回避するために使われることがあるが、通常の教材コードへ無条件に入れる設定ではない。また、重複エラーが`import torch`中に起きる環境では、import後に設定しても最初のimportを救えない。

2026-10-04の独立したCLI検証では、環境変数を設定しない`import torch`で`OMP: Error #15`が再現した。これはCLI検証プロセスの結果であり、保存済みNotebook kernelの状態を示すものではない。回避変数をimport前に設定した隔離プロセスでは、このノートの小行列・norm数値例を実行できた。

したがって、現在の環境について単純に「不要」とは判定しない。一方、この回避変数はruntime衝突を隠す可能性があるため、恒久解決として扱わない。依存関係とOpenMP runtimeの重複原因を別途確認し、必要な場合だけimportより前に限定使用する。

## 11. 現行Notebookの静的監査

2026-10-04時点の保存状態は次である。

現行ソースでは`TODO 1`から`TODO 6`までの実装欄にコードが入っている。これは現在保存されている学習経過であり、この教材化のために解答コードを追加したものではない。

| 内容 | execution count | 保存出力・判定 |
| --- | ---: | --- |
| 実行環境 | 19 | `float64`、CPUを表示 |
| $2\times4$ toy行列の作成と要素assert | 20 | assertのみ、表示なし |
| 全8要素の符号ケース | 21 | 全8要素の表示あり |
| toy行列のFrobenius不等式 | 22 | assertのみ、値の表示なし |
| random Linearのshape | 23 | `(4,6)`, `(5,6)`, `(5,)`を表示 |
| $\Delta Z=X\Delta W^{\mathsf T}$ | 24 | `assert_close`のみ |
| ReLU要素不等式 | 25 | assertのみ |
| 1本目のnorm連鎖 | 28 | 5個の数値を表示 |
| 2本目のnorm連鎖 | null | 未実行、表示なし |

また、1本目の上界cellには同じ`linear_bound`代入が2回あり、tensor比較のassertと`float(...)`へ変換したassertが重複している。数学的結果は変わらないが、教材を整理するときは一方へ統一できる。

## 12. 完了範囲

このNotebookと保存出力で確認できるのは、次である。

- ReLUの4種類の符号ケース
- 全8要素の $|\Delta H_{ij}|\le|\Delta Z_{ij}|$
- Frobeniusノルムでのnon-expansive性
- $\Delta Z=X\Delta W^{\mathsf T}$ のコード上の照合
- $\|\Delta H\|_F\le\|\Delta Z\|_F\le\|X\|_F\|\Delta W\|_2$
- もう一つの上界に対応するコード
- `abs()`、要素比較、`torch.all`、`matrix_norm`の意味

次は未完了である。

- 現行NotebookのFresh Run All
- 2本目の上界cellの保存済み実行結果
- 学習済みFashion-MNIST MLPの打ち切りrankでのhidden activation比較
- 後続Linearを含むlogits誤差上界
- marginとprediction安定性
- rank sweep、test accuracy、fine-tuning

したがって、この結果は「小さいtoy例でLinearからReLUまでの誤差伝播を確認した」と判定する。Fashion-MNIST実データでのNN圧縮検証が完了したとは判定しない。
