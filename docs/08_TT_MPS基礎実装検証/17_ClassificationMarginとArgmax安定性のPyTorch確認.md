---
title: Classification MarginとArgmax安定性のPyTorch確認
tags:
  - PyTorch
  - logits
  - classification-margin
  - argmax
  - TTLinear
---

# Classification MarginとArgmax安定性のPyTorch確認

## 1. このノートの役割

1 sampleのdense logitsとcompressed logitsから、次をPyTorchで計算する方法を整理する。

$$
\delta l_n
=
\widetilde l_n-l_n,
$$

$$
y_n^*
=
\operatorname*{argmax}_k(l_n)_k,
$$

$$
\gamma_n
=
(l_n)_{y_n^*}
-
\max_{k\ne y_n^*}(l_n)_k,
$$

$$
\varepsilon_n
=
\|\delta l_n\|_\infty.
$$

一般式の証明は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]を参照する。このノートでは、shape、PyTorch API、copyとin-place変更、assert、直接条件の実装に焦点を置く。

## 2. 変数とshape

class数を $c$ とし、1 sampleだけを扱う。

| 変数 | 意味 | shape | dtype |
| --- | --- | ---: | --- |
| `c` | class数 | Python scalar | `int` |
| `l_n` | dense logits | `(c,)` | floating Tensor |
| `l_tilde_n` | compressed logits | `(c,)` | floating Tensor |
| `delta_l_n` | logits error | `(c,)` | floating Tensor |
| `y_star` | dense predicted class | `()` | integer Tensor |
| `competitor_index` | strongest competitorのclass index | `()` | integer Tensor |
| `top_logit` | top-class logit | `()` | floating Tensor |
| `competitor_logit` | strongest competitor logit | `()` | floating Tensor |
| `gamma_n` | classification margin | `()` | floating Tensor |
| `epsilon_n` | infinity-norm logits error | `()` | floating Tensor |

PyTorchのscalar Tensorのshapeは、

~~~text
torch.Size([])
~~~

である。`y_star` はlogit値ではなくclass indexである。

## 3. 実行条件とdense logits

~~~python
import torch
from torch.testing import assert_close

# 小さい差を確認しやすいようにfloat64を使う。
torch.manual_seed(0)
dtype = torch.float64
device = torch.device("cpu")
rtol = 1e-10
atol = 1e-12

# 1 sample、5 classesのdense logits。
c = 5
l_n = torch.tensor(
    [2.4, 1.6, 0.2, -0.5, 1.2],
    dtype=dtype,
    device=device,
)

assert l_n.shape == (c,)
~~~

この例ではclass 0がtop、class 1がstrongest competitorになる。

## 4. `max`と`argmax`

~~~python
# 最大logitの値を返す。
top_logit_by_max = torch.max(l_n)

# 最大logitを持つclass indexを返す。
y_star = torch.argmax(l_n)

assert top_logit_by_max.ndim == 0
assert y_star.ndim == 0
assert int(y_star) == 0
~~~

`torch.max(l_n)`の結果はfloating scalar Tensorであり、`torch.argmax(l_n)`の結果はinteger scalar Tensorである。

## 5. Strongest competitorとmargin

### 5.1 実装

~~~python
# top classを除外する作業用Tensorを作る。
masked = l_n.clone()

# top classを最大値探索の候補から外す。
masked[y_star] = float("-inf")

# top class以外で最大logitを持つclass index。
competitor_index = torch.argmax(masked)

# marginは元のdense logitsに対して定義する。
top_logit = l_n[y_star]
competitor_logit = l_n[competitor_index]
gamma_n = top_logit - competitor_logit

assert y_star.ndim == 0
assert competitor_index.ndim == 0
assert gamma_n.ndim == 0
assert int(y_star) == 0
assert int(competitor_index) == 1
assert_close(
    gamma_n,
    torch.tensor(0.8, dtype=dtype, device=device),
    rtol=rtol,
    atol=atol,
)
~~~

各行は次の数式に対応する。

$$
y_n^*
=
\operatorname*{argmax}_k(l_n)_k,
$$

$$
k_n^{\mathrm{comp}}
=
\operatorname*{argmax}_{k\ne y_n^*}(l_n)_k,
$$

$$
\gamma_n
=
(l_n)_{y_n^*}
-
(l_n)_{k_n^{\mathrm{comp}}}.
$$

全要素の変化は、

$$
l_n
=
\begin{pmatrix}
2.4\\
1.6\\
0.2\\
-0.5\\
1.2
\end{pmatrix}
\longrightarrow
\texttt{masked}
=
\begin{pmatrix}
-\infty\\
1.6\\
0.2\\
-0.5\\
1.2
\end{pmatrix}
$$

である。従って、`torch.argmax(masked)`はclass 1を返す。

### 5.2 `clone()`が必要な理由

~~~python
masked = l_n.clone()
~~~

は、値を複製した別Tensorを作る。続く、

~~~python
masked[y_star] = float("-inf")
~~~

はin-place変更なので、作業用Tensorだけを変更する必要がある。

次の書き方ではcopyにならず、`masked`と`l_n`が同じTensorを参照する。

~~~python
# この書き方は避ける。
masked = l_n
masked[y_star] = float("-inf")
~~~

この場合、元の`l_n`まで書き換わり、その後のmargin計算が壊れる。

### 5.3 なぜ`-inf`を入れるのか

top logitを有限の小さい数へ置き換えると、それより小さいlogitが存在する場合に除外へ失敗する。`-inf`は全ての有限logitより小さいので、top classを最大値探索から確実に外せる。

`masked`はcompetitor indexを探す作業用Tensorである。competitorの値は、定義との対応を明確にするため、

~~~python
competitor_logit = l_n[competitor_index]
~~~

と元の`l_n`から取得する。

## 6. Infinity normの実装

$$
\varepsilon_n
=
\|\delta l_n\|_\infty
=
\max_k|(\delta l_n)_k|
$$

は、次のどちらでも計算できる。

~~~python
# 数学記号のinfinity normに直接対応する書き方。
epsilon_by_norm = torch.linalg.vector_norm(
    delta_l_n,
    ord=float("inf"),
)

# 絶対値を取り、その最大成分を選ぶ書き方。
epsilon_by_max = torch.max(torch.abs(delta_l_n))

assert_close(
    epsilon_by_norm,
    epsilon_by_max,
    rtol=rtol,
    atol=atol,
)
~~~

`ord`はnormの種類を指定する。

| 指定 | 数式 | 計算 |
| --- | --- | --- |
| `ord=1` | $\|x\|_1$ | 絶対値の総和 |
| `ord=2` | $\|x\|_2$ | 二乗和の平方根 |
| `ord=float("inf")` | $\|x\|_\infty$ | 絶対値の最大値 |

`float("inf")`は誤差を無限大にする指定ではない。infinity normを選ぶための値である。

## 7. Safe case

~~~python
# 十分条件を満たすlogits error。
delta_l_safe = torch.tensor(
    [-0.20, 0.15, -0.10, 0.05, 0.10],
    dtype=dtype,
    device=device,
)

# compressed logitsを作る。
l_tilde_safe = l_n + delta_l_safe

# 最大成分誤差を測る。
epsilon_safe = torch.linalg.vector_norm(
    delta_l_safe,
    ord=float("inf"),
)

# compressed predictionを得る。
y_tilde_safe = torch.argmax(l_tilde_safe)

# 十分条件と実際のprediction一致を別々に判定する。
sufficient_condition_safe = epsilon_safe < gamma_n / 2
argmax_preserved_safe = y_tilde_safe == y_star
~~~

各要素を足すと、

$$
\begin{aligned}
\widetilde l_n
&=
\begin{pmatrix}
2.4\\
1.6\\
0.2\\
-0.5\\
1.2
\end{pmatrix}
+
\begin{pmatrix}
-0.20\\
0.15\\
-0.10\\
0.05\\
0.10
\end{pmatrix}\\
&=
\begin{pmatrix}
2.2\\
1.75\\
0.10\\
-0.45\\
1.30
\end{pmatrix}.
\end{aligned}
$$

$$
\varepsilon_n
=
\max(0.20,0.15,0.10,0.05,0.10)
=
0.20,
$$

$$
\frac{\gamma_n}{2}
=
\frac{0.8}{2}
=
0.4.
$$

従って、

$$
0.20<0.40
$$

であり、class 0が保存される。

~~~python
assert l_tilde_safe.shape == (c,)
assert epsilon_safe.ndim == 0
assert y_tilde_safe.ndim == 0
assert sufficient_condition_safe.item() is True
assert argmax_preserved_safe.item() is True

assert_close(
    l_tilde_safe,
    torch.tensor(
        [2.2, 1.75, 0.1, -0.45, 1.3],
        dtype=dtype,
        device=device,
    ),
    rtol=rtol,
    atol=atol,
)
~~~

## 8. 全competitorに対する下界の確認

~~~python
lower_bound = gamma_n - 2.0 * epsilon_safe

for k in range(c):
    if k == int(y_star):
        continue

    # compressed logitsにおけるtop classとclass kの差。
    compressed_gap_k = l_tilde_safe[y_star] - l_tilde_safe[k]

    # dense gapと誤差によるgap変化を別々に計算する。
    dense_gap_k = l_n[y_star] - l_n[k]
    error_gap_k = delta_l_safe[y_star] - delta_l_safe[k]

    # 理論で使った恒等式を独立に確認する。
    assert_close(
        compressed_gap_k,
        dense_gap_k + error_gap_k,
        rtol=rtol,
        atol=atol,
    )

    # 共通下界とstrictな正値を確認する。
    assert compressed_gap_k >= lower_bound
    assert compressed_gap_k > 0.0
~~~

## 9. Worst-case方向で$2\varepsilon$を確認する

~~~python
# dtypeとdeviceを他のTensorに合わせたscalar Tensorにする。
epsilon_worst = torch.tensor(
    0.30,
    dtype=dtype,
    device=device,
)

# 全成分0から始める。
delta_l_worst = torch.zeros(
    c,
    dtype=dtype,
    device=device,
)

# top classを下げ、strongest competitorを上げる。
delta_l_worst[y_star] = -epsilon_worst
delta_l_worst[competitor_index] = +epsilon_worst

# 誤差前後の同じ2 class間のgapを比較する。
l_tilde_worst = l_n + delta_l_worst
gap_before = l_n[y_star] - l_n[competitor_index]
gap_after = (
    l_tilde_worst[y_star]
    - l_tilde_worst[competitor_index]
)

assert_close(gap_before, gamma_n, rtol=rtol, atol=atol)
assert_close(
    gap_after,
    gap_before - 2.0 * epsilon_worst,
    rtol=rtol,
    atol=atol,
)
~~~

この例では、

$$
\delta l_{\mathrm{worst}}
=
\begin{pmatrix}
-0.3\\
0.3\\
0\\
0\\
0
\end{pmatrix},
$$

$$
\widetilde l_{\mathrm{worst}}
=
\begin{pmatrix}
2.1\\
1.9\\
0.2\\
-0.5\\
1.2
\end{pmatrix}.
$$

$$
\texttt{gap\_before}
=
2.4-1.6
=
0.8,
$$

$$
\texttt{gap\_after}
=
2.1-1.9
=
0.2
=
0.8-2\cdot0.3.
$$

`epsilon_worst`をPython `float`にしてもこの例は動くが、scalar Tensorにするとdtypeとdeviceを一貫して管理できる。

## 10. 十分条件ではなく直接条件を確認する関数

~~~python
def check_direct_argmax_stability(
    l_n: torch.Tensor,
    delta_l_n: torch.Tensor,
) -> bool:
    """dense top classがperturbation後もstrictなtopか確認する。"""
    if l_n.ndim != 1 or delta_l_n.ndim != 1:
        raise ValueError("l_n and delta_l_n must have shape (c,)")
    if l_n.shape != delta_l_n.shape:
        raise ValueError("l_n and delta_l_n must have the same shape")
    if l_n.numel() < 2:
        raise ValueError("at least two classes are required")

    y_star = torch.argmax(l_n)
    l_tilde_n = l_n + delta_l_n

    all_strictly_stable = True

    for k in range(l_n.numel()):
        if k == int(y_star):
            continue

        # competitorがtop classに対して得た相対的な誤差優位。
        relative_error_advantage = (
            delta_l_n[k]
            - delta_l_n[y_star]
        )

        # dense modelが元々持っていたtop-vs-competitor gap。
        dense_gap = l_n[y_star] - l_n[k]

        # strictなtopを保つ直接条件。
        condition_k = relative_error_advantage < dense_gap
        all_strictly_stable = (
            all_strictly_stable
            and bool(condition_k)
        )

        print(
            f"k={k}: "
            f"relative_error_advantage="
            f"{relative_error_advantage.item():.6f}, "
            f"dense_gap={dense_gap.item():.6f}, "
            f"stable={bool(condition_k)}"
        )

    # y_starが全competitorへstrictに勝つかをlogitsから直接確認する。
    actual_strict_preservation = all(
        bool(l_tilde_n[y_star] > l_tilde_n[k])
        for k in range(l_n.numel())
        if k != int(y_star)
    )
    assert all_strictly_stable == actual_strict_preservation

    return all_strictly_stable
~~~

この関数の判定は、

$$
(\delta l_n)_k
-
(\delta l_n)_{y_n^*}
<
(l_n)_{y_n^*}
-
(l_n)_k
$$

を全competitorについて直接実装している。境界でtieになる例では、strictなtopの判定と`torch.argmax`の返すindexを同一視しない。

## 11. 同じnormで異なる結果を確認する

~~~python
# 3-classの小さい例。
l_direction = torch.tensor(
    [1.0, 4.0, 3.0],
    dtype=dtype,
    device=device,
)

# top classに有利な誤差。
delta_favorable = torch.tensor(
    [0.0, 0.8, -0.8],
    dtype=dtype,
    device=device,
)

# top classに不利な誤差。
delta_adversarial = torch.tensor(
    [0.0, -0.8, 0.8],
    dtype=dtype,
    device=device,
)

epsilon_favorable = torch.linalg.vector_norm(
    delta_favorable,
    ord=float("inf"),
)
epsilon_adversarial = torch.linalg.vector_norm(
    delta_adversarial,
    ord=float("inf"),
)

assert_close(
    epsilon_favorable,
    epsilon_adversarial,
    rtol=rtol,
    atol=atol,
)

assert check_direct_argmax_stability(
    l_direction,
    delta_favorable,
)
assert not check_direct_argmax_stability(
    l_direction,
    delta_adversarial,
)
~~~

ここで`adversarial`は入力空間のadversarial attackではなく、top classを下げ、competitorを上げるためlogit順位に不利な誤差方向、という意味で使っている。

両方の誤差normは $0.8$ だが、前者はpredictionを保存し、後者はclass 1からclass 2へ反転させる。

## 12. Common shiftの確認

~~~python
# 全classへ同じ+1を加える。
delta_l_large_same_shift = torch.ones(
    c,
    dtype=dtype,
    device=device,
)

l_tilde_large_same_shift = (
    l_n + delta_l_large_same_shift
)
epsilon_large_same_shift = torch.linalg.vector_norm(
    delta_l_large_same_shift,
    ord=float("inf"),
)
y_tilde_large_same_shift = torch.argmax(
    l_tilde_large_same_shift
)

# 十分条件は満たさない。
assert epsilon_large_same_shift >= gamma_n / 2

# しかし実際のargmaxは変わらない。
assert int(y_tilde_large_same_shift) == int(y_star)

# 各class間の差も全て保存される。
assert_close(
    l_tilde_large_same_shift[:, None]
    - l_tilde_large_same_shift[None, :],
    l_n[:, None] - l_n[None, :],
    rtol=rtol,
    atol=atol,
)
~~~

$$
l_n
=
\begin{pmatrix}
2.4\\
1.6\\
0.2\\
-0.5\\
1.2
\end{pmatrix},
\qquad
\delta l_n
=
\begin{pmatrix}
1\\
1\\
1\\
1\\
1
\end{pmatrix}
$$

なので、

$$
\widetilde l_n
=
\begin{pmatrix}
3.4\\
2.6\\
1.2\\
0.5\\
2.2
\end{pmatrix}.
$$

$\varepsilon_n=1.0$ は $\gamma_n/2=0.4$ より大きいが、全classに同じ値を加えただけなので順位は変わらない。

## 13. 実装時の判定を分ける

次の二つは別のboolとして保持する。

~~~python
# normだけから得る理論上の十分条件。
sufficient_condition = epsilon_n < gamma_n / 2

# 実際のlogitsでpredictionが一致したか。
argmax_preserved = (
    torch.argmax(l_tilde_n)
    == torch.argmax(l_n)
)
~~~

`sufficient_condition == False` は、`argmax_preserved == False` を意味しない。前者はworst-case保証の不成立、後者は実際のprediction変化である。

## 14. 現行Notebookとの対応

対応Notebookは、[04_classification_margin_argmax_stability.ipynb](../../notebooks/30_tt_mps/10_fashion_mnist_mlp/04_classification_margin_argmax_stability.ipynb)である。

Notebookは次を扱う。

- `clone()`と`-inf`によるstrongest competitorの取得。
- safe caseの$\ell_\infty$ errorとargmax一致。
- 全competitorに対するgap下界。
- worst-case方向による$2\varepsilon$のmargin縮小。
- 大きなcommon shiftによる「十分条件は必要条件ではない」例。

Notebookの現行ソースではTODO欄に実装が記入済みである。学習用として再利用するときは、完成コードを先に読むのではなく、各TODOの説明と期待値を確認してから自分で再実装する。

この実装ノートでは補足として、現行Notebookにないfavorable/adversarialの対比例と、実際の誤差方向を使う直接条件も示した。これらをNotebookの保存済み実行結果としては扱わない。

Notebookの設定、保存済み結果、確認できない範囲は、[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/01_ClassificationMarginとArgmax安定性_設定結果考察]]を参照する。
