---
title: Fashion-MNIST MLP一層TTLinear置換とlogits等価性のPyTorch確認
tags:
  - TT-matrix
  - TTLinear
  - Fashion-MNIST
  - MLP
  - PyTorch
  - validation
---

# Fashion-MNIST MLP一層TTLinear置換とlogits等価性のPyTorch確認

## このノートの位置づけ

このノートは、[00_mlp_ttlinear_logits_equivalence.ipynb](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/00_mlp_ttlinear_logits_equivalence.ipynb) に保存された、学習済みFashion-MNIST MLPの第1層を打ち切りなしTTLinearへ置換する検証を整理する。

数学上の等価性は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/59_TTLinearによるMLP一層置換とlogits等価性]]を参照する。mode設計は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/62_TT-matrixのmode分解とrank設計]]へ分ける。

## 1. 証拠レベル

このノートに記録する数値は、Notebookファイルに保存されている出力である。全code cellにはexecution countが残っているが、この教材化の時点で新しいkernelからRestart Kernel、Run Allを再実行したわけではない。

従って、次を区別する。

- 保存済み出力から確認できること
- 現行ソースを静的に読んで確認できること
- Fresh Run Allで今回再確認したこと

今回の第3項は実施していない。

## 2. 実行条件

保存済みNotebookでは次を用いた。

~~~text
PyTorch: 2.11.0+cu128
device: CPU
default dtype: torch.float64
torch.manual_seed(0)
batch size: 4
~~~

checkpointは次である。

~~~text
models/10_svd/20_fashion_mnist_mlp/
00_mlp_baseline_fixed_50epochs/
00_mlp_baseline_fixed_50epochs.pt
~~~

モデルは同じ構造の <code>MNISTMLP</code> を先に生成し、その <code>state_dict</code> を読み込んでいる。

## 3. Linear層の棚卸し

<code>named_modules()</code> で列挙した保存済み結果は次である。

| path | weight shape | in | out | bias込みparameter数 |
| --- | ---: | ---: | ---: | ---: |
| fc1 | $(512,784)$ | 784 | 512 | 401,920 |
| fc2 | $(256,512)$ | 512 | 256 | 131,328 |
| fc3 | $(10,256)$ | 256 | 10 | 2,570 |

最大の <code>fc1</code> を置換対象にした。

$$
W_1\in\mathbb{R}^{512\times784}.
$$

このNotebookは学習済み重みを使用しているが、Fashion-MNIST test setに対するaccuracy実験ではない。入力は同一のrandom batchである。

## 4. mode分解

4 coreを選び、各側をなるべく均等な非減少因子へ分解する関数から、

$$
\boldsymbol m=(4,4,4,8),
\qquad
\boldsymbol n=(4,4,7,7)
$$

を得る。

$$
\prod_km_k=512,
\qquad
\prod_kn_k=784.
$$

combined modeは

$$
\boldsymbol q
=
(m_1n_1,m_2n_2,m_3n_3,m_4n_4)
=(16,16,28,56).
$$

Notebookで使った「各側を個別に均等化する」方法は、次のように実装できる。

~~~python
from __future__ import annotations

import math


def unordered_factorizations(
    value: int,
    num_modes: int,
    min_factor: int = 2,
) -> list[tuple[int, ...]]:
    """valueを非減少なnum_modes個の整数因子へ分解する。"""
    if value < 1:
        raise ValueError(f"value must be positive: {value}")
    if num_modes < 1:
        raise ValueError(f"num_modes must be positive: {num_modes}")
    if min_factor < 1:
        raise ValueError(f"min_factor must be positive: {min_factor}")

    def recurse(
        remaining: int,
        factors_left: int,
        lower_bound: int,
    ) -> list[tuple[int, ...]]:
        # 最後の因子は残りそのものになる
        if factors_left == 1:
            return [(remaining,)] if remaining >= lower_bound else []

        candidates: list[tuple[int, ...]] = []
        for factor in range(lower_bound, remaining + 1):
            if remaining % factor != 0:
                continue

            next_remaining = remaining // factor

            # 残る因子を全てfactor以上にできない枝は捨てる
            if next_remaining < factor ** (factors_left - 1):
                continue

            for suffix in recurse(
                next_remaining,
                factors_left - 1,
                factor,
            ):
                candidates.append((factor,) + suffix)

        return candidates

    return recurse(value, num_modes, min_factor)


def balance_score(
    factors: tuple[int, ...],
) -> tuple[float, int, int]:
    """辞書式比較用の均等度scoreを返す。"""
    smallest = min(factors)
    largest = max(factors)
    return (
        largest / smallest,
        largest - smallest,
        largest,
    )


def balanced_modes(
    value: int,
    num_modes: int,
    min_factor: int = 2,
) -> tuple[int, ...]:
    """積を保ちながら、なるべく均等な非減少因子列を返す。"""
    candidates = unordered_factorizations(
        value,
        num_modes,
        min_factor,
    )
    if not candidates:
        raise ValueError(
            f"cannot factorize {value} into {num_modes} modes"
        )

    best = min(candidates, key=balance_score)
    if math.prod(best) != value:
        raise RuntimeError("factorization product mismatch")
    return best
~~~

この関数は $\boldsymbol m$ と $\boldsymbol n$ を別々に均等化する。$\boldsymbol q=(m_kn_k)_k$、TT-rank、近似誤差まで含むjoint optimizationではない。

## 5. 再利用APIとNotebook内wrapper

Notebookは既存srcから次のfunctional APIを使う。

~~~python
from nn_compression.compression import (
    dense_to_tt_matrix_cores,
    tt_linear_forward,
    tt_matrix_num_parameters,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)
~~~

責務は次のように分かれる。

| API | 責務 |
| --- | --- |
| <code>dense_to_tt_matrix_cores</code> | dense重みをtensorizeし、TT-SVDして4階core列へ変換する |
| <code>tt_linear_forward</code> | dense重みを作らずにTT core列からLinear forwardを計算する |
| <code>tt_matrix_to_dense</code> | 検証用にdense重みを復元する |
| <code>tt_matrix_num_parameters</code> | core要素数の総和を返す |
| <code>tt_matrix_ranks</code> | 境界rankを含むrank列を返す |

srcには現時点で再利用可能な <code>TTLinear(nn.Module)</code> はない。そのためNotebook内で、functional APIを呼ぶ薄いwrapperを定義している。

変数 <code>tt_cores</code> はcore tensorのlistであり、それ自体はlayerではない。現段階では

$$
\left(
\text{tt core列},
\text{out modes},
\text{in modes},
\text{bias}
\right)
$$

を <code>tt_linear_forward</code> に渡す組がfunctionalなTTLinear表現であり、Notebook内wrapperがそれを <code>nn.Module</code> として束ねている。

現在のwrapperはcoreを

~~~python
self.cores = nn.ParameterList(
    [nn.Parameter(core.clone(), requires_grad=False) for core in cores]
)
~~~

として登録している。これはコメントにある「buffer」ではなく、正確には「勾配を要求しないParameter」である。biasだけが <code>register_buffer</code> で登録される。

この区別はstate管理とoptimizer対象を考えると重要である。

- <code>nn.Parameter(..., requires_grad=False)</code> はparameterとして登録される
- <code>register_buffer</code> はparameterではない状態として登録される
- どちらも通常は <code>state_dict()</code> に含まれる
- 将来fine-tuningするならcoreをtrainable Parameterにする必要がある

## 6. 打ち切りなしTT-SVDの結果

<code>max_rank=None</code> で全rankを保持した保存済み結果は、

~~~text
tt ranks: (1, 16, 256, 56, 1)
tt params: 470336
in/out: 784 512
~~~

である。

core shapeを全て書くと、

$$
\begin{aligned}
G^{(1)}&\in\mathbb{R}^{1\times4\times4\times16},\\
G^{(2)}&\in\mathbb{R}^{16\times4\times4\times256},\\
G^{(3)}&\in\mathbb{R}^{256\times4\times7\times56},\\
G^{(4)}&\in\mathbb{R}^{56\times8\times7\times1}.
\end{aligned}
$$

要素数は

$$
\begin{aligned}
N_{\mathrm{TT}}
&=
1\cdot4\cdot4\cdot16\\
&\quad+
16\cdot4\cdot4\cdot256\\
&\quad+
256\cdot4\cdot7\cdot56\\
&\quad+
56\cdot8\cdot7\cdot1\\
&=
470336.
\end{aligned}
$$

bias 512を含めると470,848である。元のdense層は401,920なので、このexact TTは圧縮になっていない。

## 7. denseモデルを壊さず一層だけ置換する

比較基準のdenseモデルを保持するため、

~~~python
tt_model = copy.deepcopy(dense_model)
~~~

としてから、copy側の <code>fc1</code> だけを置換する。

単純なモデルなら

~~~python
tt_model.fc1 = tt_layer
~~~

でよい。ネストしたmodule pathも扱うため、Notebookではpathを <code>.</code> で分割し、最後の名前の一つ手前まで <code>getattr</code> で辿ってから <code>setattr</code> する。

保存済み結果は、

~~~text
OK: dense_model unchanged; tt_model replaced only 'fc1'
dense target: Linear
tt target: TTLinear
~~~

である。

対象外moduleについて型の一致を検査し、対象外のLinearではweightとbiasが完全一致することも確認している。

## 8. 同一入力で段階別に比較する

入力は一つだけ生成し、両モデルへ同じtensorを渡す。

~~~text
x_batch.shape: (4, 1, 28, 28)
~~~

モデル内部のflatten後は

$$
X\in\mathbb{R}^{4\times784}
$$

となる。

比較した段階は次である。

$$
\begin{aligned}
Z_1&=\operatorname{fc1}(X),\\
H_1&=\operatorname{ReLU}_1(Z_1),\\
L&=\operatorname{fc3}\left(
\operatorname{ReLU}_2(
\operatorname{fc2}(H_1)
)
\right).
\end{aligned}
$$

保存済み最大絶対誤差は、

| 比較位置 | 最大絶対誤差 |
| --- | ---: |
| 対象Linear直後 | $1.3855583347321954\times10^{-13}$ |
| activation直後 | $9.725553695716371\times10^{-14}$ |
| logits | $1.0231815394945443\times10^{-12}$ |

である。全段階で

~~~python
torch.testing.assert_close(
    actual,
    expected,
    rtol=1e-11,
    atol=1e-12,
)
~~~

を通している。

## 9. predictionの補助確認

logitsのshapeは

$$
L\in\mathbb{R}^{B\times10}
$$

なので、sampleごとのclass indexは

~~~python
prediction = logits.argmax(dim=1)
~~~

で得る。<code>dim=0</code> はclassごとにbatch内のsampleを比較してしまうため誤りである。

softmaxはlogitの順序を保つので、class indexだけならsoftmaxを先に計算する必要はない。Notebookではdense側とTT側の <code>argmax(dim=1)</code> が完全一致することを確認している。

## 10. この検証から言えること

保存済み結果は、次を支持する。

- 学習済み <code>fc1</code> を打ち切りなしTT-matrixへ変換できた
- TT direct contractionがdense Linearと丸め誤差水準で一致した
- copy側の <code>fc1</code> だけを置換できた
- 同じ入力に対してLinear出力、activation、logitsが近接した
- prediction indexも一致した

## 11. この検証からは言えないこと

次はまだ示していない。

- parameter圧縮に成功したこと
- rank打ち切り後もlogitsやaccuracyを維持すること
- Fashion-MNIST test accuracy
- rank sweep
- fine-tuning後のaccuracy回復
- TT coreの学習可能性やgradient一致
- latency、MACs、memoryの改善
- 今回の教材化時点でFresh Run Allが通ったこと

従って到達点は、

$$
\boxed{
\text{学習済みMLP一層のexact TT置換とlogits等価性}
}
$$

であり、「Fashion-MNIST MLPのTT圧縮実験完了」ではない。

rank打ち切りによる層出力誤差のtoy検証は、[[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]]を参照する。PyTorchのモデル操作は、[[08_TT_MPS基礎実装検証/14_TTLinear置換で使うPyTorchモデル操作]]へ分ける。
