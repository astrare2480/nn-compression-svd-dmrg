---
title: TT-matrixのrank sweepと圧縮誤差のPyTorch確認
tags:
  - TT-matrix
  - TT-rank
  - rank-sweep
  - PyTorch
  - reconstruction-error
---

# TT-matrixのrank sweepと圧縮誤差のPyTorch確認

## 1. このノートの位置づけ

このノートは、[07_tt_rank_bond_dimension_sweep_weight_tradeoff.ipynb](../../notebooks/30_tt_mps/10_fashion_mnist_mlp/07_tt_rank_bond_dimension_sweep_weight_tradeoff.ipynb) で行った、3-core TT-matrixの小規模rank sweepを整理する。

理論は次の3ノートへ分ける。

- bond dimensionと表現能力：[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/69_TT-matrixのbond_dimensionと表現能力]]
- parameter count・compression ratio・再構成誤差：[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/70_TT-matrixのrank_sweepと圧縮率_再構成誤差]]
- TT-SVDの準最適性：[[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/47_TT-SVDの準最適性と誤差上界]]

この検証ではweightだけを扱い、Neural Networkのforward、prediction、accuracy、fine-tuningへは進まない。

## 2. 証拠レベル

現行Notebookには、各コードcellのexecution countと最終結果が保存されている。教材の穴埋め箇所にも値が入り、保存出力は理論上の期待値と整合する。

このノートへ記載する数値は、Notebookに保存された実行結果である。ここでは新しいkernelからのFresh Restart Kernel / Run Allは実施していない。そのため、次を区別する。

$$
\boxed{
\text{保存済み出力を確認した}
\ \neq\
\text{この監査時点でFresh Run Allした}
}
$$

## 3. 実験条件

Notebookの設定は次である。

~~~text
device: CPU
dtype: torch.float64
random seed: 0
dense weight shape: (8, 8)
m_modes: (2, 2, 2)
n_modes: (2, 2, 2)
requested ranks: (1, 2, 4)
~~~

全rank候補で同じweightを使う。

```python
import torch

# 専用Generatorを使い、同じweightを再現できるようにする。
generator = torch.Generator(device="cpu").manual_seed(0)
weight_dense = torch.randn(
    8,
    8,
    generator=generator,
    dtype=torch.float64,
    device="cpu",
)
```

rank以外まで変えると比較対象が変わるため、weight、tensorization、seed、dtype、deviceを固定する。

## 4. dense weightをTT-SVD用の軸順へ並べる

dense weightのrow indexとcolumn indexを、

$$
i=(i_1,i_2,i_3),
\qquad
j=(j_1,j_2,j_3)
$$

へ分ける。最初のreshapeでは、

$$
(m_1,m_2,m_3,n_1,n_2,n_3)
$$

の軸順になる。

TT-matrixの各siteでは $(m_k,n_k)$ を組にするため、

$$
(m_1,m_2,m_3,n_1,n_2,n_3)
\longrightarrow
(m_1,n_1,m_2,n_2,m_3,n_3)
$$

へpermuteする。

今回の各siteのphysical sizeは

$$
s_k=m_kn_k=2\cdot2=4
$$

なので、最後に

$$
(m_1,n_1,m_2,n_2,m_3,n_3)
\longrightarrow
(s_1,s_2,s_3)
=
(4,4,4)
$$

へreshapeする。

値を変更する操作ではなく、各要素へどの複合添字を割り当てるかを変更する操作である。

## 5. 第1段階のTT-SVD

3階テンソルを第1siteと残りに分け、

$$
M_1
\in
\mathbb R^{s_1\times(s_2s_3)}
=
\mathbb R^{4\times16}
$$

へreshapeする。

reduced SVDを

$$
M_1=U_1\Sigma_1V_1^{\mathsf T}
$$

とする。requested rankを $r$ とし、この段階で利用できる成分数を $p_1$ とすれば、実際に保持する数は

$$
r_1=\min(r,p_1)
$$

である。

第1coreは

$$
G^{(1)}
=
\operatorname{reshape}
\left(U_1[:,0:r_1],\ 1,m_1,n_1,r_1\right)
$$

であり、残りは

$$
C_1
=
\Sigma_1[0:r_1,0:r_1]
(V_1^{\mathsf T})[0:r_1,:]
$$

である。

shapeは

$$
G^{(1)}:(1,2,2,r_1),
\qquad
C_1:(r_1,16)
$$

となる。

## 6. 第2段階のTT-SVD

$C_1$ を

$$
M_2
=
\operatorname{reshape}(C_1,r_1s_2,s_3)
\in
\mathbb R^{(4r_1)\times4}
$$

へ変形し、

$$
M_2=U_2\Sigma_2V_2^{\mathsf T}
$$

とSVDする。実際に保持するrankを

$$
r_2=\min(r,p_2)
$$

とする。

第2coreと第3coreは、

$$
G^{(2)}
=
\operatorname{reshape}
\left(U_2[:,0:r_2],\ r_1,m_2,n_2,r_2\right),
$$

$$
G^{(3)}
=
\operatorname{reshape}
\left(
\Sigma_2[0:r_2,0:r_2]
(V_2^{\mathsf T})[0:r_2,:],
\ r_2,m_3,n_3,1
\right).
$$

最終shapeは

$$
G^{(1)}:(1,2,2,r_1),
$$

$$
G^{(2)}:(r_1,2,2,r_2),
$$

$$
G^{(3)}:(r_2,2,2,1)
$$

である。actual rank vectorはcore shapeから

$$
(1,r_1,r_2,1)
$$

と読む。

## 7. dense reconstructionの軸順

3つのcoreのbondを縮約すると、最初に

$$
(m_1,n_1,m_2,n_2,m_3,n_3)
$$

の軸順を持つテンソルが得られる。

これを

$$
(m_1,m_2,m_3,n_1,n_2,n_3)
$$

へpermuteし、最後に

$$
\left(
\prod_{k=1}^{3}m_k,
\prod_{k=1}^{3}n_k
\right)
=
(8,8)
$$

へreshapeする。

分解時と再構成時のpermuteが逆操作になっていなければ、rank 4でも元のweightへ戻らない。rank sweepの前に、shapeだけでなく要素の添字順を揃える必要がある。

## 8. parameter countはactual coreから計算する

理論上の指定shapeからは、

$$
N_{\mathrm{TT}}=8r+4r^2
$$

を予想できる。実験結果では、実際に得たcoreの要素数を使う。

```python
def tt_parameter_count(cores: tuple[torch.Tensor, ...]) -> int:
    """全TT coreに保存するスカラー数を返す。"""
    return sum(core.numel() for core in cores)
```

requested rankがshape上の最大値を超えた場合でも、actual core shapeに基づく保存量を記録できる。

## 9. rank sweepで記録するrecord

各rank候補に対し、少なくとも次を辞書へ保存する。

```python
record = {
    "requested_rank": requested_rank,
    "actual_rank_vector": actual_rank_vector,
    "TT params": tt_params,
    "compression ratio": compression_ratio,
    "absolute Frobenius reconstruction error": absolute_error,
    "relative Frobenius reconstruction error": relative_error,
    "parameter count compressed": tt_params < dense_params,
}
```

relative errorの分母へ小さい定数を足して定義を変更するのではなく、先に

```python
# 今回は非零weightなので、そのことを明示的に確認する。
weight_norm = torch.linalg.matrix_norm(weight_dense, ord="fro")
assert weight_norm.item() > 0.0

relative_error = absolute_error / weight_norm
```

と確認する。

## 10. `records[1:]`、`zip`、`all` の読み方

保存量がrank候補の順に増えたかを、次の式で確認できる。

```python
params_increase_with_rank = all(
    a["TT params"] < b["TT params"]
    for a, b in zip(records, records[1:])
)
```

例えば、

```python
records = [
    {"rank": 1, "TT params": 12},
    {"rank": 2, "TT params": 32},
    {"rank": 4, "TT params": 96},
]
```

とする。

`records[1:]` は先頭を除いた

```python
[
    {"rank": 2, "TT params": 32},
    {"rank": 4, "TT params": 96},
]
```

である。従って、

```python
zip(records, records[1:])
```

は隣接する2組を作る。

```text
1回目:
  a = rank 1, TT params 12
  b = rank 2, TT params 32

2回目:
  a = rank 2, TT params 32
  b = rank 4, TT params 96
```

generator式では、`for a, b in ...` が毎回 `a` と `b` を更新し、その後に

```python
a["TT params"] < b["TT params"]
```

を評価する。`records` 自体を書き換えているのではない。

通常のfor文へ展開すると、

```python
# 最初は、すべての隣接比較が成立すると仮定する。
params_increase_with_rank = True

for a, b in zip(records, records[1:]):
    # 次のrecordでparameter数が増えていなければ判定をFalseにする。
    if not (a["TT params"] < b["TT params"]):
        params_increase_with_rank = False
        break
```

となる。`all(...)` は各比較がすべて `True` のときだけ `True` を返す。

この判定は `records` がrequested rankの昇順に並んでいることを前提とする。式自体はrank列をsortしない。

## 11. 実測誤差の単調性は観察として扱う

同じ方法で、今回の誤差が非増加だったかを確認することはできる。

```python
observed_non_increasing = all(
    a["absolute Frobenius reconstruction error"]
    >= b["absolute Frobenius reconstruction error"]
    for a, b in zip(records, records[1:])
)
```

ただし、これをmulti-core TT-SVD一般の理論assertにはしない。このbooleanは今回の保存結果を要約する観察値である。最良TT近似誤差の単調性と、逐次TT-SVDの実測誤差の単調性を区別する。

## 12. 保存済み結果

Notebookの保存出力は次である。

$$
\begin{array}{c|c|r|r|r|r|c}
\text{requested}
&\text{actual rank vector}
&N_{\mathrm{TT}}
&C
&E_{\mathrm{abs}}
&E_{\mathrm{rel}}
&\text{compressed}\\ \hline
1&(1,1,1,1)&12&5.333333&7.201489&0.8298814&\text{True}\\
2&(1,2,2,1)&32&2.000000&5.420826&0.6246823&\text{True}\\
4&(1,4,4,1)&96&0.666667&6.301471\times10^{-15}&7.261656\times10^{-16}&\text{False}
\end{array}
$$

保存された誤差列は

~~~text
[7.201489362626453,
 5.420826316238032,
 6.301470836549541e-15]
~~~

であり、この実験では非増加だった。

## 13. 結果の考察

### 13.1 requested rank 1

$$
N_{\mathrm{TT}}=12,
\qquad
C\approx5.33
$$

で、parameter countは大きく減る。一方、relative errorは約0.83であり、元のrandom weightの多くを再構成できていない。

### 13.2 requested rank 2

$$
N_{\mathrm{TT}}=32,
\qquad
C=2
$$

となり、denseの半分の要素数である。rank 1より保存量を増やした結果、このweightではabsolute／relative errorが小さくなった。

### 13.3 requested rank 4

両cutの最大rankは4なので、非零特異成分をshape上すべて保持できる。実測誤差

$$
6.30\times10^{-15}
$$

はfloat64の丸め誤差水準であり、数値的にexact reconstructionである。

しかし、

$$
N_{\mathrm{TT}}=96
>
64=N_{\mathrm{dense}}
$$

なので、parameter count上は圧縮ではない。

$$
\boxed{
\text{exact reconstruction}
\ \not\Rightarrow\
\text{parameter compression}
}
$$

この例は、rankを上げると表現能力は増える一方、保存量も増え、完全再構成点ではTT表現の方が大きくなりうることを示す。

## 14. 確認できたこと

この小規模検証で確認したのは、次である。

1. tensorizationとweightを固定し、requested rankだけを変えた。
2. requested rankとactual rank vectorを区別した。
3. core shapeの式と、`numel()`の合計が一致した。
4. compression ratioをdense params / TT paramsで計算した。
5. absolute／relative Frobenius reconstruction errorを記録した。
6. rank 4で数値的exact reconstructionになった。
7. exact reconstructionでもparameter count上は非圧縮になりうることを確認した。
8. 実測誤差の非増加を今回の観察として扱い、一般保証とは書かなかった。

Fashion-MNISTの学習済み重み、prediction agreement、classification accuracy、fine-tuningはこの検証範囲に含まれない。
