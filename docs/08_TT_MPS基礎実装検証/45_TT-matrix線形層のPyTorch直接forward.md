---
title: TT-matrix線形層のPyTorch直接forward
tags:
  - TT-matrix
  - MPO
  - PyTorch
  - Linear
---

# TT-matrix線形層のPyTorch直接forward

[[17_TT_MPS学習ロードマップ]] では、線形層の重みをTT-matrixへ変換するためのmode分解、paired順、コアshape、格納量を扱った。本章では、そのコアから密な重み行列を作らずに

$$
y=Wx+b
$$

を計算するPyTorch上の添字対応だけを扱う。通常のTT tensorの分解・再構成は [[27_TT_MPS基礎のPyTorch実装]]、`reshape` と `permute` の順序は [[29_TT_cutとPyTorchのreshape_Kronecker順序]] を参照する。

## 1. TT tensorとTT-matrixを区別する

入力・出力次元を

$$
N=\prod_{k=1}^{d}n_k,
\qquad
M=\prod_{k=1}^{d}m_k
$$

と分解する。TT-matrixの第 $k$ コアは入力脚と出力脚を一つずつ持つ。

$$
G^{(k)}\in
\mathbb R^{r_{k-1}\times m_k\times n_k\times r_k},
\qquad r_0=r_d=1.
$$

重みの各要素は

$$
W_{(o_1,\ldots,o_d),(i_1,\ldots,i_d)}
=\sum_{\alpha_1,\ldots,\alpha_{d-1}}
G^{(1)}_{1,o_1,i_1,\alpha_1}
\cdots
G^{(d)}_{\alpha_{d-1},o_d,i_d,1}
$$

である。3階TTコア $(r_{k-1},n_k,r_k)$ に軸を一つ加えたのではなく、線形写像の入力脚 $n_k$ と出力脚 $m_k$ を明示した4階コアである。

batch添字を $b$ とすれば、直接forwardは

$$
\begin{aligned}
y_{b,o_1,\ldots,o_d}
&=\sum_{i_1,\ldots,i_d}
W_{(o_1,\ldots,o_d),(i_1,\ldots,i_d)}
x_{b,i_1,\ldots,i_d}\\
&=\sum_{\substack{i_1,\ldots,i_d\\
\alpha_1,\ldots,\alpha_{d-1}}}
x_{b,i_1,\ldots,i_d}
G^{(1)}_{1,o_1,i_1,\alpha_1}
\cdots
G^{(d)}_{\alpha_{d-1},o_d,i_d,1}
\end{aligned}
$$

を計算することに対応する。

## 2. 単純な逐次loopで軸位置を誤らない

最初のコアを縮約した直後の中間Tensorは、例えば

$$
(b,r_1,m_1,n_2,\ldots,n_d)
$$

のように、確定した出力軸 $m_1$ と未処理の入力軸 $n_2,ldots,n_d$ を同時に持つ。次の反復で中間Tensorを無条件に

```text
(batch, left_rank, current_input, ...)
```

と読むと、第3軸の $m_1$ を $n_2$ と誤認する。逐次実装では各反復後の `permute` とshapeを明示する必要がある。本章では、添字を一度に指定する `torch.einsum` により、この曖昧さを避ける。

## 3. 一般のサイト数に対応する実装

`torch.einsum` のsublist形式では、各軸へ整数ラベルを割り当てられる。同じラベルは縮約し、出力リストに残したラベルだけが結果の軸になる。

```python
import math

import torch
import torch.nn as nn


class TTMatrixLinear(nn.Module):
    """密な重み行列を構成せず、TT-matrixコアから線形写像を計算する。"""

    def __init__(
        self,
        in_modes,
        out_modes,
        tt_ranks,
        bias=True,
        dtype=torch.float32,
    ):
        super().__init__()

        self.in_modes = tuple(in_modes)
        self.out_modes = tuple(out_modes)
        self.tt_ranks = tuple(tt_ranks)
        self.d = len(self.in_modes)

        if self.d == 0 or self.d != len(self.out_modes):
            raise ValueError("in_modesとout_modesは同じ正の長さにする")
        # einsumの整数ラベルは0以上52未満なので、この割当てではd <= 16。
        if self.d > 16:
            raise ValueError("この実装例のeinsumラベル割当てはd <= 16に対応する")
        if len(self.tt_ranks) != self.d + 1:
            raise ValueError("tt_ranksの長さはd + 1にする")
        if self.tt_ranks[0] != 1 or self.tt_ranks[-1] != 1:
            raise ValueError("開放境界では先頭rankと末尾rankを1にする")
        if any(v <= 0 for v in (*self.in_modes, *self.out_modes, *self.tt_ranks)):
            raise ValueError("mode sizeとTT-rankは正の整数にする")

        self.in_features = math.prod(self.in_modes)
        self.out_features = math.prod(self.out_modes)

        # 各コアの軸順は(left bond, output mode, input mode, right bond)。
        self.cores = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.tt_ranks[k],
                        self.out_modes[k],
                        self.in_modes[k],
                        self.tt_ranks[k + 1],
                        dtype=dtype,
                    )
                )
                for k in range(self.d)
            ]
        )

        # これはfrom-scratch学習用の一般的な初期値であり、
        # 学習済みdense重みのTT-SVD初期化ではない。
        for core in self.cores:
            nn.init.xavier_uniform_(core)

        self.bias = (
            nn.Parameter(torch.zeros(self.out_features, dtype=dtype))
            if bias
            else None
        )

    def forward(self, x):
        """xのshapeは(batch, in_features)とする。"""
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"x.shapeは(batch, {self.in_features})にする: got {tuple(x.shape)}"
            )

        # flattened inputを(i1, ..., id)へ戻す。
        z = x.reshape(x.shape[0], *self.in_modes)

        # einsumの各軸へ重複しない整数ラベルを割り当てる。
        batch_label = 0
        input_labels = list(range(1, self.d + 1))
        output_labels = list(range(self.d + 1, 2 * self.d + 1))
        bond_labels = list(range(2 * self.d + 1, 3 * self.d + 2))

        # torch.einsum(z, z_labels, core1, core1_labels, ..., output_labels)
        operands = [z, [batch_label, *input_labels]]
        for k, core in enumerate(self.cores):
            operands.extend(
                [
                    core,
                    [
                        bond_labels[k],
                        output_labels[k],
                        input_labels[k],
                        bond_labels[k + 1],
                    ],
                ]
            )
        operands.append([batch_label, *output_labels])

        y = torch.einsum(*operands)
        y = y.reshape(x.shape[0], self.out_features)

        if self.bias is not None:
            y = y + self.bias
        return y
```

境界bondのラベルも出力リストにはないため和で消えるが、そのsizeは1である。出力の軸順は

$$
(b,o_1,o_2,\ldots,o_d)
$$

と明示しているので、最後の `reshape` はこの順序を保ってflattenする。

## 4. 2サイトの全要素例

$$
(n_1,n_2)=(2,2),qquad
(m_1,m_2)=(2,2),qquad
(r_0,r_1,r_2)=(1,2,1)
$$

とする。二つのコアを、左端と右端のsize 1軸も含めて

$$
G^{(1)}=
\left[
\begin{array}{cc|cc}
1&0&0&1\\
2&1&1&-1
\end{array}
\right]
\in\mathbb R^{1\times2\times2\times2},
$$

$$
G^{(2)}
=\frac18
\left[
\begin{array}{cc|cc}
1&2&3&4\\
5&6&7&8
\end{array}
\right]
\in\mathbb R^{2\times2\times2\times1}
$$

とする。第1コアの表示では各行が $o_1$、縦線の左側と右側が $i_1=1,2$ の順で、各組の2要素が $\alpha_1=1,2$ である。第2コアでは各行が $\alpha_1$、各行の並びが $(o_2,i_2)=(1,1),(1,2),(2,1),(2,2)$ である。

$$
W_{(o_1,o_2),(i_1,i_2)}
=\sum_{\alpha_1=1}^{2}
G^{(1)}_{1,o_1,i_1,\alpha_1}
G^{(2)}_{\alpha_1,o_2,i_2,1}
$$

を全要素について計算すると

$$
W=
\begin{pmatrix}
0.125&0.250&0.625&0.750\\
0.375&0.500&0.875&1.000\\
0.875&1.250&-0.500&-0.500\\
1.625&2.000&-0.500&-0.500
\end{pmatrix}.
$$

例えば第1行第3列は $(o_1,o_2)=(1,1)$、$(i_1,i_2)=(2,1)$ なので

$$
W_{(1,1),(2,1)}
=0\cdot\frac18+1\cdot\frac58
=0.625
$$

である。次のコードは、TTコアからの直接forwardと密行列による積を同じ入力で比較する。

```python
import torch


torch.set_default_dtype(torch.float64)

layer = TTMatrixLinear(
    in_modes=(2, 2),
    out_modes=(2, 2),
    tt_ranks=(1, 2, 1),
    bias=False,
    dtype=torch.float64,
)

G1 = torch.tensor(
    [[[[1.0, 0.0], [0.0, 1.0]],
      [[2.0, 1.0], [1.0, -1.0]]]]
)
G2 = torch.arange(1.0, 9.0).reshape(2, 2, 2, 1) / 8.0

with torch.no_grad():
    layer.cores[0].copy_(G1)
    layer.cores[1].copy_(G2)

# 参照用にだけ密なWを作る。実際のforwardではWを作っていない。
W_tensor = torch.einsum("aoir,rpjs->opij", G1, G2)
W_dense = W_tensor.reshape(4, 4)

x = torch.tensor(
    [
        [1.0, 2.0, 3.0, 4.0],
        [-1.0, 0.0, 1.0, 2.0],
    ]
)

y_tt = layer(x)
y_dense = x @ W_dense.T

print("W_dense =")
print(W_dense)
print("y_tt =")
print(y_tt)
print("y_dense =")
print(y_dense)
print("max abs error =", torch.max(torch.abs(y_tt - y_dense)).item())

assert torch.allclose(y_tt, y_dense, atol=1e-12, rtol=1e-12)
```

期待値は

$$
y_{\mathrm{TT}}=y_{\mathrm{dense}}
=\begin{pmatrix}
5.5&8.0&-0.125&2.125\\
2.0&2.5&-2.375&-3.125
\end{pmatrix}
$$

である。この一致はforwardの添字順を検証するものであり、学習精度や速度を示す実験結果ではない。

## 5. 初期化・圧縮・速度を同一視しない

- 上のXavier初期化はTT層を最初から学習する場合の出発点であり、学習済みdense重みを近似していない。
- post-training圧縮では、dense重みをpaired順へ並べ替えてTT-SVDし、得たコアを `copy_` する別処理が必要である。
- exact rank、指定した最大rank、実際に得た各bond rankを区別する。
- TTコアの保存要素数が少なくても、小規模層では縮約やkernel起動のオーバーヘッドによりdense GEMMより遅くなり得る。
- 密な $W$ を再構成してから `x @ W.T` を行う検証版は、理論上の保存量は確認できても、直接forwardの実メモリ・速度を検証したことにはならない。
- 比較では重み誤差、層出力誤差、タスク精度、fine-tuning後の回復、parameter数、実測時間を別々に記録する。

この段階はPyTorchの縮約方法を固定しただけであり、TT-SVD初期化、rank sweep、Fashion-MNIST学習を実行した結果ではない。
