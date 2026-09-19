---
title: 混合正準形の中心摂動と等長性のPyTorch確認
tags:
  - TT
  - MPS
  - PyTorch
  - mixed-canonical
---

# 混合正準形の中心摂動と等長性のPyTorch確認

ここでいう混合正準形（mixed-canonical form）は左右を正規直交化したTT/MPS全体の配置、直交中心（orthogonality center）はその間にある中心コア $C$ を指す。用語と構成は [[00_基礎理論/01_数学基礎/02_テンソル代数/39_TT_MPSの混合正準形への導入]] を参照する。

## 確認する式

左右のブロックを固定し、左展開 $L\in\mathbb R^{n_1\times r_1}$ と右展開 $R\in\mathbb R^{r_2\times n_3}$ が

$$
L^TL=I_{r_1},\qquad RR^T=I_{r_2}
$$

を満たすとする。第2中心コア $C\in\mathbb R^{r_1\times n_2\times r_2}$ から全テンソルへの写像は

$$
X(i_1,i_2,i_3)
=\sum_{\alpha_1,\alpha_2}
L(i_1,\alpha_1)C(\alpha_1,i_2,\alpha_2)R(\alpha_2,i_3)
$$

である。同じ $L,R$ を固定して $C$ だけを $C+\Delta C$ へ変えると、

$$
\Delta X(i_1,i_2,i_3)
=\sum_{\alpha_1,\alpha_2}
L(i_1,\alpha_1)\Delta C(\alpha_1,i_2,\alpha_2)R(\alpha_2,i_3),
\qquad
\|\Delta X\|_F=\|\Delta C\|_F.
$$

これは「摂動が小さいときの近似」ではなく、固定した正規直交環境についての厳密な等式である。導出は [[00_基礎理論/01_数学基礎/02_テンソル代数/40_TT_MPSの中心ノルムと内積の導出]]、中心を移すQR操作は [[40_TensorTrain_MPS/01_直交中心移動のPyTorch確認]] に分ける。

## 決定的な全要素例

$n_1=n_3=3$、$r_1=r_2=n_2=2$ とし、

$$
L=\begin{pmatrix}
1/\sqrt2&0\\
1/\sqrt2&0\\
0&1
\end{pmatrix},
\qquad
R=\begin{pmatrix}
1/\sqrt2&1/\sqrt2&0\\
0&0&1
\end{pmatrix}
$$

と置く。列を掛け合わせれば $L^TL=I_2$、行を掛け合わせれば $RR^T=I_2$ である。中心コアの全要素を二つのsliceで

$$
C(:,1,:)=\begin{pmatrix}1&2\\0&1\end{pmatrix},
\qquad
C(:,2,:)=\begin{pmatrix}2&0\\1&-1\end{pmatrix}
$$

と指定する。摂動は8要素中の2要素だけを非ゼロにする。

$$
\Delta C(:,1,:)=\begin{pmatrix}0&0\\0.1&0\end{pmatrix},
\qquad
\Delta C(:,2,:)=\begin{pmatrix}0&-0.2\\0&0\end{pmatrix}.
$$

第1sliceの摂動は $L$ の第2列と $R$ の第1行の外積を $0.1$ 倍し、第2sliceは $L$ の第1列と $R$ の第2行の外積を $-0.2$ 倍する。積の全要素は

$$
\begin{aligned}
\Delta X(:,1,:)
&=L\Delta C(:,1,:)R
=\begin{pmatrix}
0&0&0\\
0&0&0\\
0.1/\sqrt2&0.1/\sqrt2&0
\end{pmatrix},\\
\Delta X(:,2,:)
&=L\Delta C(:,2,:)R
=\begin{pmatrix}
0&0&-0.2/\sqrt2\\
0&0&-0.2/\sqrt2\\
0&0&0
\end{pmatrix}.
\end{aligned}
$$

従って両側の二乗ノルムは要素ごとに

$$
\|\Delta C\|_F^2=0.1^2+(-0.2)^2=0.05,
\qquad
\|\Delta X\|_F^2
=2\left(\frac{0.1^2}{2}\right)
+2\left(\frac{0.2^2}{2}\right)
=0.05.
$$

以下は単独で実行できる確認コードである。`einsum` の `a` と `b` は消えるbond添字、`i,j,k` は残る物理添字に対応する。

```python
import torch

dtype = torch.float64
sqrt2 = 2.0**0.5

# 左の列と右の行がそれぞれ正規直交するように配置する。
left = torch.tensor(
    [[1.0 / sqrt2, 0.0], [1.0 / sqrt2, 0.0], [0.0, 1.0]],
    dtype=dtype,
)
right = torch.tensor(
    [[1.0 / sqrt2, 1.0 / sqrt2, 0.0], [0.0, 0.0, 1.0]],
    dtype=dtype,
)

# 軸順は (alpha1, i2, alpha2)。二つの physical slice の全要素を指定する。
center = torch.tensor(
    [[[1.0, 2.0], [2.0, 0.0]], [[0.0, 1.0], [1.0, -1.0]]],
    dtype=dtype,
)
delta_center = torch.zeros_like(center)
delta_center[1, 0, 0] = 0.1
delta_center[0, 1, 1] = -0.2

# 元の全テンソル、更新後の全テンソル、摂動だけの収縮を別々に作る。
full = torch.einsum("ia,ajb,bk->ijk", left, center, right)
full_after = torch.einsum(
    "ia,ajb,bk->ijk", left, center + delta_center, right
)
delta_full = torch.einsum(
    "ia,ajb,bk->ijk", left, delta_center, right
)

# Gram、線形性、全要素、ノルムを独立して確かめる。
assert torch.allclose(left.T @ left, torch.eye(2, dtype=dtype))
assert torch.allclose(right @ right.T, torch.eye(2, dtype=dtype))
assert torch.allclose(full_after - full, delta_full)
expected_first = torch.tensor(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
     [0.1 / sqrt2, 0.1 / sqrt2, 0.0]],
    dtype=dtype,
)
expected_second = torch.tensor(
    [[0.0, 0.0, -0.2 / sqrt2], [0.0, 0.0, -0.2 / sqrt2],
     [0.0, 0.0, 0.0]],
    dtype=dtype,
)
assert torch.allclose(delta_full[:, 0, :], expected_first)
assert torch.allclose(delta_full[:, 1, :], expected_second)
assert torch.allclose(
    torch.linalg.vector_norm(delta_full),
    torch.linalg.vector_norm(delta_center),
)
print("center and full squared norm:", delta_center.square().sum().item(),
      delta_full.square().sum().item())
```

## 乱数摂動の大きさと検証条件

`delta_center = 1e-3 * torch.randn_like(center)` は同じshape・dtype・deviceの要素を作る。各要素は平均0、標準偏差 $10^{-3}$ の正規分布に従い、$[-10^{-3},10^{-3}]$ に収まるという意味ではない。要素数を $N=r_1n_2r_2$ とすると、

$$
\mathbb E\|\Delta C\|_F^2=N(10^{-3})^2,
\qquad
\sqrt{\mathbb E\|\Delta C\|_F^2}=10^{-3}\sqrt N.
$$

右辺は二乗ノルムの期待値の平方根であり、個々の試行で得るノルムの固定値ではない。実際に「小さい」かどうかは $\|\Delta C\|_F/\|C\|_F$ で判定する。$C=0$ ならこの比は定義できない。正規直交な左右を固定する限り、乱数摂動でも $\|\Delta X\|_F=\|\Delta C\|_F$ の検証条件は同じである。

```python
# 上の決定的な例を実行した後、再現可能な乱数摂動でも確認する。
torch.manual_seed(0)
random_delta = 1e-3 * torch.randn_like(center)
random_full_delta = torch.einsum(
    "ia,ajb,bk->ijk", left, random_delta, right
)
relative_size = (
    torch.linalg.vector_norm(random_delta)
    / torch.linalg.vector_norm(center)
)
assert torch.allclose(
    torch.linalg.vector_norm(random_full_delta),
    torch.linalg.vector_norm(random_delta),
)
print("relative perturbation:", relative_size.item())
```

左右ブロックも同時に変える場合、あるいはGramが単位行列でない場合は、この固定環境の等長性をそのまま主張しない。`torch.linalg.vector_norm` は全要素を一列に見たEuclideanノルムであり、ここではテンソルのFrobeniusノルムに一致する。
