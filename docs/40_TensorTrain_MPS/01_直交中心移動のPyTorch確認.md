---
title: 直交中心移動のPyTorch確認
aliases:
  - orthogonality centerのPyTorch操作
tags:
  - TT
  - MPS
  - PyTorch
  - QR
---

# 直交中心移動のPyTorch確認

## このノートの位置づけ

[[00_基礎理論/01_数学基礎/02_テンソル代数/41_TT_MPSの直交中心の移動]] の小さい例を、PyTorchの `reshape`、`torch.linalg.qr`、`torch.tensordot` で確認するための実装ノートである。TT-SVDや圧縮rankの選択、学習Notebookの演習解答、既存 `src` の公開APIを追加するものではない。中心ノルムの理論は [[00_基礎理論/01_数学基礎/02_テンソル代数/40_TT_MPSの中心ノルムと内積の導出]] を参照する。

## 行列化とshapeの対応

第2中心コア $C$ のshapeを $(r_1,n_2,r_2)$ とする。右へ中心を動かす左展開は、$(\alpha_1,i_2)$ を行、$\alpha_2$ を列にまとめる。

$$
C_L((\alpha_1,i_2),\alpha_2)=C(\alpha_1,i_2,\alpha_2),
\qquad C_L\in\mathbb R^{(r_1n_2)\times r_2}.
$$

`C.reshape(r1 * n2, r2)` の行位置は、0始まりで `alpha1 * n2 + i2` となる。QRの `Q_L` を `(r1, n2, q2)` に戻し、`T_L` の列軸と第3コアの左ボンド軸を収縮する。

左へ中心を動かす右展開は、$\alpha_1$ を行、$(i_2,\alpha_2)$ を列にする。

$$
C_R(\alpha_1,(i_2,\alpha_2))=C(\alpha_1,i_2,\alpha_2),
\qquad C_R\in\mathbb R^{r_1\times(n_2r_2)}.
$$

`C.reshape(r1, n2 * r2)` の列位置は `i2 * r2 + alpha2` である。右直交な行を得るため `C_R.T` をQR分解し、`Q_R.T` を `(q1, n2, r2)` に戻す。`T_R.T` は第1コアの右ボンド軸へ吸収する。

### 左右の`reshape`を同じ18要素で比較する

例えば $(r_1,n_2,r_2)=(2,3,3)$ とし、0始まりの要素を $C(\alpha_1,i_2,\alpha_2)=9\alpha_1+3i_2+\alpha_2$ と置く。左展開では $(\alpha_1,i_2)$ が行になり、

$$
C_L=\begin{pmatrix}
0&1&2\\
3&4&5\\
6&7&8\\
9&10&11\\
12&13&14\\
15&16&17
\end{pmatrix}\in\mathbb R^{6\times3}.
$$

右展開では同じ値を $(i_2,\alpha_2)$ の列へ並べ、

$$
C_R=\begin{pmatrix}
0&1&2&3&4&5&6&7&8\\
9&10&11&12&13&14&15&16&17
\end{pmatrix}\in\mathbb R^{2\times9}.
$$

例えば $C(1,2,0)=15$ は $C_L(5,0)=15$ かつ $C_R(1,6)=15$ である。行位置 $5=1\cdot3+2$、列位置 $6=2\cdot3+0$ なので、左右でまとめる軸だけが異なる。どちらも`permute`はせず、18個の値と論理的な格納順を保った`reshape`である。

```python
import torch

# 0始まりの値を全要素に割り当て、左右の展開位置を確認する。
center = torch.arange(18).reshape(2, 3, 3)
center_left = center.reshape(2 * 3, 3)
center_right = center.reshape(2, 3 * 3)

assert center_left.shape == (6, 3)
assert center_right.shape == (2, 9)
assert center[1, 2, 0].item() == center_left[5, 0].item() == center_right[1, 6].item() == 15
assert torch.equal(center.flatten(), center_left.flatten())
assert torch.equal(center.flatten(), center_right.flatten())
```

## 小さい全要素例の数値確認

以下は単独で実行できる確認用コードである。最初の `G1` と `G3` は境界軸を持つ単位行列、第2コアは理論ノートと同じ $(2,2,2)$ の全要素例を使う。`float64` は丸め誤差を見やすくするためであり、QRの符号の一致を要求するためではない。

```python
import torch


def reconstruct(g1: torch.Tensor, g2: torch.Tensor, g3: torch.Tensor) -> torch.Tensor:
    """3階TTをボンド軸で縮約し、境界のサイズ1軸だけを除く。"""
    first_two = torch.tensordot(g1, g2, dims=([-1], [0]))
    full = torch.tensordot(first_two, g3, dims=([-1], [0]))
    return full.squeeze(0).squeeze(-1)


dtype = torch.float64
identity = torch.eye(2, dtype=dtype)
swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)

# 境界軸を加え、(1, 2, 2), (2, 2, 2), (2, 2, 1) を作る。
g1_left = identity.unsqueeze(0)
center = torch.stack((identity, swap), dim=1)
g3_right = identity.unsqueeze(-1)
original = reconstruct(g1_left, center, g3_right)

r1, n2, r2 = center.shape

# 右移動: (alpha1, i2) を行、alpha2 を列にする。
center_left = center.reshape(r1 * n2, r2)
q_left, t_left = torch.linalg.qr(center_left, mode="reduced")
new_r2 = q_left.shape[1]
g2_left = q_left.reshape(r1, n2, new_r2)
g3_center = torch.tensordot(t_left, g3_right, dims=([1], [0]))
after_right_move = reconstruct(g1_left, g2_left, g3_center)

# 左移動: alpha1 を行、(i2, alpha2) を列にする。
center_right = center.reshape(r1, n2 * r2)
q_right, t_right = torch.linalg.qr(center_right.T, mode="reduced")
new_r1 = q_right.shape[1]
g2_right = q_right.T.reshape(new_r1, n2, r2)
g1_center = torch.tensordot(g1_left, t_right.T, dims=([2], [0]))
after_left_move = reconstruct(g1_center, g2_right, g3_right)

# Q の向きごとに直交性を確認する。比較単位行列はdtype/deviceを揃える。
torch.testing.assert_close(
    q_left.T @ q_left,
    torch.eye(new_r2, dtype=q_left.dtype, device=q_left.device),
)
torch.testing.assert_close(
    q_right.T @ q_right,
    torch.eye(new_r1, dtype=q_right.dtype, device=q_right.device),
)
torch.testing.assert_close(after_right_move, original)
torch.testing.assert_close(after_left_move, original)

# 全要素の期待値と、中心へノルムが集まることを別々に確かめる。
torch.testing.assert_close(original[:, 0, :], identity)
torch.testing.assert_close(original[:, 1, :], swap)
torch.testing.assert_close(torch.linalg.vector_norm(original), torch.linalg.vector_norm(g3_center))
torch.testing.assert_close(torch.linalg.vector_norm(original), torch.linalg.vector_norm(g1_center))

print("C_L =", center_left)
print("C_R =", center_right)
print("right move error =", torch.linalg.vector_norm(after_right_move - original).item())
print("left move error =", torch.linalg.vector_norm(after_left_move - original).item())
```

`Q_R.T` の行Gramは `Q_R.T @ Q_R = I` であり、コードでは同じ行列積 `q_right.T @ q_right` を確認している。元の `G2` のshapeと新しいコアのshapeがたまたま一致する例でも、`q_left.shape[1]` や `q_right.shape[1]` を用いる。一般の横長行列ではreduced QRの列数が元のボンド次元と異なる場合があるためである。

全体の再構成誤差と局所Gram誤差は別の確認項目である。QR符号や丸め値だけの一致では合否を決めない。既存の `src` のTT-SVD関数の契約、dtype、rank検証は [[00_基礎理論/05_PyTorch実装/27_TT_MPS基礎のPyTorch実装]] を参照する。
