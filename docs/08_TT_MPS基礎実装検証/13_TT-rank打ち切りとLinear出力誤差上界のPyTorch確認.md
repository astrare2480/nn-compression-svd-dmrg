---
title: TT-rank打ち切りとLinear出力誤差上界のPyTorch確認
tags:
  - TT-matrix
  - TT-rank
  - PyTorch
  - Frobenius-norm
  - spectral-norm
  - error-bound
---

# TT-rank打ち切りとLinear出力誤差上界のPyTorch確認

## このノートの位置づけ

このノートは、[01_ttlinear_output_error_norm_bounds.ipynb](../../notebooks/30_tt_mps/10_fashion_mnist_mlp/01_ttlinear_output_error_norm_bounds.ipynb) の2-core toy例と保存済み数値結果を整理する。

式の導出は、[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/60_TT-rank打ち切りによるLinear出力誤差とnorm上界]]を参照する。

## 1. 証拠レベル

Notebookには最終結果の保存出力がある。しかし、現行ファイルでは重み誤差を計算するcellと出力誤差恒等式を計算するcellの <code>execution_count</code> がnullであり、その後の上界cellにはexecution countと出力が残っている。

従って、保存出力は過去の実行状態を示すが、現行ソースを新しいkernelで上からRun Allした証明にはならない。このノートでは「保存済み結果」として記録し、Fresh Run All済みとは書かない。

## 2. 実験条件

~~~text
device: CPU
dtype: torch.float64
torch.manual_seed(0)

batch size B = 5
input dimension n = 12 = 3 x 4
output dimension h = 20 = 4 x 5
truncated TT rank = 2
~~~

modeは

$$
(n_1,n_2)=(3,4),
\qquad
(m_1,m_2)=(4,5)
$$

である。

## 3. tensorization

dense重み

$$
W\in\mathbb{R}^{20\times12}
$$

をまず

$$
W_{\mathrm{grouped}}
\in
\mathbb{R}^{4\times5\times3\times4}
$$

へreshapeし、その後

$$
\mathcal W
\in
\mathbb{R}^{4\times3\times5\times4}
$$

へpermuteする。

要素対応は

$$
\mathcal W[i_1,j_1,i_2,j_2]
=
W[5i_1+i_2,\ 4j_1+j_2].
$$

2-core TT-SVD用のunfoldingは

$$
M
\in
\mathbb{R}^{(m_1n_1)\times(m_2n_2)}
=
\mathbb{R}^{12\times20}
$$

である。

## 4. rank 2 TT-SVD

reduced SVDを

$$
M=U\Sigma V^{\mathsf T}
$$

とし、上位2成分だけを保持する。

$$
U_2=U[:,0:2],
\qquad
\Sigma_2=\Sigma[0:2,0:2],
\qquad
V_2^{\mathsf T}=V^{\mathsf T}[0:2,:].
$$

coreは

$$
G^{(1)}
=
\operatorname{reshape}(U_2,1,4,3,2),
$$

$$
G^{(2)}
=
\operatorname{reshape}(\Sigma_2V_2^{\mathsf T},2,5,4,1)
$$

である。保存済みshapeは、

~~~text
core_1.shape = (1, 4, 3, 2)
core_2.shape = (2, 5, 4, 1)
~~~

である。

## 5. dense reconstructionとdirect contraction

検証用dense reconstructionは

$$
\widetilde{\mathcal W}[i_1,j_1,i_2,j_2]
=
\sum_{\alpha=0}^{1}
G^{(1)}[0,i_1,j_1,\alpha]
G^{(2)}[\alpha,i_2,j_2,0]
$$

を計算し、

$$
\widetilde W[5i_1+i_2,\ 4j_1+j_2]
=
\widetilde{\mathcal W}[i_1,j_1,i_2,j_2]
$$

へ戻す。

direct contractionは

$$
\begin{aligned}
\widetilde Y[b,i_1,i_2]
&=
\sum_{j_1=0}^{2}
\sum_{j_2=0}^{3}
\sum_{\alpha=0}^{1}
X[b,j_1,j_2]\\
&\qquad\qquad\cdot
G^{(1)}[0,i_1,j_1,\alpha]
G^{(2)}[\alpha,i_2,j_2,0]
+b[5i_1+i_2]
\end{aligned}
$$

を、dense重みを作らずに計算する。Notebookではdirect contractionと、復元した $\widetilde W$ を使う <code>F.linear</code> の一致を検査している。

## 6. 誤差恒等式の検証

コード上では

~~~python
delta_w = weight_tt - weight_dense
delta_y_direct = y_tt - y_dense
delta_y_formula = x @ delta_w.T

torch.testing.assert_close(
    delta_y_direct,
    delta_y_formula,
    rtol=rtol,
    atol=atol,
)
~~~

として、

$$
\Delta Y
=
\widetilde Y-Y
=
X\Delta W^{\mathsf T}
$$

を検査する。

保存済み結果では、

$$
\max_{b,i}
\left|
(\Delta Y_{\mathrm{direct}})_{b,i}
-
(\Delta Y_{\mathrm{formula}})_{b,i}
\right|
=
4.440892\times10^{-16}.
$$

## 7. norm APIの対応

数学とPyTorch APIを明示的に対応させる。

| 数学 | PyTorch |
| --- | --- |
| $\|\Delta W\|_F$ | <code>torch.linalg.matrix_norm(delta_w, ord="fro")</code> |
| $\|\Delta W\|_2$ | <code>torch.linalg.matrix_norm(delta_w, ord=2)</code> |
| $\|X\|_F$ | <code>torch.linalg.vector_norm(x)</code> またはmatrix Frobenius norm |
| $\|X\|_2$ | <code>torch.linalg.matrix_norm(x, ord=2)</code> |
| $\sigma_k(\Delta W)$ | <code>torch.linalg.svdvals(delta_w)</code> |

2階tensorをflattenしたvector 2-normは数値的にFrobenius normと等しい。しかし教材ではvector norm、matrix Frobenius norm、matrix spectral normを区別するため、重みと出力には <code>matrix_norm(..., ord="fro")</code> を使う。

例えば

$$
A=
\begin{pmatrix}
3&0\\
0&4
\end{pmatrix}
$$

では、

~~~python
a = torch.tensor(
    [[3.0, 0.0], [0.0, 4.0]],
    dtype=torch.float64,
)

spectral = torch.linalg.matrix_norm(a, ord=2)
frobenius = torch.linalg.matrix_norm(a, ord="fro")

torch.testing.assert_close(
    spectral,
    torch.tensor(4.0, dtype=a.dtype),
)
torch.testing.assert_close(
    frobenius,
    torch.tensor(5.0, dtype=a.dtype),
)
~~~

となる。最大特異値だけを見るspectral normは4、全特異値の二乗和を見るFrobenius normは5である。

## 8. 保存済み数値結果

shapeは次である。

~~~text
X.shape       = (5, 12)
W.shape       = (20, 12)
W_tt.shape    = (20, 12)
Delta_W.shape = (20, 12)
Y.shape       = (5, 20)
Delta_Y.shape = (5, 20)
~~~

重み誤差は、

~~~text
||Delta_W||_F         = 2.010011e+00
||Delta_W||_2         = 9.661027e-01
relative_weight_error = 8.047474e-01
~~~

である。

層出力誤差は、

~~~text
||Delta_Y||_F = 4.814268e+00
~~~

である。

二つの上界は、

~~~text
||X||_F ||Delta_W||_2 = 8.592132e+00
||X||_2 ||Delta_W||_F = 1.171245e+01
~~~

となった。

差と比率は、

~~~text
slack_1 = 3.777864e+00
slack_2 = 6.898180e+00
ratio_1 = 5.603112e-01
ratio_2 = 4.110386e-01
~~~

である。従って、

$$
4.814268
\le
8.592132,
$$

$$
4.814268
\le
11.71245
$$

が保存出力上で確認されている。

## 9. 数値の読み方

このtoy例ではrelative weight errorが約0.805と大きい。これはrank 2が高品質な圧縮設定であるという結果ではなく、非零誤差を持つ例で恒等式と上界を確認するための設定である。

また、上界が実測誤差より大きいのは正常である。spectral normは任意入力方向に対する最大gainを使うため、random batchが最悪方向だけで構成されていなければ余裕が生じる。

## 10. 現行Notebookソースの注意

静的確認では、教材コードに次の注意が残る。

1. tensorization関数は無効入力を <code>ValueError</code> で拒否すると説明しているが、一部を <code>assert</code> と <code>print</code> で処理している。再利用関数にするなら明示的な例外へ揃える。
2. 2-core TT-SVD関数のreshapeが引数のmode列ではなくNotebook外側の <code>m1</code>、<code>m2</code>、<code>n1</code>、<code>n2</code> に依存している。toy例では動くが、汎用関数としては引数から展開すべきである。
3. direct contractionでは同じ入力を二度reshapeしており、一方は冗長である。
4. relative output errorは理論で定義したが、現在の最終表示には含まれない。
5. 中間の主要2 cellにexecution countがないため、最終保存出力だけで現行ソースのFresh Run All成功を主張しない。

これらは保存された数値関係を否定するものではなく、教材コードを再利用可能な実装へ上げる際の修正点である。

## 11. 完了範囲

このNotebookが扱うのは、小さいrandom Linearに対する

- 2-core rank truncation
- weight error
- direct contraction
- $\Delta Y=X\Delta W^{\mathsf T}$
- Frobenius normとspectral norm
- 2本の出力誤差上界

である。

Fashion-MNIST実データ、MLP全体のReLU以降の誤差伝播、classification margin、accuracy、rank sweep、fine-tuningはまだ扱っていない。
