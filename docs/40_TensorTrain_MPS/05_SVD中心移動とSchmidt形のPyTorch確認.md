---
title: SVD中心移動とSchmidt形のPyTorch確認
tags:
  - TT
  - MPS
  - PyTorch
  - SVD
  - Schmidt
  - canonical
---

# SVD中心移動とSchmidt形のPyTorch確認

## このノートの位置づけ

[Notebook 08](../../notebooks/30_tt_mps/00_fundamentals/08_svd_center_move_and_schmidt_form.ipynb) の実装操作と保存済み数値結果をまとめる。ここで記す数値はNotebook内の出力であり、このノートを整理する際にRun Allで再取得した値ではない。

理論は [[00_基礎理論/01_数学基礎/02_テンソル代数/43_TT_MPSのSVD中心移動とSchmidt形]] を参照する。QRによる中心移動は [[40_TensorTrain_MPS/01_直交中心移動のPyTorch確認]]、mixed-canonical環境の等長性は [[40_TensorTrain_MPS/02_混合正準形の中心摂動と等長性のPyTorch確認]] を参照する。

## Setup

第2サイトを中心とするmixed-canonical form

$$
X
=
G_1^{[L]}
G_2^{[C]}
G_3^{[R]}
$$

を使った。

core shapeは、

$$
G_1^{[L]}:(1,4,2),\qquad
G_2^{[C]}:(2,3,3),\qquad
G_3^{[R]}:(3,5,1)
$$

だった。

元Tensorからmixed-canonical形への再構成誤差は、

$$
8.426803138067863\times10^{-15}
$$

だった。

## 第2中心コアのSVD

左展開

$$
A
=
G_2^{[C]\langle L\rangle}
\in
\mathbb R^{6\times3}
$$

へreduced SVDを適用した。

この `A = G2_center.reshape(r1 * n2, r2)` は、第2コアのshape $(r_1,n_2,r_2)$ のうち旧左ボンド $\alpha_1$ と物理添字 $i_2$ を行にまとめる操作である。Pythonの0始まりでは行番号 $p=\alpha_1n_2+i_2$、$A[p,\alpha_2]=G_2^{[C]}[\alpha_1,i_2,\alpha_2]$。同じコアを右展開するなら行は $\alpha_1$、列は $c=i_2r_2+\alpha_2$ で、中心を左へ動かす別の操作になる。両方の展開を同じ要素で比較した行列は [[40_TensorTrain_MPS/01_直交中心移動のPyTorch確認]] に置いた。

Notebook 08の独立した添字確認用小例では $(r_1,n_2,r_2)=(2,3,4)$。`A_left: (6,4)`、`C_right: (2,12)` となり、`G[1,2,3] = A_left[5,3] = C_right[1,11] = 23`（$5=1\cdot3+2$、$11=2\cdot4+3$）を保存出力で確認できる。この小例の $r_2=4$ は、以下の本実験の $r_2=3$ とは異なる。

得られたshapeは、

$$
U:(6,3),\qquad
S:(3,),\qquad
V^T:(3,3)
$$

だった。

$U$を第2coreへ戻し、$\Sigma V^T$を第3coreへ吸収したSVD center move後のshapeは、

$$
G_2^{[L]}:(2,3,3),
$$

$$
G_3^{[C]}:(3,5,1)
$$

となった。

再構成誤差は、

$$
1.5759259231498567\times10^{-14}
$$

だった。

すべての特異値を保持するexact SVDなので、全Tensorは丸め誤差水準で不変だった。

## SVD後の直交性と中心ノルム

第2coreの左直交性誤差は、

$$
9.038292206142435\times10^{-16}
$$

だった。

centerを第3サイトへ移した後、

$$
\left|
\|X\|_F
-
\|G_3^{[C]}\|_F
\right|
\approx
3.552713678800501\times10^{-15}
$$

となり、center normの局所化を確認した。

## Sigmaをbond上へ明示する

Notebookの `R_old = G3_right.squeeze(-1)` は旧右コアの行列表現 $(r_2,n_3)=(3,5)$、`Vh` は実数SVDの $V^T$ で $(\rho,r_2)=(3,3)$ である。`R_tilde = Vh @ R_old` は旧ボンド添字を足し、新しいSchmidtラベル $\beta$ を行にする。

$$
R_{\mathrm{tilde}}[\beta,i_3]
=\sum_{\alpha_2=0}^{r_2-1}V^T[\beta,\alpha_2]R_{\mathrm{old}}[\alpha_2,i_3],
\qquad R_{\mathrm{tilde}}\in\mathbb R^{\rho\times n_3}.
$$

これを第3中心コアへ戻すには対角係数も掛け、$G_3^{[C]}[\beta,i_3,0]=(\Sigma V^TR_{\mathrm{old}})[\beta,i_3]=\sigma_\beta R_{\mathrm{tilde}}[\beta,i_3]$ とする。右基底回転と係数の吸収は別の操作である。右直交性の途中式は [[00_基礎理論/01_数学基礎/02_テンソル代数/43_TT_MPSのSVD中心移動とSchmidt形]] の第3節を参照する。

右coreを$V^T$でbond基底回転した行列は、

$$
\widetilde R
\in
\mathbb R^{3\times5}
$$

となった。

右直交性誤差は、

$$
\|\widetilde R\widetilde R^T-I\|_F
\approx
8.275842379697559\times10^{-16}
$$

だった。

$$
X
=
G_1^{[L]}
G_2^{[L]}
\Sigma
\widetilde G_3^{[R]}
$$

というSigmaをbond上に残した形での再構成誤差は、

$$
1.3927300431570368\times10^{-14}
$$

だった。

## Schmidt states

`torch.tensordot(G1_left, G2_left_svd, dims=([2], [0]))` は共通の $r_1$ 軸だけを縮約する。入力shape $(1,n_1,r_1)$ と $(r_1,n_2,\rho)$ から未縮約の $(1,n_1,n_2,\rho)$ が残るため、保存出力の `L_tensor` は **4階**の $(1,4,3,3)$ になる。各要素は

$$
\mathrm{L\_tensor}[0,i_1,i_2,\beta]
=\sum_{\alpha_1=0}^{r_1-1}
G_1^{[L]}[0,i_1,\alpha_1]G_2^{[L]}[\alpha_1,i_2,\beta].
$$

`squeeze(0)` 後の $(n_1,n_2,\rho)$ は二つの物理添字を別々に持つテンソル表示である。左Schmidt状態を**列**としてGram行列を計算するため、$a=i_1n_2+i_2$ を行にして `L_block = L_tensor.reshape(n1 * n2, rho)` とする。Notebookでは一度 `L_tensor.squeeze(0)` を `L_block` に代入した後、reshapeした行列で上書きしている。係数は同じで、最終的な行列表示が $(12,3)$ である。

$$
\begin{aligned}
(L_{\mathrm{block}}^TL_{\mathrm{block}})_{\beta\gamma}
&=\sum_{a=0}^{n_1n_2-1}L_{\mathrm{block}}[a,\beta]L_{\mathrm{block}}[a,\gamma]\\
&=\sum_{i_1,i_2}L_\beta(i_1,i_2)L_\gamma(i_1,i_2)
=\delta_{\beta\gamma}.
\end{aligned}
$$

最後の等号は第1・第2コアの左直交性を順に使う。行要素を省かずに証明した式は43番ノートの第4節にある。

左Schmidt blockは、

$$
L\in\mathbb R^{12\times3},
$$

右Schmidt blockは、

$$
R\in\mathbb R^{3\times5}
$$

となった。

直交性誤差は、

$$
\|L^TL-I\|_F
\approx
8.242513085424968\times10^{-16},
$$

$$
\|RR^T-I\|_F
\approx
8.275842379697559\times10^{-16}
$$

だった。

したがって、

$$
|X\rangle
=
\sum_{\beta=1}^{\rho}
\sigma_\beta
|L_\beta\rangle
\otimes
|R_\beta\rangle
$$

というSchmidt形の左右block statesが数値的に正規直交していることを確認した。

## Norm identity

数値結果は、

$$
\|X\|_F^2
=
714.0546134934193,
$$

$$
\|G_2^{[C]}\|_F^2
=
714.0546134934189,
$$

$$
\sum_\beta\sigma_\beta^2
=
714.0546134934187
$$

だった。

差は、

$$
\left|
\|X\|_F^2
-
\|G_2^{[C]}\|_F^2
\right|
\approx
4.547473508864641\times10^{-13},
$$

$$
\left|
\|X\|_F^2
-
\sum_\beta\sigma_\beta^2
\right|
\approx
5.684341886080801\times10^{-13}
$$

で、float64の丸め誤差水準だった。

## 保存出力と再実行の区別

上の数値はNotebook 08に保存された出力である。現行のNotebookでは、添字確認用の小例で `r1, n2, r2 = 2, 3, 4` を設定した後、本実験の中心コア `G2_center: (2,3,3)` に `reshape(r1 * n2, r2)` を適用している。この順に実行すると、要素数18のコアを要素数24のshape $(6,4)$ に変えることになり失敗する。小例と本実験の変数名を分けるか、本実験のshapeをコアから再取得する必要がある。このノートは保存済み出力を記録したものであり、現行NotebookのRun All成功を示すものではない。

## 確認できたこと

1. 全特異値を保持するSVD center moveはexactである。
2. $U$を左coreへ戻すことで左直交性が得られる。
3. $V^T$によるbond基底回転後も右直交性が保たれる。
4. $\Sigma$をbond上へ明示するとSchmidt係数を直接読める。
5. 左右Schmidt statesは正規直交する。
6. $\|X\|_F^2=\sum_\beta\sigma_\beta^2$ を数値確認した。
