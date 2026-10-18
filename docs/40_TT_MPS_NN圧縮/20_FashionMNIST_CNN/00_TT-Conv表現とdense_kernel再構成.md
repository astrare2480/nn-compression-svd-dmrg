---
title: TT-Conv表現とdense kernel再構成の確認
tags:
  - TTConv
  - CNN
  - Conv2d
  - TT-matrix
  - autograd
---

# TT-Conv表現とdense kernel再構成の確認

## このノートの位置づけ

このノートは、Notebook 00の保存結果と、同じ内容についてChatGPTおよびPerplexityとの対話で確認した事項を統合する。

- [00_ttconv_representation_and_dense_kernel_reconstruction.ipynb](../../../notebooks/30_tt_mps/20_fashion_mnist_cnn/00_ttconv_representation_and_dense_kernel_reconstruction.ipynb)

Notebook 00の目的は、学習済みCNNを圧縮することではない。小さい`Conv2d`のkernelを行列化してTT-matrixへ変換し、次を順に確認することである。

1. Conv kernelとTT-matrixのshape対応
2. exact TTとrank制限TTの違い
3. TT coresからdense kernelを再構成できること
4. 再構成kernelを使ったConv出力
5. 再構成経路を通したTT coreのgradient
6. dense、exact TT、rank制限TTのparameter数

Conv、patch、TT-matrix、直接forwardを含む一般的な説明は、[[00_基礎理論/03_モデル圧縮理論/31_TTConv圧縮/79_Conv2dのTT-matrix表現と直接forward]]へ分ける。このノートではNotebook 00の設定と保存結果を扱う。

## 1. 証拠レベル

Notebookの保存状態では、16個のコードセルすべてに実行番号があり、保存されたerror出力はない。shape、誤差、gradient、parameter数のassertionも通過している。

ただし、このノート作成時にFresh Restart KernelからRun Allを新たに実行したわけではない。ここで確認したのは、保存済みNotebookのコード・出力と数値の整合性である。

## 2. 小さいConv2dの設定

| 項目 | 値 |
| :-- | :-- |
| batch size | 2 |
| 入力channel | 4 |
| 出力channel | 8 |
| kernel size | $3\times3$ |
| stride | $(1,1)$ |
| padding | $(1,1)$ |
| dilation | $(1,1)$ |
| groups | 1 |
| bias | あり |
| 入力shape | $(2,4,7,7)$ |
| 出力shape | $(2,8,7,7)$ |
| dtype | `torch.float64` |
| device | CPU |
| seed | 42 |

Conv weightのshapeは、

$$
(C_{\mathrm{out}}, C_{\mathrm{in}}, K_H, K_W)
=
(8,4,3,3)
$$

で、weight部分は288要素、biasを含めると296 parameterである。

## 3. kernelの行列化とmode分解

PyTorchのweight配置を保ったまま、

$$
W\in\mathbb{R}^{8\times4\times3\times3}
\longrightarrow
W_{\mathrm{mat}}\in\mathbb{R}^{8\times36}
$$

とreshapeした。

採用したmodeは、

$$
\mathrm{OUT\_MODES}=(2,2,2),
\qquad
\mathrm{IN\_MODES}=(3,3,4)
$$

である。

$$
2\times2\times2=8=C_{\mathrm{out}},
$$

$$
3\times3\times4=36=C_{\mathrm{in}}K_HK_W
$$

を満たす。対応するcombined physical modesは`(6, 6, 8)`で、TT-SVDへ渡すtensorのshapeも`(6, 6, 8)`になった。

このmode分解は、要素を欠落・重複させず並べ替えるための有効な1候補である。最適なtensorizationであることは比較していない。

## 4. 保存結果

### 4.1 actual TT-rankとcore shape

| 表現 | actual ranks | core shapes |
| :-- | :-- | :-- |
| exact TT | $(1,6,8,1)$ | $(1,2,3,6)$、$(6,2,3,8)$、$(8,2,4,1)$ |
| rank制限TT | $(1,2,2,1)$ | $(1,2,3,2)$、$(2,2,3,2)$、$(2,2,4,1)$ |

exact TTでは、各unfoldingに必要なrankを保持した。rank制限TTでは、内部rankを最大2へ制限した。

### 4.2 kernel再構成誤差

| 表現 | 元kernelに対する相対Frobenius誤差 |
| :-- | --: |
| exact TT | $2.47\times10^{-15}$ |
| rank制限TT | $0.883719$ |

exact TTの再構成kernelは、float64の丸め誤差水準で元kernelと一致した。rank制限TTでは、内部rankを2へ制限したため近似誤差が発生した。

### 4.3 Conv出力誤差

同じ入力、bias、stride、padding、dilation、groupsを使い、再構成kernelで`F.conv2d`を計算した。

| 比較 | 出力RMSE |
| :-- | --: |
| exact再構成kernel vs 元Conv | $1.23\times10^{-15}$ |
| rank制限再構成kernel vs 元Conv | $0.440254$ |

exact TTではkernelだけでなくConv出力も元の`Conv2d`と一致した。rank制限TTの出力差は、このNotebookでは圧縮近似による観測値として記録する。分類accuracyへの影響は評価していない。

### 4.4 再構成経路のgradient

rank制限coresを`detach().clone()`して独立な`nn.Parameter`とし、`nn.ParameterList`へ登録した。そこから、

$$
\{G^{(k)}\}
\rightarrow W_{\mathrm{mat}}
\rightarrow W_{\mathrm{conv}}
\rightarrow Y
\rightarrow \mathcal{L}
$$

という再構成経路を作り、MSE lossをbackwardした。

| core | shape | gradient norm |
| ---: | :-- | --: |
| 0 | $(1,2,3,2)$ | 0.117466 |
| 1 | $(2,2,3,2)$ | 0.118728 |
| 2 | $(2,2,4,1)$ | 0.220505 |

全coreについて、gradientが存在し、shapeがcoreと一致し、全要素がfiniteであることを確認した。これは、dense kernel再構成を含む計算グラフがTT coresまで切れずにつながっていることのsanity checkである。gradientが存在することだけでは、その値が別実装と一致することまでは確認できない。値の照合はNotebook 01で行う。

### 4.5 parameter数

| 表現 | actual ranks | bias込みparameter数 | dense比圧縮率 |
| :-- | :-- | --: | --: |
| dense | - | 296 | 1.000倍 |
| exact TT | $(1,6,8,1)$ | 396 | 0.747倍 |
| rank制限TT | $(1,2,2,1)$ | 60 | 4.933倍 |

rank制限TTのcore部分は、

$$
1\times2\times3\times2
+
2\times2\times3\times2
+
2\times2\times4\times1
=
52
$$

要素である。bias 8要素を加えて60 parameterになる。

## 5. 保存結果の読み方

### 5.1 exact TTは表現の整合性確認であり、圧縮ではない

exact TTはkernelとConv出力を再現したが、parameter数はdenseの296より多い396だった。従って、「TTへ変換できた」ことと「圧縮できた」ことは別である。

Notebook 00でexact TTを作った目的は、kernelの行列化、mode分解、TT core、dense再構成という一連の対応が正しいことを確認するためである。

### 5.2 rank制限で初めてparameter削減と近似誤差が生じた

rankを$(1,2,2,1)$へ制限するとparameter数は60まで減った一方、kernel相対誤差`0.883719`、Conv出力RMSE`0.440254`が生じた。

この値だけから圧縮として良いか悪いかは決めない。random kernelとrandom入力に対する小規模例であり、出力RMSEの相対的な大きさや分類性能を評価する実験ではない。ここでは、rank制限によって「parameter削減と近似誤差が同時に生じる」ことを数値で確認した。

### 5.3 mode分解とTT-rank制限は別の選択である

`OUT_MODES=(2,2,2)`と`IN_MODES=(3,3,4)`は、8行・36列をどのsiteへ分けるかを定める。`max_rank=2`は、分解後のbond dimensionをどこまで保持するかを定める。

今回のmodeで全要素を表現できることと、rank 2で元kernelを完全表現できることは別問題である。exact TTでは同じmodeでも再現でき、rank制限時に誤差が生じたことが、その違いを示している。

### 5.4 Notebook 00ではTTを「重み表現」として使った

Notebook 00のforwardは、

```text
TT cores → dense weight matrix → dense Conv kernel → F.conv2d
```

という経路である。TT coresを保存・学習可能なparameterとして扱えることは確認したが、Conv計算時にはdense kernelを明示的に作っている。

従って、Notebook 00だけでは「TT coresから直接Convを計算できる」とは言えない。これを確認するのがNotebook 01の役割である。

## 6. このNotebookで確認した範囲

- Conv kernelを$8\times36$の行列へ対応づけられた。
- 固定したmode分解からexact TTとrank制限TTを作れた。
- exact TTから元kernelとConv出力を再現できた。
- rank制限によるparameter削減と近似誤差を観測した。
- 再構成経路から全TT coreへgradientが届いた。
- exact TTは必ずしも圧縮にならないことを数値で確認した。

一方、dense kernelを作らない直接TT forward、直接経路のgradient照合、速度、peak memory、MACs、学習済みCNN、分類accuracyは、このNotebookでは扱っていない。

## 7. 結論

> Notebook 00では、Conv weightを$8\times36$へ行列化し、`OUT_MODES=(2,2,2)`、`IN_MODES=(3,3,4)`のTT-matrixとして表現した。exact TTは元kernelとConv出力を約$10^{-15}$の誤差で再現したが、396 parameterとなりdenseの296より大きかった。rankを$(1,2,2,1)$へ制限すると60 parameterまで減る一方、kernelとConv出力に近似誤差が生じた。また、TT coresからdense kernelを再構成する計算グラフを通して、全coreへgradientが届くことを確認した。これはTTをConv weightの表現として扱う基礎確認であり、dense kernelを介さない直接forwardはまだ実装していない。
