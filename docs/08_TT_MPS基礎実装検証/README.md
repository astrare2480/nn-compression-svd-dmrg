# Tensor Train / MPS

このディレクトリは、SVD → Tucker / HOSVD / HOOIの次に進む **Tensor Train（TT）/ Matrix Product State（MPS）の基礎実装・検証**用である。

理論そのものは [[00_基礎理論/README]] に置き、この章では小さいTensorによるPyTorch確認、配列操作、実装contract、保存済み実行結果を扱う。NN重みのTT圧縮・学習実験は、ここで完了したとは扱わない。

基礎Notebook 00〜15の到達点、代表的な保存済み数値結果、src化状況、未確認事項をまとめて読む場合は、最初に[[08_TT_MPS基礎実装検証/00_TT_MPS基礎実装の確認結果]]を参照する。学習済みニューラルネットワークへの適用は [[40_TT_MPS_NN圧縮/README]] に分ける。

## 前提として完了した内容

```text
SVD
→ 行列の低rank近似
→ 特異値truncation
→ Fine-tuning

Tucker / HOSVD
→ Tensorのmode
→ modeごとのrank
→ core / factor

HOOI
→ 固定rankでfactorを反復更新
→ sweep
→ convergence
```

特にHOOIで学んだ「初期分解 → 固定rank → 局所的な更新 → sweep → 収束判定」は、TT/MPSからDMRGへ進むときの重要な橋渡しになる。

## 学習NotebookとPyTorch確認ノートの対応

`notebooks/30_tt_mps/00_fundamentals/` では、理論を学習しながら小さいTensorで数値確認している。

`docs/08_TT_MPS基礎実装検証/` では、Notebookの全文やTODO解答を複製せず、**確認した式・shape・代表的な数値結果・結論**をテーマ単位でまとめる。

| 学習Notebook | 主題 | PyTorch確認ノート |
| --- | --- | --- |
| 00〜02 | TT-SVD、TT-rank、rank truncationと誤差 | [[08_TT_MPS基礎実装検証/03_TT_SVDとrank_truncationのPyTorch確認]] |
| 03〜05 | 基底変換、Gauge自由度、左/右QR直交化 | [[08_TT_MPS基礎実装検証/04_基底変換_Gauge自由度_左右直交化のPyTorch確認]] |
| 06 | mixed-canonical form、中心ノルム、中心摂動の等長性 | [[08_TT_MPS基礎実装検証/02_混合正準形の中心摂動と等長性のPyTorch確認]] |
| 07 | QRによるorthogonality centerの移動 | [[08_TT_MPS基礎実装検証/01_直交中心移動のPyTorch確認]] |
| 08 | SVD center move、Schmidt形 | [[08_TT_MPS基礎実装検証/05_SVD中心移動とSchmidt形のPyTorch確認]] |
| 09〜10 | truncated SVD、単一bond rank truncation、EYM最適性 | [[08_TT_MPS基礎実装検証/06_単一ボンドSVD打ち切りのPyTorch確認]]、[[08_TT_MPS基礎実装検証/00_TT_MPS基礎実装の確認結果]] |
| 11 | TT-SVDの基本誤差上界、準最適性 | [[08_TT_MPS基礎実装検証/07_TT-SVDの準最適性と誤差上界のPyTorch確認]] |
| 12 | TT-rounding、right-canonical化、局所予算によるrank選択 | [[08_TT_MPS基礎実装検証/08_TT-roundingのPyTorch実装設計と検証]] |
| 13 | TT内積、left environment、Frobeniusノルム、TT間距離 | [[08_TT_MPS基礎実装検証/09_TT内積_Frobeniusノルム_距離のPyTorch確認]] |
| 14 | TT-matrix、dense reconstruction、TT-Linear forward | [[08_TT_MPS基礎実装検証/10_TT-matrix_Dense_Reconstruction_TT-Linear_ForwardのPyTorch確認]]、[[08_TT_MPS基礎実装検証/45_TT-matrix線形層のPyTorch直接forward]] |
| 15 | truncationなし2-core TT-SVD、dense `nn.Linear`とのforward等価性 | [[08_TT_MPS基礎実装検証/11_TTLinearとdense_Linearのforward等価性のPyTorch確認]] |
| `10_fashion_mnist_mlp/01` | NN適用前に一般化できる2-core rank truncation、Linear出力誤差恒等式、Frobenius / spectral norm上界のtoy確認 | [[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]] |
| `10_fashion_mnist_mlp/02` | ReLUの4符号ケース、要素不等式、hidden activationのFrobenius誤差、LinearからReLUまでのtoy上界 | [[08_TT_MPS基礎実装検証/15_ReLU_hidden_activation誤差伝播のPyTorch確認]] |
| `10_fashion_mnist_mlp/03` | 第2 Linearのlogits誤差恒等式、spectral norm、合成上界、固定方向の人工摂動scale sweep | 実装は[[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]、設定・結果・考察は[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/00_第2Linearのlogits誤差伝播_設定結果考察]] |
| `10_fashion_mnist_mlp/04` | 1 sampleのclassification margin、$\ell_\infty$ logits error、worst-case方向、argmax安定性 | 実装は[[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]、設定・結果・考察は[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/01_ClassificationMarginとArgmax安定性_設定結果考察]] |
| `10_fashion_mnist_mlp/05` | batch logits errorから1 sampleのnorm chain、重み誤差上界、classification margin、prediction stability certificateへの接続 | 実装は[[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]、設定・結果・考察は[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/02_BatchLogitsErrorからPredictionStabilityCertificate_設定結果考察]] |
| `10_fashion_mnist_mlp/06` | margin vector、minimum margin、bottleneck sample、batch内全sampleのprediction stability certificate | 実装は[[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]、設定・結果・考察は[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/03_MinimumMarginとBatch-widePredictionStability_設定結果考察]] |
| `10_fashion_mnist_mlp/07` | 固定tensorization・3-core TT-matrixでのrank sweep、parameter count、compression ratio、weight reconstruction error | [[08_TT_MPS基礎実装検証/20_TT-matrixのrank_sweepと圧縮誤差のPyTorch確認]] |

この対応は「Notebook 1本につきdocs 1本」ではなく、内容が連続するNotebookは1つの検証ノートへまとめる方針とする。

## 現在のPyTorch確認ノート

- [[08_TT_MPS基礎実装検証/00_TT_MPS基礎実装の確認結果]]
- [[08_TT_MPS基礎実装検証/01_直交中心移動のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/02_混合正準形の中心摂動と等長性のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/03_TT_SVDとrank_truncationのPyTorch確認]]
- [[08_TT_MPS基礎実装検証/04_基底変換_Gauge自由度_左右直交化のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/05_SVD中心移動とSchmidt形のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/06_単一ボンドSVD打ち切りのPyTorch確認]]
- [[08_TT_MPS基礎実装検証/07_TT-SVDの準最適性と誤差上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/08_TT-roundingのPyTorch実装設計と検証]]
- [[08_TT_MPS基礎実装検証/09_TT内積_Frobeniusノルム_距離のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/10_TT-matrix_Dense_Reconstruction_TT-Linear_ForwardのPyTorch確認]]
- [[08_TT_MPS基礎実装検証/11_TTLinearとdense_Linearのforward等価性のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/14_TTLinear置換で使うPyTorchモデル操作]]
- [[08_TT_MPS基礎実装検証/15_ReLU_hidden_activation誤差伝播のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]
- [[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]
- [[08_TT_MPS基礎実装検証/20_TT-matrixのrank_sweepと圧縮誤差のPyTorch確認]]

## 共通実装・操作ノート

- [[08_TT_MPS基礎実装検証/27_TT_MPS基礎のPyTorch実装]]：TT cut、TT-SVD、再構成、rank、パラメータ数と公開APIのcontract。
- [[08_TT_MPS基礎実装検証/28_TT_MPS実装で使うPyTorch_Python操作メモ]]：shape、reshape、縮約、Pythonコンテナなどの操作。
- [[08_TT_MPS基礎実装検証/29_TT_cutとPyTorchのreshape_Kronecker順序]]：複合添字、reshape、置換行列、Kronecker積の順序。
- [[08_TT_MPS基礎実装検証/30_PyTorchのvector_norm_matrix_normとtorch_linalg]]：要素絶対値、vector norm、Frobenius norm、spectral norm、特異値関連APIの使い分け。
- [[08_TT_MPS基礎実装検証/45_TT-matrix線形層のPyTorch直接forward]]：TT-matrix / MPOコアから密行列を作らないforward。
- [[08_TT_MPS基礎実装検証/14_TTLinear置換で使うPyTorchモデル操作]]：checkpoint、`state_dict`、`eval()`、`no_grad()`、module pathによる一層置換。

TT-matrixの理論は、定義とKronecker積を [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/54_TT-matrixの定義とKronecker積表現]]、dense重みのtensorizationを [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/55_dense重みのTT-matrix tensorizationとTT-SVD初期化]]、dense reconstructionを [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/56_TT-matrixのdense reconstruction]]、右から左へのforward縮約を [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/57_TT-Linear_forwardの縮約とshape]]、dense `nn.Linear`との等価性を [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/58_TTLinearとdense_Linearのforward等価性]]、bond dimensionとunfoldingの表現能力を[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/69_TT-matrixのbond_dimensionと表現能力]]、固定tensorizationでのrank sweepを[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/70_TT-matrixのrank_sweepと圧縮率_再構成誤差]]に分けている。

学習済みMLPへ進む部分は、一層置換からlogitsまでの等価性を [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/59_TTLinearによるMLP一層置換とlogits等価性]]、rank打ち切りによるLinear出力誤差を [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/60_TT-rank打ち切りによるLinear出力誤差とnorm上界]]、ReLUの数学的性質を [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/61_Lipschitz連続性とReLUの1-Lipschitz性]]、hidden activationまでの誤差伝播を [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/63_ReLUによるhidden_activation誤差伝播]]、第2 Linearを通るlogits誤差と合成上界を [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]、1 sampleのprediction安定性を [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]、batch errorからsample-wise certificateへの接続を[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/66_BatchLogitsErrorからSample-wisePredictionStabilityへ]]、minimum marginによるbatch内全sampleの同時保証を[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/67_Batch内全予測不変の十分条件]]、modeとrankの設計を [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/62_TT-matrixのmode分解とrank設計]] に分けている。学習済みモデルでの検証記録は [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/README]] を参照する。

## 学習予定

1. Tensorization / reshape
2. TT / MPSのcore shape
3. TT rank / bond dimension
4. TT-SVD
5. reconstructionとrelative error
6. parameter数・圧縮率
7. canonical form / orthogonality center
8. SVD center move / Schmidt form
9. 単一bond truncationと誤差
10. Eckart–Young–Mirskyによる単一bond最適性
11. rank決定とTT rounding
12. TT内積・Frobeniusノルム・TT間距離
13. TT-matrixとNN weightへの対応
14. 必要ならCIFAR-10等でSVD / Tuckerとの比較
15. 局所2-site更新とDMRGへの接続

## 評価軸

SVD / Tuckerで使った評価軸を可能な限り維持する。

```text
weight / tensor relative error
parameters
MACs
圧縮直後accuracy
decomposition time
Fine-tuning後accuracy
```

TT rank / bond dimensionを変えたときも、圧縮率だけでなくtask性能と分けて評価する。

基礎Notebookでは、上記に加えて次を数値確認する。

```text
reconstruction error
orthogonality / Gram error
center norm
Schmidt-state orthogonality
discarded singular-value energy
single-bond truncation error
TT inner product
TT Frobenius norm
TT-to-TT distance
```

TT-to-TT distanceはNotebookの小規模検証項目である。現行srcでPublic APIとして公開するのは`tt_inner`と`tt_fro_norm`までであり、`tt_distance_sq` / `tt_distance`は数値安定性とautogradのcontractが未確立なため延期している。

## ノート設計

これまでと同様に、

```text
Notebook
→ 自作して理解
→ shapeと途中式を確認
→ 小さいTensorでsanity check

docs/08_TT_MPS基礎実装検証
→ Notebookで確認した重要結果をテーマ単位で整理

src
→ 学習後に再利用処理だけ共通化
```

とする。

理論ノートでは、reshape / unfolding / SVD / truncate / core / reconstruction / errorを途中式から確認する。

直交中心の移動は [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/41_TT_MPSの直交中心の移動]]、ブロック状態としての意味は [[00_基礎理論/07_物理基礎/42_MPS正準形と直交中心の物理的意味]]、PyTorchでのshape・QR・収縮の確認は [[08_TT_MPS基礎実装検証/01_直交中心移動のPyTorch確認]] を参照する。中心だけの摂動と等長性を検証する場合は [[08_TT_MPS基礎実装検証/02_混合正準形の中心摂動と等長性のPyTorch確認]] へ進む。TT-roundingでは [[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/48_TT-roundingの定義と正準化sweep]]、[[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/49_TT-roundingの環境行列と誤差直交分解]]、[[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/50_TT-roundingの誤差予算とrank選択]] を読んでから、[[08_TT_MPS基礎実装検証/08_TT-roundingのPyTorch実装設計と検証]] で実装上のshapeと検証項目へ接続する。既存の学習NotebookのTODOはこの実装ノートでは変更しない。

TT-rounding後の内積・Frobeniusノルムをdense復元なしで評価する場合は、[[00_基礎理論/01_数学基礎/02_テンソル代数/50_縮約_ノルム_代数演算/51_TT_MPSの内積とenvironment縮約]]、[[00_基礎理論/01_数学基礎/02_テンソル代数/50_縮約_ノルム_代数演算/52_TT_MPSのFrobeniusノルムと距離]]、[[00_基礎理論/01_数学基礎/02_テンソル代数/50_縮約_ノルム_代数演算/53_TT_MPSの加減算とrank増加]] を読み、[[08_TT_MPS基礎実装検証/09_TT内積_Frobeniusノルム_距離のPyTorch確認]] で`einsum`の添字、environment shape、dense照合へ接続する。距離は同ノートの学習検証に留め、現行Public APIとしては利用しない。

## 先に読む

- [[00_基礎理論/06_手法間のつながり/14_低ランク学習からテンソルネットワークへの発展]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/10_テンソル基礎_Tucker_HOSVD_HOOI/20_Tucker_HOSVD_HOOI数式の導出]]
- [[06_Tucker基礎実装検証/README]]
- [[README_実装編]]

## 次

TT / MPSでbond dimensionとchain型Tensor表現を理解した後、[[50_DMRG/README]] へ進む。
