# Fashion-MNIST MLPのTTLinear圧縮

この章では、Fashion-MNISTの学習済みMLPへTTLinearを適用し、exact置換の等価性確認からrank打ち切り、精度評価、fine-tuningへ進む。

## 現在の到達点

[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/00_TTLinear一層置換とlogits等価性]]では、`fc1: 784 -> 512`だけを打ち切りなしTTLinearへ置換した。保存済み出力では、対象Linear直後、ReLU直後、最終logitsがfloat64の丸め誤差水準で一致し、predictionも一致した。

一方、exact TT-rankは$(1,16,256,56,1)$、TT重みは470,336要素であり、dense重み401,408要素より大きい。したがって、現在の結果は学習済みモデル上の関数等価性検証であり、圧縮結果ではない。

一般的なtoy modelでは、第1 Linearの重み誤差をReLUと第2 Linearへ伝播させる合成上界、1 sampleのclassification margin、batch logits errorからsample-wise prediction certificate、minimum marginによるbatch内全sampleの同時保証までを確認している。PyTorch操作は[[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]、[[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]、[[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]、[[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]、設定・結果・考察は[[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/README]]に分け、Fashion-MNISTの実測結果とは数えない。

## 読む順序

1. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/59_TTLinearによるMLP一層置換とlogits等価性]]
2. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/62_TT-matrixのmode分解とrank設計]]
3. [[08_TT_MPS基礎実装検証/11_TTLinearとdense_Linearのforward等価性のPyTorch確認]]
4. [[08_TT_MPS基礎実装検証/14_TTLinear置換で使うPyTorchモデル操作]]
5. [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/00_TTLinear一層置換とlogits等価性]]
6. [[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]]
7. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/61_Lipschitz連続性とReLUの1-Lipschitz性]]
8. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/63_ReLUによるhidden_activation誤差伝播]]
9. [[08_TT_MPS基礎実装検証/15_ReLU_hidden_activation誤差伝播のPyTorch確認]]
10. [[00_基礎理論/01_数学基礎/01_線形代数/07_spectral_normと最大特異値_supとmax]]
11. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]
12. [[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]
13. [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/00_第2Linearのlogits誤差伝播_設定結果考察]]
14. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]
15. [[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]
16. [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/01_ClassificationMarginとArgmax安定性_設定結果考察]]
17. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/66_BatchLogitsErrorからSample-wisePredictionStabilityへ]]
18. [[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]
19. [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/02_BatchLogitsErrorからPredictionStabilityCertificate_設定結果考察]]
20. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/67_Batch内全予測不変の十分条件]]
21. [[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]
22. [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/03_MinimumMarginとBatch-widePredictionStability_設定結果考察]]

対応Notebookは [10_fashion_mnist_mlp](../../../notebooks/30_tt_mps/10_fashion_mnist_mlp/) に置く。

## 未完了の次段階

- 学習済みFashion-MNIST MLPの打ち切りrankで、ReLUを通るhidden errorを数値確認すること。一般的なtoy例は実装検証ノート15で確認済み
- 学習済みFashion-MNIST MLPの実際の打ち切りrankで、後続Linearを通るlogits errorと合成上界を数値確認すること。random toy modelは実装検証ノート16で確認済み
- 学習済みFashion-MNIST MLPを実際にTT-rank打ち切りしたときのsample別classification margin、minimum margin、batch内prediction agreement。一般的なtoy例は理論ノート65〜67、実装ノート17〜19、toy結果ノート01〜03で確認済み
- Fashion-MNIST test setでのrank sweepとaccuracy
- TT coreのfine-tuning
- parameter、MACs、latency、memoryの実測比較
