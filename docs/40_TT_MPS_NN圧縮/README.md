# TT/MPSによるニューラルネットワーク圧縮

この章は、TT/MPSの基礎理論と小規模な実装確認を、学習済みニューラルネットワークの圧縮へ接続する教材・検証記録を置く。

配置の境界は次のとおりである。

- 数学的な定義、導出、誤差評価は [[00_基礎理論/README]]
- 小さいTensorによる一般的なPyTorch確認、共通操作、実装contractは [[08_TT_MPS基礎実装検証/README]]
- 学習済みモデルへの置換、データセット上の評価、rank sweep、fine-tuningはこの章
- 再利用可能な実装の設計は [[90_src設計/README]]

## 現在の教材

- [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/README]]：2層MLPの重み誤差からbatch logits誤差、sample-wise certificate、minimum marginによるbatch内全予測不変の十分条件までの設定・結果・考察を確認する。
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/README]]

toy検証では人工的な重み誤差を使い、第2 Linearまでの恒等式、合成上界、scale sweepを確認した。さらに、1 sampleのmargin、batch Frobenius誤差からsample-wise certificateへのchain、minimum marginを使った1 batch内全sampleの同時保証まで確認した。学習済みFashion-MNIST MLPでは、第1層を打ち切りなしTTLinearへ置換し、対象Linear出力、ReLU出力、最終logitsの等価性を確認した段階である。いずれもTT-rank打ち切りによる実データ圧縮結果ではない。

test accuracyを含むrank sweep、TT coreのfine-tuning、parameter・MACs・latency・memoryの比較は、今後この章へ追加する。

## 関連

- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/59_TTLinearによるMLP一層置換とlogits等価性]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/62_TT-matrixのmode分解とrank設計]]
- [[08_TT_MPS基礎実装検証/11_TTLinearとdense_Linearのforward等価性のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/14_TTLinear置換で使うPyTorchモデル操作]]
- [[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]
- [[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]
