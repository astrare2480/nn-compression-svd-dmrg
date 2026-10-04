# 2層MLPのTT圧縮誤差toy検証

この章には、学習済みデータセットモデルへ進む前に、2層MLPの小さいrandom行列で行う圧縮誤差伝播実験を置く。

## 教材

1. [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/00_第2Linearのlogits誤差伝播_設定結果考察]]
2. [[40_TT_MPS_NN圧縮/00_2層MLP_toy検証/01_ClassificationMarginとArgmax安定性_設定結果考察]]

## 配置の境界

- 一般式と途中式：[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]
- classification marginとargmax安定性の証明：[[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]
- PyTorch API、shape、assert、実装上の注意：[[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]
- classification marginのPyTorch確認：[[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]
- toy実験の設定、保存済み結果、その考察：この章
- 学習済みFashion-MNIST MLPへの適用：[[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/README]]

このtoy検証は人工的な重み摂動または手入力したlogits摂動を使う。TT-rankを指定して実際に圧縮した結果や、Fashion-MNISTのaccuracy結果としては扱わない。
