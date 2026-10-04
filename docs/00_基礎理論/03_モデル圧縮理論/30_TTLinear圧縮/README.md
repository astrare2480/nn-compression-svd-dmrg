# TTLinearによるニューラルネットワーク圧縮

TT-matrix・TTLinearの数学的基礎を、学習済みMLPの一層置換、rank打ち切り誤差、ReLUを通る誤差、mode・rank設計へ接続する。

## 教材

1. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/59_TTLinearによるMLP一層置換とlogits等価性]]
2. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/60_TT-rank打ち切りによるLinear出力誤差とnorm上界]]
3. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/61_Lipschitz連続性とReLUの1-Lipschitz性]]
4. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/63_ReLUによるhidden_activation誤差伝播]]
5. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]
6. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]
7. [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/62_TT-matrixのmode分解とrank設計]]

spectral normと最大特異値の一般的な導出は [[00_基礎理論/01_数学基礎/01_線形代数/07_spectral_normと最大特異値_supとmax]]、前提となるTT-matrixの定義とforwardは [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/README]]、PyTorchの一般的な小規模確認は [[08_TT_MPS基礎実装検証/README]]、学習済みFashion-MNIST MLPへの適用記録は [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/README]] を参照する。
