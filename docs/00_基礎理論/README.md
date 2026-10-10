# 基礎理論

SVDによる低ランク近似からTucker/HOSVD/HOOI、さらにTT/MPS・DMRGへ進むための共通知識を整理する。

## 数式を追うとき

最終式だけでなく途中式まで確認する場合は、最初に次を読む。

- [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出]]
- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価_数式導出補足]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/10_テンソル基礎_Tucker_HOSVD_HOOI/20_Tucker_HOSVD_HOOI数式の導出]]

## 01_数学基礎

- [[00_基礎理論/01_数学基礎/README]]

### 01_線形代数

- [[00_基礎理論/01_数学基礎/01_線形代数/01_SVDとは]]
- [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出]]
- [[00_基礎理論/01_数学基礎/01_線形代数/03_SVDによる低ランク近似]]
- [[00_基礎理論/01_数学基礎/01_線形代数/05_圧縮率とRank]]
- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]
- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価_数式導出補足]]
- [[00_基礎理論/01_数学基礎/01_線形代数/07_spectral_normと最大特異値_supとmax]]：長方形SVDからspectral normを導き、$\sup$ と $\max$ を区別する。

### 02_テンソル代数

- [[00_基礎理論/01_数学基礎/02_テンソル代数/README]]：Tensor、Tucker、TT/MPS、TT-matrixのテーマ別索引。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/10_テンソル基礎_Tucker_HOSVD_HOOI/20_Tucker_HOSVD_HOOI数式の導出]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/10_テンソル基礎_Tucker_HOSVD_HOOI/21_テンソルとmode演算]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/10_テンソル基礎_Tucker_HOSVD_HOOI/22_Tucker分解とHOSVD]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/10_テンソル基礎_Tucker_HOSVD_HOOI/23_HOOI]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/20_TT_MPS基礎/30_TT_MPSの定義]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/20_TT_MPS基礎/31_TT-rankとunfolding]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/20_TT_MPS基礎/32_TT-SVD]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/20_TT_MPS基礎/33_TT-SVDの打ち切りと誤差]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/20_TT_MPS基礎/34_基底変換とTT-rank不変性]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/35_TT_MPSの等長写像と射影]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/36_TT_MPSのGauge自由度と左QR直交化]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/37_TT_MPSの左ブロックと直交性の導出]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/38_TT_MPSの右QR直交化と右ブロック]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/39_TT_MPSの混合正準形への導入]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/40_TT_MPSの中心ノルムと内積の導出]]：中心のノルム・差分・内積を導出する。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/41_TT_MPSの直交中心の移動]]：QRによる左右移動と全テンソルの不変性を、テンソル代数として導出する。ブロック状態としての解釈は物理ノートへ分ける。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/30_Gauge_正準形_直交中心/43_TT_MPSのSVD中心移動とSchmidt形]]：exact SVDの中心移動、cut unfolding、Schmidt状態、ノルムと全要素例。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/45_TT_MPSの単一ボンドSVD打ち切り]]：$\rho\to k$ のコアshape、添字、射影、単一cut誤差と8要素例。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/46_Eckart_Young_Mirsky定理とTT_MPS単一ボンド打ち切りの最適性]]：単一ボンドのrank-$k$ 打ち切りが最良近似になる理由と、TT全体の逐次近似との区別。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/47_TT-SVDの準最適性と誤差上界]]：TT-SVDの局所誤差の直交分解、基本誤差上界、最良TT近似に対する準最適性。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/48_TT-roundingの定義と正準化sweep]]：既存TTを対象とする右→左QRと左→右SVD、各matricization・shape・計算量。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/49_TT-roundingの環境行列と誤差直交分解]]：局所残差の全体埋め込み、gauge依存性、正準環境の等長性、実局所誤差の二乗和。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/40_打ち切り_rounding_誤差評価/50_TT-roundingの誤差予算とrank選択]]：相対・絶対誤差予算、特異値尾部による最小rank、境界条件、rank cap、数値比較。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/50_縮約_ノルム_代数演算/51_TT_MPSの内積とenvironment縮約]]：TT成分表示からleft environment更新式を導き、dense復元なしの内積、複素MPSの共役、計算量へ接続する。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/50_縮約_ノルム_代数演算/52_TT_MPSのFrobeniusノルムと距離]]：自己内積からノルムを、3つの内積からTT間距離を導き、丸め誤差の扱いを整理する。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/50_縮約_ノルム_代数演算/53_TT_MPSの加減算とrank増加]]：コアの直和・block diagonal構成、交差項、加減算後のrank上界とroundingの必要性を導く。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/54_TT-matrixの定義とKronecker積表現]]：行列の入出力添字を各siteへ分けるTT-matrix表現。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/55_dense重みのTT-matrix tensorizationとTT-SVD初期化]]：dense重みのgrouped / interleaved順序とTT-SVD初期化。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/56_TT-matrixのdense reconstruction]]：core縮約からdense重みへ戻す添字とshape。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/57_TT-Linear_forwardの縮約とshape]]：dense重みを構築しないforward縮約。
- [[00_基礎理論/01_数学基礎/02_テンソル代数/60_TT_matrix_TTLinear基礎/58_TTLinearとdense_Linearのforward等価性]]：小さいLinearでの転置、bias、全要素の等価性。
## 02_ニューラルネットワーク基礎

- [[00_基礎理論/02_ニューラルネットワーク基礎/02_nn.Linearとは]]
- [[00_基礎理論/02_ニューラルネットワーク基礎/15_CNNとConv2dの基礎]]
- [[00_基礎理論/02_ニューラルネットワーク基礎/20_Global Average PoolingとCIFAR10モデル設計]]

## 03_モデル圧縮理論

- [[00_基礎理論/03_モデル圧縮理論/README]]
- [[00_基礎理論/03_モデル圧縮理論/04_Linear層を2層へ置き換える]]
- [[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]
- [[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/README]]：TTLinearによるNN圧縮理論の索引。
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/59_TTLinearによるMLP一層置換とlogits等価性]]：学習済みMLPの一層をexact TTLinearへ置換したときのlogits等価性。
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/60_TT-rank打ち切りによるLinear出力誤差とnorm上界]]：重み誤差から層出力誤差への恒等式と2本のnorm上界。
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/61_Lipschitz連続性とReLUの1-Lipschitz性]]：Lipschitz連続性の量化、ReLUの4場合証明、tensorへの拡張。
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/62_TT-matrixのmode分解とrank設計]]：mode因子、site size、rank、パラメータ数の結合した設計。
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/63_ReLUによるhidden_activation誤差伝播]]：第1 Linearの近似誤差をReLU後のhidden activation誤差へ接続する。
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/64_第2Linearによるlogits誤差伝播と合成上界]]：hidden errorからlogits errorへの恒等式と、第1 Linearからの合成上界。
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/65_ClassificationMarginとArgmax安定性]]：1 sampleのclassification margin、$\ell_\infty$ logits error、argmax不変の十分条件と直接判定。
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/66_BatchLogitsErrorからSample-wisePredictionStabilityへ]]：batch Frobenius誤差からsampleの$\ell_\infty$誤差、classification margin、prediction stability certificateへ至る途中式。
- [[00_基礎理論/03_モデル圧縮理論/30_TTLinear圧縮/67_Batch内全予測不変の十分条件]]：minimum marginと共通上界を使い、1つのbatch内の全sampleを同時に保証する十分条件。

## 04_実験設計

- [[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]
- [[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]]
- [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]
- [[00_基礎理論/04_実験設計/25_Tucker_HOOI圧縮の評価設計]]

## PyTorch・Python操作への接続

置換、rank選択、誤差集計、学習・ベンチマークの数式と考え方は本章に残す。これらをPyTorch・Pythonでどう実行するかは、SVDなら [[05_SVD基礎実装検証/README]]、Tucker / HOOIなら [[06_Tucker基礎実装検証/README]] を参照する。

TT / MPSの実装教材はSVD・Tuckerの検証章とは別系統として、現時点では次のノートからたどる。

- [[08_TT_MPS基礎実装検証/27_TT_MPS基礎のPyTorch実装]]
- [[08_TT_MPS基礎実装検証/28_TT_MPS実装で使うPyTorch_Python操作メモ]]
- [[08_TT_MPS基礎実装検証/29_TT_cutとPyTorchのreshape_Kronecker順序]]
- [[08_TT_MPS基礎実装検証/45_TT-matrix線形層のPyTorch直接forward]]
- [[08_TT_MPS基礎実装検証/08_TT-roundingのPyTorch実装設計と検証]]
- [[08_TT_MPS基礎実装検証/09_TT内積_Frobeniusノルム_距離のPyTorch確認]]
- [[40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/00_TTLinear一層置換とlogits等価性]]
- [[08_TT_MPS基礎実装検証/13_TT-rank打ち切りとLinear出力誤差上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/14_TTLinear置換で使うPyTorchモデル操作]]
- [[08_TT_MPS基礎実装検証/15_ReLU_hidden_activation誤差伝播のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/16_第2Linearのlogits誤差伝播と合成上界のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/17_ClassificationMarginとArgmax安定性のPyTorch確認]]
- [[08_TT_MPS基礎実装検証/18_BatchLogitsErrorからPredictionStabilityCertificateのPyTorch確認]]
- [[08_TT_MPS基礎実装検証/19_MinimumMarginとBatch-widePredictionStabilityのPyTorch確認]]

## 06_手法間のつながり

- [[00_基礎理論/06_手法間のつながり/14_低ランク学習からテンソルネットワークへの発展]]
- [[00_基礎理論/06_手法間のつながり/15_TuckerとTTの比較]]
- [[00_基礎理論/06_手法間のつながり/17_TT_MPS学習ロードマップ]]

## 07_物理基礎

- [[00_基礎理論/07_物理基礎/README]]
- [[00_基礎理論/07_物理基礎/16_TT_MPSの物理解釈_縮約密度行列と局所写像]]：bond状態・Schmidt係数・縮約密度行列・局所写像。
- [[00_基礎理論/07_物理基礎/42_MPS正準形と直交中心の物理的意味]]：左右のブロック基底、中心移動、QRとSchmidt分解の違い。
- [[00_基礎理論/07_物理基礎/44_Schmidt打ち切りとボンド状態の物理的意味]]：有効状態、縮約密度行列、捨てた重みと再規格化。

## 推奨読み順

### SVD

```text
01_SVDとは
→ 02_SVD数式の導出
→ 03_SVDによる低ランク近似
→ 04_Linear 2層化
→ 05_圧縮率とRank
→ 06_誤差評価
→ 06_誤差評価_数式導出補足
→ 実験
```

### Tucker / HOOI

```text
20_数式の導出
→ 21_mode演算
→ 22_Tucker/HOSVD
→ 24_Tucker-2 Conv
→ 23_HOOI
→ 25_評価設計
→ 26_src実装
→ 06_Tucker基礎実装検証
```

### TT / MPS

```text
30_TT/MPSの定義
→ 31_TT-rankとunfolding
→ 32_TT-SVD
→ 33_TT-SVDの打ち切りと誤差
→ 34_基底変換とTT-rank不変性
→ 35_TT/MPSの等長写像と射影
→ 36_Gauge自由度と左QR直交化
→ 37_左ブロックと直交性の導出
→ 38_右QR直交化と右ブロック
→ 39_混合正準形への導入
→ 40_中心ノルムと内積の導出
→ 41_直交中心の移動
→ 43_SVD中心移動とSchmidt形
→ 45_単一ボンドSVD打ち切り
→ 46_Eckart–Young–Mirsky定理と単一ボンド打ち切りの最適性
→ 47_TT-SVDの準最適性と誤差上界
→ 48_TT-roundingの定義と正準化sweep
→ 49_TT-roundingの環境行列と誤差直交分解
→ 50_TT-roundingの誤差予算とrank選択
→ 08_TT-roundingのPyTorch実装設計と検証
→ 51_TT/MPSの内積とenvironment縮約
→ 52_TT/MPSのFrobeniusノルムと距離
→ 53_TT/MPSの加減算とrank増加
→ 09_TT内積・Frobeniusノルム・距離のPyTorch確認
→ 07_物理基礎（42_直交中心、44_Schmidt打ち切り）
→ 28_PyTorch/Python操作メモ
→ 27_TT/MPS基礎のPyTorch実装
→ 29_TT cutとreshape/Kronecker順序
→ 17_TT/MPS学習ロードマップ
→ 45_TT-matrix線形層のPyTorch直接forward
```
