# nn-compression-svd-dmrg

ニューラルネットワーク圧縮を、**SVD → Tucker / HOSVD / HOOI → TT / MPS → DMRG** の順に学習・実験するリポジトリ。

現在は、**SVD編のcorrected実験と、Tucker / HOSVD / HOOI編まで完了**している。

- MNIST / Fashion-MNIST / CIFAR-10でSVD低rank圧縮を検証
- Linear / Conv2dの分解、Fine-tuning、rank選択、model-wide rank allocationを実施
- CIFAR-10 CNNのConv2d weightへTucker-2を適用
- HOSVD / HOOIを自作し、TensorLyと数値照合
- HOSVD / HOOIを同条件でFine-tuning比較
- Tucker / HOOI処理を `src/nn_compression/` へ共通化
- SVDからHOOIまでの基礎理論を、shape・途中式・成立条件まで含めて再監査
- src全体のpublic API contractをレビューし、SVD〜Tucker/HOOIを **Core API v1** として固定する設計書を作成

## 現在の到達点

```text
MNIST MLP
  Linear SVD / validation-based rank selection
        ↓
Fashion-MNIST MLP
  Pareto / knee / Fine-tuning
        ↓
Fashion-MNIST CNN
  Linear SVD / Conv SVD / Conv + Linear
        ↓
CIFAR-10 CNN / SVD
  Conv1 / Conv2 / Conv3 model-wide rank allocation
        ↓
Tucker / HOSVD
  Tensor mode演算 / Tucker-2 Conv / rank sweep
        ↓
HOOI
  iterative refinement / TensorLy照合 / Fine-tuning比較
        ↓
Core API v1
  public contract freeze
        ↓
TT / MPS
        ↓
DMRG
```

## 基礎理論

基礎理論は [docs/00_基礎理論](docs/00_基礎理論/README.md) にまとめている。

今回、最終式だけでなく**途中式まで追えること**を基準に、既存SVDノートを含めて数式を再監査した。

特に長い導出は次をcanonicalとする。

- [数式導出監査](docs/00_基礎理論/00_数式導出監査.md)
- [SVD数式の導出](docs/00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出.md)
- [誤差評価の数式導出補足](docs/00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価_数式導出補足.md)
- [Tucker / HOSVD / HOOI数式の導出](docs/00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出.md)

数式では、定義だけでなく必要に応じて、

```text
shape / 次元
→ 一般式
→ 途中の変形
→ 最終式
→ 成立条件
→ 実験での解釈
```

まで残す方針としている。

## SVD編

### まとめ

- [SVD実験まとめ](docs/SVD実験まとめ.md)
- [SVD実験で修正した問題と設計原則](docs/05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則.md)
- [MNIST結果](docs/10_MNIST_MLP_SVD/04_MNISTでの実験結果.md)
- [Fashion-MNIST MLP結果](docs/20_FashionMNIST/04_Fashion-MNISTでの実験結果.md)
- [Fashion-MNIST CNN結果](docs/20_FashionMNIST/11_CNNでの実験結果.md)
- [CIFAR-10結果](docs/30_CIFAR10_CNN/01_CIFAR10_SVD実験.md)

### 主な知見

- truncated SVDで大幅にparameter数を削減しても、Fine-tuningでtask accuracyをほぼ維持できる場合がある。
- retained energyはweight近似の指標であり、task accuracyそのものではない。
- rank選択ではtestを使わず、validationでcandidateを決める。
- candidate比較ではseed / DataLoader Generator条件を揃える。
- 理論MACs削減はwall-clock latency短縮を保証しない。
- CIFAR-10のrank探索は全rank空間のglobal optimumではなく、各層のPareto / knee近傍へ制約したmodel-wide rank allocationである。
- single seedの小さなaccuracy差は「改善」と強く主張せず、「精度維持」と解釈する。

## Tucker / HOSVD / HOOI編

CIFAR-10 CNNの学習済み `conv2.weight=(64, 32, 3, 3)` を主対象に、channel mode 0 / 1をTucker-2で圧縮した。

```text
C_in
→ 1x1 / U_in^T
→ R_in
→ kxk / core
→ R_out
→ 1x1 / U_out
→ C_out
```

balanced rank `(R_out, R_in)=(32,16)` では、

```text
model parameters       128,842 → 117,578
Conv2 MAC reduction              61.11%
model MAC reduction              27.84%
```

となった。

同rankでHOOIを適用すると、weight relative errorは

```text
HOSVD  0.449042
HOOI   0.442504
```

へ低下した。自作HOOIとTensorLy `partial_tucker` は、最終weight errorと圧縮直後accuracyで整合した。

一方、圧縮直後validation accuracyは

```text
HOSVD  0.6280
HOOI   0.6202
```

であり、**weight Frobenius誤差の改善がtask accuracy改善を保証しない**ことも確認した。

同条件Fine-tuning（seed 0）では、

```text
post validation
HOSVD  0.7528
HOOI   0.7528

post test
HOSVD  0.7508
HOOI   0.7555
```

となった。ただし1 seed・0.0047差なので、HOOIの最終accuracy上の一般的優位性は主張しない。

さらにFine-tuning後は元weightへのrelative errorが増えながらaccuracyが改善し、

```text
元weightへの近さ
≠
taskにとって最適なlow-rank weight
```

であることも実測した。

詳細：

- [Tucker基礎実装検証](docs/06_Tucker基礎実装検証/README.md)
- [Tucker-2 Conv / rank sweep](docs/30_CIFAR10_CNN/02_Tucker2_Convとrank_sweepの確認結果.md)
- [HOOI / TensorLy照合](docs/30_CIFAR10_CNN/03_HOOIとTensorLy照合.md)
- [HOSVD / HOOI Fine-tuning比較](docs/30_CIFAR10_CNN/04_HOSVD_HOOI_FineTuning比較.md)
- [Tucker実験で得た設計原則と考察](docs/06_Tucker基礎実装検証/05_Tucker実験で得た設計原則と考察.md)

## canonical / historical

正式な実験結果を引用するときは、SVDでは原則として `*_corrected.ipynb` と対応するcorrected resultsを使用する。

Tucker / HOOIでは、

```text
notebooks/20_tucker/
+
results/20_tucker/
```

の現行Notebook・CSVをcanonicalとする。

```text
corrected
→ canonicalなSVD実験結果

using_src
→ src共通化時点のsnapshot

original
→ 初期実験・学習履歴

before_src
→ src共通化前のhistorical snapshot
```

historical Notebookは削除せず、何が問題で、なぜcorrected版・src版へ修正したかを学習履歴として残す。

## コード

再利用可能な処理は `src/nn_compression/` に分離している。

```text
src/nn_compression/
├─ tensor/
│  ├─ operations.py        # unfold / fold / mode product
│  └─ validation.py        # tensor shape / mode contract
├─ compression/
│  ├─ svd.py
│  ├─ linear_svd.py
│  ├─ conv_svd.py
│  ├─ tucker.py            # HOSVD / Tucker reconstruction
│  ├─ tucker_validation.py # Tucker/HOOI contract
│  ├─ hooi.py              # generic / partial HOOI
│  └─ conv_tucker.py       # Conv2d Tucker-2
├─ models/
├─ training/
├─ metrics/
├─ selection/
├─ datasets/
└─ utils/
```

Notebookは学習過程・自作実装を残し、再利用可能な処理をsrcへ共通化する方針としている。

現行srcで固定する主なcontract：

- Tensor / Tucker coreは2次元以上・各dimension正
- mode / rankでboolを整数として受理しない
- Tucker `ranks` はMapping、rank上限はmode-n unfoldingの最大rank
- HOSVD / HOOIと、それらを使うTucker-2分解APIは現時点で**実数浮動小数点Tensor限定**
- HOOI feasibilityは `max_iter=0` でも反復前に検証
- Tucker-2のTensor-level HOOIは入力weightをdetachせず、Module構築時にautogradを切る
- 評価・ベンチマーク後はrootだけでなく全submoduleのtrain/eval状態を復元
- empty loaderは明示的に拒否し、対応箇所では `len(loader)` を仮定しない
- MACs rank / 出力空間size、benchmark countではTensor scalarを整数として受理しない

src contract reviewのbaseline `49836bc` ではローカル全pytest `252 passed / 0 failed` を確認済み。Core API v1 self reviewで追加したvalidation/test差分については、GitHub CIが無いため最終merge前にローカルpytest確認を行う。

詳細：

- [src設計書](docs/07_src設計/README.md)
- [アーキテクチャ設計](docs/07_src設計/01_アーキテクチャ設計.md)
- [Core API v1 / 関数仕様](docs/07_src設計/05_Core_API_v1.md)
- [テスト設計](docs/07_src設計/06_テスト設計.md)
- [実装編](docs/README_実装編.md)
- [Tucker基礎実装検証](docs/06_Tucker基礎実装検証/README.md)

## 次

次は **Tensor Train / MPS**。

SVDで学んだ「低rank近似」、Tucker/HOOIで学んだ「Tensor mode・反復更新・sweep・収束」の考え方を、TT rank / bond dimensionを持つchain型Tensor networkへつなげ、その後DMRGへ進む。
