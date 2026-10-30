# nn-compression-svd-dmrg

ニューラルネットワーク圧縮を、**SVD → Tucker / HOSVD / HOOI → TT / MPS → DMRG** の順に学習・実験するリポジトリ。

現在は、**SVD、Tucker / HOSVD / HOOI、TT / MPSの基礎とNN圧縮検証**まで進み、DMRGでは2-site局所更新の複数往復、energy停止判定、global residual・energy分散、bond上限比較まで確認している。

- MNIST / Fashion-MNIST / CIFAR-10でSVD低rank圧縮を検証
- Linear / Conv2dの分解、Fine-tuning、rank選択、model-wide rank allocationを実施
- CIFAR-10 CNNのConv2d weightへTucker-2を適用
- HOSVD / HOOIを自作し、TensorLyと数値照合
- HOSVD / HOOIを同条件でFine-tuning比較
- Tucker / HOOI処理を `src/nn_compression/` へ共通化
- SVDからHOOIまでの基礎理論を、shape・途中式・成立条件まで含めて再監査
- src全体のpublic API contractをレビューし、SVD〜Tucker/HOOIを **Core API v1** として固定する設計書を作成
- TT-SVD、Gauge自由度、正準形、Schmidt形、TT-rounding、TT-matrix / TT-Linearを小さいTensorで検証
- Fashion-MNISTのMLPとCNNへTT層を接続し、rank sweepとFine-tuningを検証
- DMRGの中心線形写像、2-site局所最適化、複数往復、停止判定、bond上限比較を小さいMPSで確認

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
  基礎実装 / TT-Linear / TT-Conv / Fashion-MNIST検証
        ↓
DMRG
  2-site局所更新の複数往復 / 停止判定 / bond上限比較
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

## TT / MPS編

TT / MPSでは、chain型Tensor表現の基礎から、学習済みFashion-MNISTモデルへの接続までを段階的に確認した。

```text
TT-SVD / TT-rank
→ Gauge自由度 / 左右正準形 / 直交中心
→ Schmidt形 / 単一bond打ち切り
→ TT-rounding / TT内積 / Frobeniusノルム
→ TT-matrix / TT-Linear / TT-Conv
→ Fashion-MNIST MLP・CNNでのrank sweepとFine-tuning
```

Fashion-MNIST MLPでは、`fc1`のtensorization・rank・更新範囲を比較した。保存済みの固定条件では、dense accuracyから0.001以内という許容差規則により、rank 12・TT coresのみ更新を圧縮優先の暫定候補とした。ただしrank 12は探索下端であり、さらに低rankを調べる余地がある。

Fashion-MNIST CNNでは、`conv2`をTT-Convへ置換した。rank制限なしの置換でdense modelとのforward・backward整合性を確認した後、rank sweepとTT cores-only Fine-tuningを行った。rank 12は開発用accuracyが0.6910から0.9238へ回復したが、test性能、全parameter更新との比較、速度・peak memoryの優位性は未評価である。

詳細：

- [TT/MPS基礎実装検証](docs/08_TT_MPS基礎実装検証/README.md)
- [TT/MPSによるNN圧縮](docs/40_TT_MPS_NN圧縮/README.md)
- [Fashion-MNIST MLPのTT圧縮](docs/40_TT_MPS_NN圧縮/10_FashionMNIST_MLP/README.md)
- [Fashion-MNIST CNNのTT圧縮](docs/40_TT_MPS_NN圧縮/20_FashionMNIST_CNN/README.md)

## DMRG入門

DMRGでは、3-site実数MPSでの局所更新から始め、横磁場Ising HamiltonianのMPOと左右environmentを用いた2-site局所更新を複数往復へ接続した。energy安定による停止と解の精度を区別し、bond上限2・4・無切り捨てを共通初期状態から比較した。

```text
固定した左右環境
→ 中心から全状態への線形写像 x = Pc
→ 有効Hamiltonian H_eff = P^T H P
→ 最小固有値問題による2-site中心の更新
→ SVDによる2コアへの再分割
→ bond dimension打ち切り
→ 左右environmentの逐次更新
→ left-to-right / right-to-leftの1往復
→ 1往復を反復単位とする複数往復
→ energy安定による停止と全系診断
```

正準環境では $P^{\mathsf T}P=I$ となり、局所問題が通常の固有値問題へ簡約されることを整理した。小さい数値例では、更新前エネルギー $-0.936$、局所最適化後 $-2$、選択したrank-1打ち切り後 $-1$ を得ており、エネルギー最小化とSVDによる状態近似が別の目的であることを確認した。

現時点で確認済みなのは複数往復、energy停止判定、global residual・energy分散、bond上限比較までである。反復固有値solverとNN目的関数への接続は今後の範囲である。

- [DMRG章](docs/50_DMRG/README.md)
- [中心テンソルの線形写像と2-site局所更新](docs/50_DMRG/02_中心テンソルの線形写像と2-site局所更新.md)
- [MPOとenvironmentによる1-site局所有効問題](docs/50_DMRG/03_MPOとenvironmentによる1-site局所有効問題.md)
- [environmentを用いた2-site DMRG 1往復](docs/50_DMRG/04_environmentを用いた2-site_DMRG_1往復.md)
- [複数往復・停止判定・bond上限比較](docs/50_DMRG/05_複数往復_停止判定_bond上限比較.md)
- [2-site DMRG局所更新の演習Notebook](notebooks/40_dmrg/00_fundamentals/00_two_site_dmrg_local_update_exercises.ipynb)
- [MPO・environment・1-site局所更新の演習Notebook](notebooks/40_dmrg/00_fundamentals/01_mpo_environment_one_site_local_update_exercises.ipynb)
- [environmentを逐次更新する2-site DMRG 1往復演習Notebook](notebooks/40_dmrg/00_fundamentals/02_environment_two_site_dmrg_sweep_exercises.ipynb)
- [複数往復・停止判定・bond上限比較の演習Notebook](notebooks/40_dmrg/00_fundamentals/03_dmrg_convergence_and_bond_dimension_exercises.ipynb)

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
│  ├─ conv_tucker.py       # Conv2d Tucker-2
│  ├─ tt.py                # TT-SVD / reconstruction
│  ├─ tt_canonical.py      # 正準化
│  ├─ tt_contraction.py    # TT縮約
│  ├─ tt_rounding.py       # TT-rounding
│  ├─ tt_matrix.py         # TT-matrix
│  ├─ tt_linear.py         # TT-Linear
│  └─ tt_conv.py           # TT-Conv
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

- [src設計書](docs/90_src設計/README.md)
- [アーキテクチャ設計](docs/90_src設計/01_アーキテクチャ設計.md)
- [Core API v1 / 関数仕様](docs/90_src設計/05_Core_API_v1.md)
- [テスト設計](docs/90_src設計/06_テスト設計.md)
- [実装編](docs/README_実装編.md)
- [Tucker基礎実装検証](docs/06_Tucker基礎実装検証/README.md)

## 次

物理DMRGの基礎Notebookは03で一区切りとし、次はFashion-MNIST TT-MLPに対するDMRG-likeな2-core局所更新へ接続する。

NNへ適用する段階では、量子Hamiltonianの最小固有値問題をそのまま移すのではなく、教師あり損失、局所更新変数、局所solver、通常のgradient-based Fine-tuningとの比較条件を別途定義する。
