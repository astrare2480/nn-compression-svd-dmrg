# nn-compression-svd-dmrg

ニューラルネットワーク圧縮を、**SVD → Tucker → TT/MPS → DMRG** の順に学習・実験するリポジトリ。

現在はSVD編のcorrected実験まで完了し、MNIST / Fashion-MNIST / CIFAR-10で、Linear・Conv2dの低rank化、Fine-tuning、rank選択、model-wide rank allocationを検証している。

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
CIFAR-10 CNN
  Conv1 / Conv2 / Conv3 model-wide rank allocation
        ↓
Tucker decomposition
        ↓
TT / MPS
        ↓
DMRG
```

## SVD編のまとめ

- [SVD実験まとめ](docs/SVD実験まとめ.md)
- [基礎理論](docs/00_基礎理論/README.md)
- [SVD実験で修正した問題と設計原則](docs/05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則.md)
- [MNIST結果](docs/10_MNIST_MLP_SVD/04_MNISTでの実験結果.md)
- [Fashion-MNIST MLP結果](docs/20_FashionMNIST/04_Fashion-MNISTでの実験結果.md)
- [Fashion-MNIST CNN結果](docs/20_FashionMNIST/11_CNNでの実験結果.md)
- [CIFAR-10結果](docs/30_CIFAR10_CNN/01_CIFAR10_SVD実験.md)

## canonical / historical

実験結果を引用するときは、原則として `*_corrected.ipynb` と対応するcorrected resultsを使用する。

```text
corrected
→ 現在のcanonicalな実験結果

using_src
→ src共通化時点のsnapshot

original / before_src
→ 初期実験・historical record
```

historical Notebookは削除せず、どの問題があり、なぜcorrected版へ修正したかを学習履歴として残す。

## SVD編で得た主な知見

- truncated SVDで大幅にparameter数を削減しても、Fine-tuningによりtask accuracyをほぼ維持できる場合がある。
- retained energyはweight近似の指標であり、task accuracyそのものではない。
- rank選択ではtestを使わず、validationでcandidateを決める。
- candidate比較ではseed / DataLoader Generator条件を揃える。
- 理論MACs削減はwall-clock latency短縮を保証しない。
- CIFAR-10のrank探索は全rank空間のglobal optimumではなく、各層のPareto / knee近傍に制約したmodel-wide rank allocationである。
- single seedの小さなaccuracy差は「改善」と強く主張せず、「精度維持」と解釈する。

## コード

再利用可能な処理は `src/nn_compression/` に分離している。

```text
compression/  SVD / Linear / Conv2d factorization
models/       MLP / Fashion-MNIST CNN / CIFAR-10 CNN
training/     学習・Early Stopping・評価用loader
metrics/      parameters / MACs / latency / agreement / logits RMSE
selection/    Pareto / knee
```

現在のSVD実装は、rank・device・dtype・requires_grad、baselineとのParameter非共有、公平なbenchmark等の契約をテストとコメントで明示している。

## 次

SVDではConv2d weightを行列化して2次元の低rank近似を行った。

次のTucker decompositionでは、Conv weightの多モード構造をより直接扱う。
