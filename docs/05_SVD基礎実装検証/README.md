# SVD基礎実装検証

SVD編の基礎実装・corrected実験で確認したcontractと修正原則をまとめる章。ランダム行列や単体Linearのsanity checkだけでなく、SVD実験全体で修正した設計上の問題まで追跡する。

Tucker / HOSVD / HOOIはこの章へ混ぜず、[[06_Tucker基礎実装検証/README]] に分離する。

## ノート

1. [[05_SVD基礎実装検証/01_SVD実装の確認結果]]
2. [[05_SVD基礎実装検証/02_Linear層2層置換の確認結果]]
3. [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]

## 数式・理論

SVDの式を途中から確認する場合は、次をcanonicalな導出として使う。

- [[00_基礎理論/00_数式導出監査]]
- [[00_基礎理論/01_数学基礎/01_線形代数/01_SVDとは]]
- [[00_基礎理論/01_数学基礎/01_線形代数/02_SVD数式の導出]]
- [[00_基礎理論/01_数学基礎/01_線形代数/03_SVDによる低ランク近似]]
- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価]]
- [[00_基礎理論/01_数学基礎/01_線形代数/06_誤差評価_数式導出補足]]

特に、complete / reduced SVDのshape、$Wv_i=\sigma_i u_i$、特異値二乗とFrobenius norm、truncated SVD誤差、retained energy、spectral error、output errorまで途中式を残している。

## 実験結果との接続

分類実験は別章で管理する。

- [[10_MNIST_MLP_SVD/README]]
- [[20_FashionMNIST/README]]
- [[30_CIFAR10_CNN/README]]
- [[SVD実験まとめ]]

正式な数値は各実験章のcanonical出典を使う。MNIST / Fashion-MNISTは2026-09-12の完全保存rerunを優先し、[Notebook・CSV・重み・manifestの対応](../../results/10_svd/rerun_20260912T074153Z/README.md) から追跡する。CIFAR-10は既存corrected学習結果と、保存済み最終FT後重みの別計測runを区別する。original / using_src / historical Notebookは学習履歴として残す。

## この章で確認する主なcontract

```text
rank validation
shape / reconstruction
Linear 2層置換
baselineとのParameter非共有
device / dtype / requires_grad
validationでのrank選択
test leakage回避
DataLoader Generator条件
benchmark公平性
MACsとlatencyの分離
```

SVDで得たこれらの実験原則は、Tucker / HOOI以降でも引き継ぐ。
