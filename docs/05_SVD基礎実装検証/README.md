# SVD基礎実装検証

SVD編のPyTorch・Python操作と、基礎実装・corrected実験で確認したcontractと修正原則をまとめる章。置換、rank選択、誤差集計、学習・ベンチマークの理論は `00_基礎理論` に残す。教材用のコード例と、Notebook・`src` で実際に確認した結果は区別する。

Tucker / HOSVD / HOOIはこの章へ混ぜず、[[06_Tucker基礎実装検証/README]] に分離する。

## PyTorch・Python操作

- [[05_SVD基礎実装検証/00_SVD理論のPyTorch確認コード]]
- [[05_SVD基礎実装検証/02_nn.LinearのPyTorch操作]]
- [[05_SVD基礎実装検証/04_Linear層置換のPyTorch手順]]
- [[05_SVD基礎実装検証/05_rank指標のPython集計と可視化]]
- [[05_SVD基礎実装検証/06_誤差評価のPyTorch集計]]
- [[05_SVD基礎実装検証/07_PyTorch実装]]
- [[05_SVD基礎実装検証/08_Linear層のSVD実装]]
- [[05_SVD基礎実装検証/09_Linear層の2層置換_実装]]
- [[05_SVD基礎実装検証/10_評価設計のPyTorchコード]]
- [[05_SVD基礎実装検証/11_ベンチマークのPyTorchコード]]
- [[05_SVD基礎実装検証/12_PyTorch学習と評価の基礎]]
- [[05_SVD基礎実装検証/13_PandasとPython実装メモ]]
- [[05_SVD基礎実装検証/15_CNNのPyTorch確認コード]]
- [[05_SVD基礎実装検証/16_Conv2d行列化のAutograd境界]]
- [[05_SVD基礎実装検証/17_Conv2d低ランク置換のPyTorch手順]]
- [[05_SVD基礎実装検証/19_再現性のPyTorchコード]]

評価設計は [[00_基礎理論/04_実験設計/10_SVD圧縮モデルの評価設計]]、再現性と乱数は [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]、理論MACs・パラメータ数の導出は [[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]] を参照する。現行の共通関数の契約は [[07_src設計/05_Core_API_v1]] を優先し、教材の一体型サンプルと混同しない。

## 実装・実験の確認結果

1. [[05_SVD基礎実装検証/01_SVD実装の確認結果]]
2. [[05_SVD基礎実装検証/02_Linear層2層置換の確認結果]]
3. [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]]

## 数式・理論

SVDの式を途中から確認する場合は、次をcanonicalな導出として使う。

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
