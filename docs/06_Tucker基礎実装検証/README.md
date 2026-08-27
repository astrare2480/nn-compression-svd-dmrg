---
title: Tucker基礎実装検証
aliases:
  - Tucker HOSVD HOOI 検証
  - Tucker実験まとめ
tags:
  - Tucker
  - HOSVD
  - HOOI
  - CIFAR10
  - NN圧縮
---

# Tucker基礎実装検証

`05_SVD基礎実装検証` はSVD編の確認結果に限定しているため、Tucker/HOSVD/HOOIは別章としてここへまとめる。

この章の目的は、Notebookを開かなくても、

- 何を実装したか
- どのshape契約を確認したか
- rank sweepで何が起きたか
- fine-tuningでどう回復したか
- HOOIがHOSVDをどう改善したか
- TensorLyとの一致
- weight errorとtask accuracyの違い
- src化後に何を回帰テストしたか

を追跡できる状態にすること。

## ノート

1. [[06_Tucker基礎実装検証/01_Tucker_HOSVD基礎実装の確認結果]]
2. [[06_Tucker基礎実装検証/02_Tucker2_Convとrank_sweepの確認結果]]
3. [[06_Tucker基礎実装検証/03_HOOIとTensorLy照合]]
4. [[06_Tucker基礎実装検証/04_HOSVD_HOOI_FineTuning比較]]
5. [[06_Tucker基礎実装検証/05_Tucker実験で得た設計原則と考察]]

## canonical source

```text
notebooks/20_tucker/00_fundamentals/
├─ 00_tucker_hosvd_basics.ipynb
├─ 01_rank_error_tradeoff.ipynb
└─ 02_hooi.ipynb

notebooks/20_tucker/10_cifar10_cnn/
├─ 01_tucker2_conv.ipynb
├─ 02_rank_sweep.ipynb
├─ 03_finetuning.ipynb
├─ 04_hooi_tucker2.ipynb
└─ 05_hosvd_vs_hooi_finetuning.ipynb
```

`*_before_src.ipynb` はsrc共通化前のhistorical snapshotとして残す。

結果の数値は `results/20_tucker/...` のCSVを優先して引用する。

## 基礎理論

- [[00_基礎理論/01_数学基礎/02_テンソル代数/21_テンソルとmode演算]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/22_Tucker分解とHOSVD]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/23_HOOI]]
- [[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]
- [[00_基礎理論/04_実験設計/25_Tucker_HOOI圧縮の評価設計]]
- [[00_基礎理論/05_PyTorch実装/26_Tucker_HOOIのPyTorch実装]]
