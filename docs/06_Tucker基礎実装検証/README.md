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

`05_SVD基礎実装検証` はSVD編の確認結果に限定しているため、Tucker / HOSVD / HOOIは別章としてここへまとめる。

この章の目的は、Notebookを開かなくても、

- 何を実装したか
- どのshape契約を確認したか
- rank sweepで何が起きたか
- Fine-tuningでどう回復したか
- HOOIがHOSVDをどう精密化したか
- TensorLyとどこまで一致したか
- weight errorとtask accuracyがなぜ一致しないか
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

正式な結果値は `results/20_tucker/...` のCSVを優先して引用する。別Notebook / 別runの値は同一runとして混ぜない。

## 数式・基礎理論

数式を途中から追う場合は、まず次を読む。

- [[00_基礎理論/00_数式導出監査]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/21_テンソルとmode演算]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/22_Tucker分解とHOSVD]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/23_HOOI]]
- [[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]
- [[00_基礎理論/04_実験設計/25_Tucker_HOOI圧縮の評価設計]]
- [[00_基礎理論/05_PyTorch実装/26_Tucker_HOOIのPyTorch実装]]

`20_Tucker_HOSVD_HOOI数式の導出` では、unfold / fold / mode productから、HOSVD、Tucker-2の3層forward、HOOIの局所最適化・sweep・収束まで途中式をまとめている。

## src化後の責務

```text
src/nn_compression/
├─ tensor/
│  └─ operations.py
├─ compression/
│  ├─ tucker.py
│  ├─ hooi.py
│  └─ conv_tucker.py
└─ metrics/
   └─ tensor_approximation.py
```

```text
operations.py
→ unfold / fold / mode_dot

tucker.py
→ HOSVD / Tucker再構成

hooi.py
→ layer非依存のgeneric / partial HOOI

conv_tucker.py
→ Conv2d固有のTucker-2構築・HOOI helper

tensor_approximation.py
→ relative Frobenius error
```

HOOIでは `ranks` のkeyを更新対象modeとして扱うため、全mode HOOIとpartial Tucker-2を同じ汎用実装で扱う。1 sweep内では更新済みfactorを次のmode更新に使うGauss-Seidel型とし、入力factorは破壊しない。

## 現在の到達点

CIFAR-10 CNNの `conv2.weight=(64,32,3,3)` を主対象に、Tucker-2 rank sweep、HOSVD Fine-tuning、HOOI、TensorLy照合、HOSVD/HOOI同条件Fine-tuningまで完了した。

balanced rank `(32,16)` では、HOOIがHOSVDよりweight relative errorを下げた一方、圧縮直後accuracyは改善しなかった。このため、

```text
weight approximation quality
≠
task performance
```

を実測結果として残している。

同条件Fine-tuningの最終test差は1 seedで0.0047なので、HOOIの一般的なaccuracy優位性とは解釈しない。

## 回帰テスト

Tucker/HOOI src整理後のローカル全pytestでは、

```text
99 passed
0 failed
```

を確認している。

これはローカル実行結果であり、GitHub CIによる独立確認を意味しない。

## 関連

- [[30_CIFAR10_CNN/README]]
- [[README_実装編]]
- [[40_TensorTrain_MPS/README]]
