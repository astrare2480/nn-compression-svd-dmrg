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

`05_SVD基礎実装検証` はSVD編のPyTorch操作と確認結果に限定しているため、Tucker / HOSVD / HOOIのPyTorch操作と確認結果は別章としてここへまとめる。評価の理論は `00_基礎理論` に残す。

この章の目的は、Notebookを開かなくても、

- 何を実装したか
- どのshape / mode / rank contractを確認したか
- rank sweepで何が起きたか
- Fine-tuningでどう回復したか
- HOOIがHOSVDをどう精密化したか
- TensorLyとどこまで一致したか
- weight errorとtask accuracyがなぜ一致しないか
- src化後に何を回帰テストしたか
- 現行public APIがどの入力を明示的に拒否するか

を追跡できる状態にすること。

## PyTorch操作

- [[06_Tucker基礎実装検証/02_Tucker2_Convの実装契約と注意]]
- [[06_Tucker基礎実装検証/10_Tucker2_Conv2dのPyTorch契約]]
- [[06_Tucker基礎実装検証/20_Tucker数式のPyTorch確認コード]]
- [[06_Tucker基礎実装検証/20_CIFAR10モデルのPyTorch確認コード]]
- [[06_Tucker基礎実装検証/25_評価設計のPyTorchコード]]
- [[06_Tucker基礎実装検証/26_Tucker_HOOIのPyTorch実装]]

数式の導出は [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]]、評価設計は [[00_基礎理論/04_実験設計/25_Tucker_HOOI圧縮の評価設計]]、現行の共通関数の契約は [[07_src設計/05_Core_API_v1]] を参照する。

## 実装・実験の確認結果

1. [[06_Tucker基礎実装検証/01_Tucker_HOSVD基礎実装の確認結果]]
2. [[30_CIFAR10_CNN/02_Tucker2_Convとrank_sweepの確認結果]]
3. [[30_CIFAR10_CNN/03_HOOIとTensorLy照合]]
4. [[30_CIFAR10_CNN/04_HOSVD_HOOI_FineTuning比較]]
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

- [[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/21_テンソルとmode演算]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/22_Tucker分解とHOSVD]]
- [[00_基礎理論/01_数学基礎/02_テンソル代数/23_HOOI]]
- [[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]

`20_Tucker_HOSVD_HOOI数式の導出` では、unfold / fold / mode productから、HOSVD、Tucker-2の3層forward、HOOIの局所最適化・sweep・収束まで途中式をまとめている。

## src化後の責務

現行 `src` では、アルゴリズム本体とpublic contractを分けている。

```text
src/nn_compression/
├─ tensor/
│  ├─ operations.py
│  └─ validation.py
├─ compression/
│  ├─ tucker.py
│  ├─ tucker_validation.py
│  ├─ hooi.py
│  └─ conv_tucker.py
├─ metrics/
│  ├─ tensor_approximation.py
│  ├─ macs.py
│  └─ model_comparison.py
└─ training/
   └─ loops.py
```

```text
operations.py
→ unfold / fold / mode_dot

validation.py
→ Tensor shape / modeの共通contract

tucker.py
→ HOSVD / Tucker再構成 / Tucker要素数

tucker_validation.py
→ Tucker rank / dtype / tolerance / max_iter contract

hooi.py
→ layer非依存のgeneric / partial HOOI

conv_tucker.py
→ Conv2d固有のTucker-2構築・HOOI helper

tensor_approximation.py
→ relative Frobenius error

macs.py
→ 圧縮前後の理論MACs

model_comparison.py
→ accuracy / agreement / logits RMSE / latency等

training/loops.py
→ train / evaluate とtrain/eval状態管理
```

HOOIでは `ranks` のkeyを更新対象modeとして扱うため、全mode HOOIとpartial Tucker-2を同じ汎用実装で扱う。1 sweep内では更新済みfactorを次のmode更新に使うGauss-Seidel型とし、入力factorは破壊しない。

## 現行public API contract

### Tensor shape / mode

- `unfold` / `mode_dot` の対象Tensorは2階以上
- shapeの各dimensionは正
- modeはboolではない整数で、有効範囲内
- `fold` は2次元行列、row/column/numel整合まで確認

### Tucker ranks

- `ranks` は `Mapping` が必要
- bool rankは拒否
- HOSVDでは空 `{}` をidentityとして許可
- HOOIでは空rankを拒否
- rank上限はmode-n unfoldingの最大rankで判定
- projected unfolding上で実現不可能なHOOI rankも反復前に拒否

### dtype

現行HOSVD/HOOI/Tucker-2分解は**実数Tensor限定**。factor射影が `U.T` 前提のため、複素Tensorは誤った結果を黙って返さず `TypeError` で拒否する。複素対応を行う場合は共役転置を含むアルゴリズム仕様変更として別途扱う。

### HOOI iteration

- `max_iter` はboolではない0以上の整数
- `abs_tol` / `rel_tol` は有限かつ0以上
- rank feasibilityは `max_iter=0` の場合でも検証
- zero Tensorはrelative Frobenius errorを定義できないためHOOIで拒否

### Autograd境界

- `tucker2_hooi()` は低レベルTensor APIなので入力weightをdetachしない
- `build_tucker2_conv()` はmodule初期化処理なので元weightをdetachし、新しいleaf Parameterへcopyする
- 元Convと置換後3層でParameter storageを共有しない
- device / dtype / requires_gradを維持する

### train / eval状態

`evaluate()` とmodel comparison系は、一時的に `eval()` へ切り替えてもrootだけでなく全submoduleの `.training` 状態を個別保存・復元する。frozen BatchNormのような「root=trainだが一部submodule=eval」の状態も正常終了・例外時の両方で保持する。

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

src review / contract整理後のローカル全pytestでは、

```text
252 passed
0 failed
```

を確認している。

Tucker/HOOIに加えて、今回のsrc reviewでは次も回帰対象になった。

```text
ndim / positive shape contract
bool rank / Mapping contract
mode-n unfolding最大rank
複素dtype拒否
HOOI feasibilityのmax_iter非依存
Tucker-2 autograd境界
empty DataLoader / IterableDataset
全submoduleのtrain/eval状態復元
benchmark warmup / repeats validation
```

これはローカル実行結果であり、GitHub CIによる独立確認を意味しない。

## 既知の制約 / 今後の整理候補

現行実装では、以下はCritical/Majorではなく後続課題として残している。

- NumPy scalarを各APIでどこまで受理するかは完全には統一していない
- `collect_compression_metrics()` は同じloaderを複数回走査するため、再走査可能なloaderを前提とする
- 整数validation helperは依存方向・循環import回避のため複数moduleに小さく重複している
- `tensor/validation.py` と `compression/tucker_validation.py` に類似helperがあるが、大規模refactorを避け現状は意図的に分離している

これらは実験結果を無効にする問題ではなく、現行public contractを理解したうえでTT/MPS以降へ進む。

## 関連

- [[30_CIFAR10_CNN/README]]
- [[README_実装編]]
- [[40_TensorTrain_MPS/README]]
