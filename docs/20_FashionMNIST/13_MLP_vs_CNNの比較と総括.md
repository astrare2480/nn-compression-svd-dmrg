---
title: MLP vs CNNの比較と総括
aliases:
  - Fashion-MNIST MLP CNN 比較
  - Linear SVD Conv SVD 比較
  - SVD圧縮 総括
  - CIFAR-10への移行
tags:
  - Fashion-MNIST
  - MLP
  - CNN
  - SVD
  - NN圧縮
  - FineTuning
  - CIFAR-10
---

# MLP vs CNNの比較と総括

## サマリー

Fashion-MNISTで、まずMLPのLinear層をSVD圧縮し、その後CNNでLinear SVDとConv SVDを試した。

最終結果を相対変化で並べると次のとおり。

| Model | 圧縮対象 | Parameters | Parameter削減率 | MACs | MAC削減率 | Test acc | ΔTest acc |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP Baseline | — | 535,818 | — | 535,040 | — | 88.71% | — |
| MLP SVD + FT | fc1=16, fc2=16 | 36,362 | **93.21%** | 35,584 | **93.35%** | 87.14% | -1.57pt |
| CNN Baseline | — | 421,642 | — | 4,241,152 | — | 91.75% | — |
| CNN SVD + FT | conv2=28, fc1=24 | 89,994 | **78.66%** | 2,237,184 | **47.25%** | 91.80% | +0.05pt |

> [!important]
> MLPとCNNではモデル構造・Baseline精度・圧縮対象が異なるため、Raw accuracyや削減率だけで「どちらの手法が優れている」とは結論しない。比較の目的は、**Linear SVDとConv SVDで何が違い、ParametersとMACsへどう効くかを整理すること**である。

今回の最重要点は次の3つ。

```text
MLP
→ ほぼLinearだけで構成
→ ParametersとMACsがほぼ同じ比率で減る

CNN
→ LinearはParametersを持ちやすい
→ Convはweightを空間方向に繰り返し使うためMACsを持ちやすい

Fine-tuning
→ truncated SVDで生じた近似誤差を、タスクlossに合わせて再調整する工程
```

---

## 1. MLPで分かったこと

MLP編では、`fc1` と `fc2` の2つのLinear層をSVDで2因子化した。

```text
Linear(in → out)
        ↓ SVD
Linear(in → r)
        ↓
Linear(r → out)
```

最終的に、

```text
fc1 rank = 16
fc2 rank = 16
```

を選択した。

```text
Parameters 535,818 → 36,362
           -93.21%

MACs       535,040 → 35,584
           -93.35%

Test acc   88.71% → 87.14%
           -1.57pt
```

非常に強い圧縮であり、parameterとMACsを約93%削減できた一方、Test accuracyは1.57ポイント低下した。

ここでは、**Linear中心のネットワークではparameter削減とMACs削減がほぼ同じ方向へ動く**ことが確認できた。

関連：

- [[20_FashionMNIST/03_Fashion-MNISTのFine-tuning]]
- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]

---

## 2. CNNで分かったこと

CNN編では、役割の異なる2種類の層を圧縮した。

```text
fc1   : Linear SVD
conv2 : Conv SVD
```

最終的に、

```text
conv2 rank = 28
fc1 rank   = 24
```

を同時に適用した。

```text
Parameters 421,642 → 89,994
           -78.66%

MACs       4,241,152 → 2,237,184
           -47.25%

Test acc   91.75% → 91.80%
```

Test accuracyの差は+0.05ポイントと非常に小さいため、ここでは「改善」ではなく**精度維持**と解釈する。

CNNでは、parameter数とMACsが同じ層に集中していなかった。

```text
fc1   : Parametersの95.23% / MACsの9.46%
conv2 : Parametersの 4.39% / MACsの85.18%
```

したがって、

```text
fc1 SVD   → モデルサイズ削減に強い
conv2 SVD → 理論演算量削減に強い
```

という役割分担になった。

関連：

- [[20_FashionMNIST/08_CNNのLinear SVD]]
- [[20_FashionMNIST/09_CNNのConv SVD]]
- [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
- [[20_FashionMNIST/11_CNNでの実験結果]]

---

# 3. Linear SVDとConv SVDの違い

## 3.1 共通点

どちらも本質は、学習済みweightを行列として見て、truncated SVDで低rank近似することである。

```text
W ≈ U_r Σ_r V_r^T
```

そのままdenseな近似weightへ戻すだけでは保存parameter数は減らない。実際に圧縮するには、SVD因子を**2層に分けたまま保持する**。

また、SVD直後のweightはランダム初期化ではなく、元の学習済みweightから作られた近似値である。

---

## 3.2 Linear SVD

Linearのweightは最初から2次元行列である。

```text
W: (out_features, in_features)
```

そのため、直接SVDできる。

```text
Linear(in → out)

W(out, in)
   ↓ truncated SVD
(out, r) × (r, in)

Linear(in → r)
→ Linear(r → out)
```

圧縮後の主なweight数は、

```text
元       : in × out
低rank化 : in × r + r × out
```

となる。

Linearでは1サンプルのforwardで各weightは基本的に1回の積和に使われるため、**weight数とMACsがかなり近い関係**になる。

MLP編で、

```text
Parameters -93.21%
MACs       -93.35%
```

とほぼ同じ削減率になったのは、この構造が大きい。

---

## 3.3 Conv SVD

Conv2dのweightは4次元Tensorである。

```text
(out_channels, in_channels, kH, kW)
```

今回の`conv2`は、

```text
(64, 32, 3, 3)
```

なので、そのままでは通常の行列SVDへ入れない。

出力フィルタを行、`in_channels × kH × kW` を列としてまとめ、

```text
(64, 32, 3, 3)
       ↓ flatten
(64, 288)
```

へ行列化する。

そして、

```text
(64, 288)
   ↓ truncated SVD
(64, r) × (r, 288)
```

と分解し、Convへ戻す。

```text
Conv2d(32 → 64, 3×3)

        ↓

Conv2d(32 → r, 3×3)
        ↓
Conv2d(r → 64, 1×1)
```

Linearとの大きな違いは、**Convの同じweightが複数の空間位置で繰り返し利用される**ことである。

したがって、Convはparameter数が小さくてもMACsが非常に大きくなり得る。

---

## 3.4 比較表

| 項目 | Linear SVD | Conv SVD |
|---|---|---|
| 元weight | 2次元 | 4次元 |
| SVD前処理 | そのまま | `(out, in, kH, kW) → (out, in×kH×kW)` |
| 圧縮後 | Linear + Linear | k×k Conv + 1×1 Conv |
| 中間次元 | rank `r` | rank channel `r` |
| Parametersへの効き方 | 大きいLinearなら非常に大きい | 元Convのweight数に依存 |
| MACsへの効き方 | parameter削減と近い | 空間サイズが大きいほど非常に大きい |
| 主な注意 | rankを上げすぎると2層化で逆に増える | 空間再利用・出力shape・stride/padding/dilationを考える |

---

# 4. ParametersとMACsの違い

## 4.1 Parameters

Parametersは、モデルが保持・学習する重みやbiasの数である。

主に関係するもの：

- モデルサイズ
- checkpointサイズの主要因
- 学習時に保持するweight / gradient / optimizer state

ただしparameter数だけでは、推論時に何回演算されるかは分からない。

---

## 4.2 MACs

MACsはMultiply-Accumulateの理論回数であり、forward時の演算量の目安である。

Linearなら、

```text
MACs ≈ in × out
```

なのでweight数とほぼ同じ。

一方Convでは、biasを除いて概念的には、

```text
MACs
≈ H_out × W_out
 × C_out × C_in × kH × kW
```

となる。

つまり、Convのweightを、

```text
H_out × W_out
```

個の出力位置で再利用する。

今回の`conv2`では、

```text
Parameters = 18,496
MACs       = 3,612,672
```

となり、少ないparameterでも大きな演算量を持っていた。

---

## 4.3 MLPとCNNの違いがここに出た

### MLP

```text
Baseline Parameters = 535,818
Baseline MACs       = 535,040
```

ほぼ同じオーダー。

圧縮後も、

```text
Parameter reduction = 93.21%
MAC reduction       = 93.35%
```

とほぼ同じだった。

### CNN

```text
Baseline Parameters =   421,642
Baseline MACs       = 4,241,152
```

MACsはparameter数の約10倍ある。

さらに層別に見ると、

```text
fc1   : parameter巨大 / MACsは相対的に小さい
conv2 : parameter小さい / MACs巨大
```

となる。

ここから、**圧縮では「モデルサイズを小さくしたい」のか「計算量を減らしたい」のかを先に決める必要がある**。

---

## 4.4 Parameters/MACsが減ってもlatencyは自動では減らない

MLPでもCNNでも、低rank化後は元の1層が2層になる。

そのため、理論MACsが減っても、実測latencyは必ずしも同じ割合では減らない。

候補要因：

- 演算呼び出し回数の増加
- 中間Tensor生成
- メモリアクセス
- kernel/backendの実装効率
- batch size
- CPU/GPU
- 小規模モデルで固定オーバーヘッドが支配的になること

したがって、今後も、

```text
Parameters
MACs
実測latency
Accuracy / loss
```

は独立した評価軸として残す。

---

# 5. Fine-tuning効果の比較

## 5.1 MLP

MLPのAggressive候補`16/16`はSVD直後、Rank-Selection Validationで大きく劣化していた。

```text
Validation acc
71.76% → 88.32%

+16.56pt
```

```text
Validation loss
0.858264 → 0.322567
```

非常に強い低rank化でも、Fine-tuningによって大部分を回復した。

一方、Conservative候補ではFine-tuning後にValidationが悪化したため、**Fine-tuningは必ず改善する処理ではない**ことも確認できた。

---

## 5.2 CNN

CNNではSVD直後から比較的高い性能を残していたため、Fine-tuningの回復量はMLPより小さい。

### fc1 rank24

```text
Validation acc
91.72% → 92.26%
+0.54pt
```

### conv2 rank28

```text
Validation acc
92.16% → 92.60%
+0.44pt
```

### conv2 rank28 + fc1 rank24

```text
Validation acc
91.24% → 92.04%
+0.80pt
```

同時圧縮では、単独圧縮よりSVD直後の誤差が大きかったが、Fine-tuningでかなり回復した。

---

## 5.3 Fine-tuningの意味

SVDは、

```text
weight行列の近似誤差
```

を小さくする方法であり、直接、

```text
分類loss
```

を最小化しているわけではない。

truncated SVDで不要と判断された成分が、タスク上は重要な場合もある。

Fine-tuningでは、SVD因子を初期値として、

```text
分類lossを下げる方向
```

へweightを再調整できる。

したがって今回の流れは、

```text
学習済みモデル
↓
truncated SVD
↓
低rank構造へ写す
↓
分類性能が一部低下
↓
Fine-tuning
↓
低rankという制約の中でタスクへ再適応
```

と理解できる。

MLPのAggressive候補が大幅回復したことは、**SVD直後の低いaccuracyだけを見て「このrankは使えない」と判断すると、圧縮可能性を過小評価する場合がある**ことを示している。

一方で、Fine-tuning後の性能までValidationで比較して最終rankを決める必要がある。

---

# 6. MLP vs CNNから得た総括

## 6.1 SVD圧縮の基本手順は共通

MLPとCNNで実装は違っても、実験設計は同じだった。

```text
Baseline学習
↓
ValidationでBest state選択
↓
weightをSVD
↓
rank sweep
↓
Pareto / knee
↓
Fine-tuning
↓
Validationで最終選択
↓
Testを最後に1回評価
```

この流れは、今後のデータセットでも共通の基準として使える。

---

## 6.2 圧縮対象はparameter数だけで選ばない

MLPではLinearがほぼ全体を占めるため、parameter数を見るだけでも圧縮対象がほぼ決まった。

CNNではそうならない。

今回、

```text
fc1   → Parameters支配
conv2 → MACs支配
```

だったため、**parameter内訳とMACs内訳の両方を最初に出すことが重要**だった。

---

## 6.3 「高いrankほど良い」とは限らない

rankを上げれば近似誤差は小さくなるが、圧縮率は悪化する。

さらに、元の1層を2層に分解するため、rankが大きすぎると元モデルよりParameters/MACsが増える場合もある。

したがってrank選択は、

```text
accuracy最大化
```

ではなく、

```text
圧縮率 × Validation性能
```

のトレードオフとして行う。

---

## 6.4 retained energyだけでも決めない

retained energyはweight行列のSVD近似として重要だが、分類性能とは別の指標である。

同じretained energyでも、

- 層の役割
- 特異値分布
- 下流層
- 非線形処理
- タスクへの感度

が異なる。

そのため、rankはretained energyだけではなく、Validation loss / accuracyと合わせて決める。

---

# 7. 次はCIFAR-10へ

ここからは、Fashion-MNIST実験の結果を受けた**次段階の実験方針**である。

## 7.1 CIFAR-10へ進む目的

Fashion-MNISTでは、

- Linear SVD
- Conv SVD
- 複数層同時圧縮
- Fine-tuning
- Parameters / MACs / accuracyの評価

まで一通り確認できた。

次は、より情報量の多いカラー画像を扱うCIFAR-10へ進み、今回得た傾向が別データセット・別CNNでも再現するかを確認する。

検証したい主な問いは、

```text
1. Fashion-MNISTより難しい画像分類でもSVD圧縮後の精度を維持できるか
2. Conv層が増えたとき、どの層を圧縮するのが有効か
3. ParametersとMACsの支配層はどう変わるか
4. Conv SVDのrank選択は層ごとにどの程度変わるか
5. Fine-tuningによる回復量はFashion-MNISTと同じ傾向か
```

である。

---

## 7.2 CIFAR-10で最初にやること

いきなりSVDせず、まずBaselineを確定する。

```text
01 CIFAR-10 CNN Baseline
↓
02 層ごとのParameters / MACsを集計
↓
03 圧縮候補層を決定
↓
04 Linear SVD / Conv SVDを個別に実施
↓
05 Fine-tuning
↓
06 複数層同時圧縮
↓
07 Fashion-MNISTとの比較
```

特に、Baselineが安定していない状態で圧縮実験を始めない。

---

## 7.3 Fashion-MNISTから引き継ぐ実験ルール

CIFAR-10でも次を維持する。

```text
Train
Early-Stopping Validation
Rank-Selection Validation
Test
```

を分離する。

また、

- Testは最終確認まで使わない
- rank候補間でseed / DataLoader条件を揃える
- Fine-tuning後もValidationでモデル選択する
- Parametersと全モデルMACsを必ず併記する
- latencyは同一benchmark条件で測る
- accuracyだけでなくlossも残す

を継続する。

---

## 7.4 CIFAR-10で追加したい改善

Fashion-MNISTでは単一seed中心だったため、CIFAR-10では可能なら複数seedを導入する。

```text
seed 0
seed 1
seed 2
...
```

として、

```text
Test accuracy mean ± std
Test loss mean ± std
```

まで出すと、小さな差を過度に解釈しにくくなる。

また、CNNの同時圧縮では、Fashion-MNISTの05は個別に選んだrankを組み合わせただけだった。

CIFAR-10では計算量が許せば、

```text
conv rank × conv rank × linear rank
```

の全探索ではなくても、候補を絞った多層rank探索を行い、Pareto frontierを作るとよい。

---

## 7.5 CIFAR-10での仮説

> [!note]
> 以下はFashion-MNISTの結果から立てる**次実験の仮説**であり、まだ実測結果ではない。

CNNがConv中心になるほど、

```text
Parametersだけを見る圧縮設計
```

よりも、

```text
Parameters + MACs
```

の両方を見る重要性がさらに高くなると予想する。

また、画像特徴抽出を担うConv層の低rank化では、層ごとに圧縮耐性が異なる可能性がある。そのため、全Convへ同じrank比率を一律適用するより、層ごとにrankを選択する方が有効かを検証する。

Fine-tuningについても、Fashion-MNISTと同様に、SVD直後より大きく性能を回復できるかを確認する。

---

# 8. 今後のロードマップ

```text
MNIST / MLP
    ↓
Fashion-MNIST / MLP
    ↓
Fashion-MNIST / CNN
    ↓
MLP vs CNN 比較・総括   ← 現在
    ↓
CIFAR-10 / CNN Baseline
    ↓
CIFAR-10 / Linear SVD + Conv SVD
    ↓
複数Conv・複数層のrank最適化
    ↓
SVD圧縮baseline確立
    ↓
Tensor Train / MPO
    ↓
DMRG-like最適化との比較
```

SVDは、今後より高度なTensor Network系圧縮を比較するための**baseline手法**として残す。

比較時は、

```text
同程度のparameter数でaccuracyを比較
```

または、

```text
同程度のaccuracyで必要parameter数 / MACsを比較
```

という軸が有効である。

---

# 9. 最終結論

Fashion-MNISTまでの実験で、SVDによるニューラルネットワーク圧縮について、次を実験として確認できた。

```text
Linear SVD
→ weight行列を直接低rank化
→ MLPではParametersとMACsを同時に大きく削減

Conv SVD
→ filter群を行列化して低rank化
→ 3×3 Conv + 1×1 Convへ置換
→ CNNではMACs削減に大きく寄与

Fine-tuning
→ SVDによる近似誤差を分類taskへ再適応
→ 強い圧縮でも性能を大きく回復できる場合がある

評価
→ Parametersだけでは不十分
→ MACs / latency / loss / accuracyを分けて見る
```

MLPでは約93%のparameter/MAC削減と引き換えにTest accuracyが1.57ポイント低下した。一方CNNでは、LinearとConvを役割別に圧縮することで、Parametersを78.66%、MACsを47.25%削減しつつTest accuracyをBaselineと同程度に維持した。

この段階で、**単純なLinear SVDだけでなく、CNNのConv層を含めた低rank圧縮の基本実験まで完了した**。

次のCIFAR-10では、より複雑な画像分類でも同じ設計・評価方法が通用するかを検証する。

## 関連ノート

- [[20_FashionMNIST/04_Fashion-MNISTでの実験結果]]
- [[20_FashionMNIST/11_CNNでの実験結果]]
- [[00_基礎理論/03_SVDによる低ランク近似]]
- [[00_基礎理論/04_Linear層を2層へ置き換える]]
- [[00_基礎理論/11_理論計算量とベンチマーク]]
- [[00_基礎理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/17_Conv2dの低ランク2層置換]]
