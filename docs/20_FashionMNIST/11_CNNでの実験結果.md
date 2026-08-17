---
title: CNNでの実験結果
aliases:
  - Fashion-MNIST CNN SVD圧縮結果
  - CNN SVD最終結果
  - CNN圧縮比較
tags:
  - Fashion-MNIST
  - CNN
  - SVD
  - NN圧縮
  - 実験結果
---

# CNNでの実験結果

## 最終比較

圧縮前後の比較には、02〜05で共通に使ったBaselineを使用する。

| Model | rank | Parameters | Param reduction | Total MACs | Total MAC reduction | Test loss | Test acc | ΔTest acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | — | 421,642 | — | 4,241,152 | — | **0.234131** | 91.75% | — |
| fc1 SVD + FT | fc1=24 | 98,570 | **76.62%** | 3,918,080 | **7.62%** | 0.238414 | **91.87%** | +0.12pt |
| conv2 SVD + FT | conv2=28 | 413,066 | **2.03%** | 2,560,256 | **39.63%** | 0.236488 | **91.97%** | +0.22pt |
| conv2 + fc1 SVD + FT | conv2=28, fc1=24 | **89,994** | **78.66%** | **2,237,184** | **47.25%** | 0.240347 | **91.80%** | +0.05pt |

> [!important]
> `Total MACs` と `Total MAC reduction` は単独圧縮も同じCNN全体を分母にして比較するために整理した値。02/03のNotebook内MACsはLinear部分、04はconv2単体、05はCNN全体を計算しているため、Notebookの `compute_reduction` をそのまま横並びにはしない。

最終的に、**conv2 rank28 + fc1 rank24** の同時圧縮で、Parametersを78.66%、理論MACsを47.25%削減しながら、Test accuracy 91.80%を得た。

---

## 01と02〜05のBaselineの違い

01：

```text
Train / Validation = 55,000 / 5,000
Test acc            = 90.90%
```

02〜05：

```text
Train / ES Val / Rank Val = 50,000 / 5,000 / 5,000
Test acc                  = 91.75%
```

01はCNN構造・shape・学習確認用。圧縮実験の比較には02〜05側のBaselineを使う。

---

## Parameters: fc1が支配的

Baselineのparameter内訳：

| Layer | Parameters | 全体に占める割合 |
|---|---:|---:|
| conv1 | 320 | 0.08% |
| conv2 | 18,496 | 4.39% |
| fc1 | 401,536 | **95.23%** |
| fc2 | 1,290 | 0.31% |
| **Total** | **421,642** | **100%** |

したがって、fc1 rank24だけで、

```text
421,642 → 98,570
76.62% reduction
```

まで減らせる。conv2だけでは、

```text
421,642 → 413,066
2.03% reduction
```

に留まる。

---

## MACs: conv2が支配的

Baseline全CNNのMACs：

| Layer | MACs |
|---|---:|
| conv1 | 225,792 |
| conv2 | **3,612,672** |
| fc1 | 401,408 |
| fc2 | 1,280 |
| **Total** | **4,241,152** |

parameter数とは逆に、計算量ではconv2が大きい。

### fc1 rank24

```text
Total MACs = 3,918,080
Reduction  = 7.62%
```

### conv2 rank28

```text
Total MACs = 2,560,256
Reduction  = 39.63%
```

### conv2 rank28 + fc1 rank24

```text
Total MACs = 2,237,184
Reduction  = 47.25%
```

この結果から、

```text
fc1 SVD   → parameter削減の主役
conv2 SVD → MACs削減の主役
```

という役割分担が確認できた。

より具体的には、Baseline全体に対して、

```text
fc1   : Parametersの95.23% / MACsの9.46%
conv2 : Parametersの 4.39% / MACsの85.18%
```

を占める。`conv2` はweight数自体は少なくても、そのweightを `14×14=196` 個の空間位置で再利用するためMACsが大きい。一方 `fc1` は巨大なweight行列を持つが、各weightを空間位置ごとに繰り返し使わない。

この実験から、**圧縮対象は「parameterを減らしたいか」「演算量を減らしたいか」で選ぶ必要がある**ことが分かった。

---

## rank選択は唯一の最適解ではない

今回の `fc1=24`, `conv2=28` は、単純にaccuracy最大のrankを採用したものではない。Pareto frontierとkneeを使い、圧縮率とValidation性能の折衷点として選んだ。

またfull-rank近くまでrankを増やすと近似誤差は減る一方、2層化のオーバーヘッドにより元モデルよりParameters/MACsが増える場合がある。実際、

```text
fc1 rank128 : CNN Parameters 438,026 > Baseline 421,642
conv2 rank64: CNN Parameters 425,738 > Baseline 421,642
```

となった。

したがって、

```text
SVD分解した = 圧縮できた
```

ではなく、**十分小さいrankを選ぶことで初めて圧縮になる**。

---

## Fine-tuningの効果

### fc1 rank24

```text
Validation acc : 91.72% → 92.26%
Validation loss: 0.233418 → 0.220129
```

### conv2 rank28

```text
Validation acc : 92.16% → 92.60%
Validation loss: 0.230316 → 0.209334
```

### conv2 rank28 + fc1 rank24

```text
Validation acc : 91.24% → 92.04%
Validation loss: 0.246371 → 0.222078
```

特に同時圧縮ではSVD直後にBaselineから `-1.30pt` まで落ちたaccuracyが、Fine-tuning後は `-0.50pt` まで回復した。

03ではbest epoch=1、04ではbest epoch=2、05ではbest epoch=1で、比較的短いFine-tuningで大きく回復している。結果からは、SVD因子が元の学習済みweightを近似した**良い初期値**になっていると考えられる。

ただし、この結果だけからSVDそのものにaccuracy改善効果や正則化効果があるとは結論しない。確認できたのは、**低rank近似で生じた性能低下を追加学習で補正しやすかった**ことである。

同時圧縮では上流の`conv2`で特徴量が変化し、その後段の`fc1`も同時に近似されるため、SVD directでは単独圧縮より誤差が重なった可能性がある。これはモデル構造と結果からの解釈であり、層ごとの誤差寄与を分離測定したわけではない。

---

## retained energyと分類性能

最終採用rank：

```text
fc1 rank24 retained energy   = 0.856742
conv2 rank28 retained energy = 0.863418
```

どちらも約86%だが、rank sweepではretained energyが増えてもValidation accuracyは完全な単調増加にはならなかった。

retained energyはSVDでどれだけ重み行列のエネルギーを保持したかを見る指標で、分類accuracy/lossとは別に評価する。

さらに、fc1とconv2でretained energyがどちらも約86%だからといって、2層が「同じ強さで圧縮されている」とは限らない。特異値分布や下流タスクへの感度が層ごとに異なるため、**retained energyの値を異なる層どうしの絶対的な圧縮強度として比較しない**。

---

## Test結果の解釈

3種類の圧縮モデルはすべてBaselineよりTest accuracyがわずかに高かった。

```text
fc1        +0.12pt
conv2      +0.22pt
combined   +0.05pt
```

ただし今回の実験はseed 0の1 runで、差も小さい。本ノートでは「SVDにより精度向上」とは結論せず、**圧縮後も精度を維持できた**と解釈する。

一方、Test lossは全圧縮モデルでBaselineよりわずかに増えた。

```text
Baseline  0.234131
fc1       0.238414
conv2     0.236488
combined  0.240347
```

accuracyだけでなくlossも残すことで、正解率が同程度でも予測分布が完全には同じでないことを確認できる。

今回のTestはrank選択やFine-tuning条件の選択には使わず、最終確認に限定している。この分離は、Test結果を見ながらrankを調整してしまうことを避けるために重要である。

一方、結果はseed 0の1 runだけなので、数十サンプル以下に相当する小さなaccuracy差を頑健な差とはみなさない。複数seedで平均・標準偏差を確認するまでは「維持」と表現するのが適切である。

また、Validationではcombined + FTがBaselineより `-0.50pt`、Testでは `+0.05pt` となっている。ValidationとTestで差の符号まで変わること自体も、今回の差が小さいことを示しており、単一split・単一seedの小差を過度に解釈しない。

---

## 同時圧縮rankの限界

05では、単独実験で選んだ `conv2=28` と `fc1=24` を固定して組み合わせた。

```text
conv2 rank × fc1 rank
```

を同時に探索したわけではないため、`(28, 24)` が同時圧縮としての全組み合わせ中の最適点であることは示していない。

この実験で示せたのは、**個別に選んだrankを組み合わせても、Parameters -78.66%、MACs -47.25%と精度維持を両立できた**ことまでである。

---

## MACsとlatency

05の同時圧縮：

```text
MACs      : 4,241,152 → 2,237,184 (-47.25%)
記録時間  : 0.384253 ms → 0.416497 ms
```

ただし、05ではbenchmark条件が揃っていない。

```text
Baseline   : warmup=20, repeats=2000
Compressed : warmup=5,  repeats=200
```

そのため、この2値から「圧縮後の方が遅い」とは結論しない。現在のlatency値は参考記録であり、**速度比較としては未確定**とする。

また、たとえ同一条件で測定しても、MACsは理論演算量であり実測latencyとは別物である。低rank化では元の1層が2層になるため、候補要因として、

- 演算呼び出し回数の増加
- 中間Tensorの生成・メモリアクセス
- backend/kernelの実装効率
- batch sizeやdevice
- 小規模モデルでの固定オーバーヘッド

などがある。これらは今回原因別に計測したものではないため、今後は同一benchmark条件で複数回測定する。

よって今後も、

- Parameters
- MACs
- 実測latency
- Accuracy / loss

を別の評価軸として残す。

---

## 今回の結論

```text
Fashion-MNIST CNN
conv2 rank = 28
fc1 rank   = 24
```

で、

```text
Parameters 421,642 → 89,994
           -78.66%

MACs       4,241,152 → 2,237,184
           -47.25%

Test acc   91.75% → 91.80%
```

を得た。

**LinearとConvを同時に低rank化することで、parameter数と理論計算量を同時に大きく減らしながら分類精度を維持できた。**

---

## この実験で言えること / まだ言えないこと

### 言えること

- このFashion-MNIST CNNでは、fc1低rank化がparameter削減、conv2低rank化がMACs削減に強く効いた。
- 個別に選択した `fc1=24`, `conv2=28` を同時適用すると、Parameters -78.66%、MACs -47.25%でもTest accuracyをBaselineと同程度に維持できた。
- SVD directの性能低下は短いFine-tuningでかなり回復した。
- retained energyだけでは最終分類性能を決められず、Validation loss/accuracyなどタスク指標も必要だった。

### まだ言えないこと

- `(conv2=28, fc1=24)` が同時圧縮の全rank組み合わせで最適か。
- 小さなTest accuracy差が複数seedでも再現するか。
- MACs -47.25%が実機latency短縮へつながるか。
- accuracyが近いモデル間で予測確率のcalibrationまで維持されているか。今回はcalibration指標を測定していない。
- 別のデータセット・別のCNN構造でも同じ圧縮率を保てるか。

## 今後の追加検証（提案）

1. 複数seedでBaseline/圧縮モデルを再実行し、Test accuracy/lossの平均と標準偏差を確認する。
2. `conv2 rank × fc1 rank` の2次元sweepを行い、Parameters・MACs・Validation lossのPareto frontierを作る。
3. latency benchmarkを同一warmup/repeats・同一batch sizeで複数回測定し、代表値とばらつきを残す。
4. parameter数だけでなく、必要なら保存checkpointサイズや実メモリ使用量も別途測定する。
5. より難しいデータセットや別CNN構造でも同じ手順を再現し、今回の傾向がモデル固有か一般化できるか確認する。
6. accuracyとlossの差をさらに調べる必要がある場合は、予測確率のcalibration指標も追加する。

## 関連

- [[20_FashionMNIST/07_CNN実験]]
- [[20_FashionMNIST/08_CNNのLinear SVD]]
- [[20_FashionMNIST/09_CNNのConv SVD]]
- [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
- [[20_FashionMNIST/12_CNN全RankSweepと学習履歴]]
