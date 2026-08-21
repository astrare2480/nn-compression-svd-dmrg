---
title: CNNのConv SVD
aliases:
  - CNN conv2 SVD
  - Fashion-MNIST Conv圧縮
  - Conv2d rank selection
tags:
  - Fashion-MNIST
  - CNN
  - Conv2d
  - SVD
  - FineTuning
  - RankSelection
---

# CNNのConv SVD

## サマリー

Fashion-MNIST CNNの、

```text
conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
```

をSVD圧縮した。

正式結果は、

```text
notebooks/10_svd/30_fashion_mnist_cnn/04_cnn_conv_svd_corrected.ipynb
results/20_fashion_mnist/04_cnn_conv_svd_corrected/
```

を基準とする。

```text
conv2.weight: (64, 32, 3, 3)
        ↓ flatten(start_dim=1)
W_mat:       (64, 288)
        ↓ truncated SVD rank=r
Conv2d(32 → r, 3x3, bias=False)
Conv2d(r  → 64, 1x1, bias=True)
```

corrected実験では、direct rank sweepのknee近傍、

```text
Aggressive   r=20
Balanced     r=24
Conservative r=28
```

をFine-tuningし、最終的に **conv2 rank 28** を採用した。

```text
Parameters: 421,642 → 413,066  (-2.03%)
Test acc:   91.28%  → 91.75%   (+0.47 percentage point)
```

parameter削減は小さいが、Convは同じweightを多数の空間位置で使うため、MACs削減への寄与はparameter数以上に大きい。

ただしcorrected benchmarkでは、MACs削減がwall-clock latency短縮にはつながらなかった。

---

# 1. 対象とした重み

SVDするのは特徴マップ `(N, 32, H, W)` ではなく、学習済みの `conv2.weight`。

```text
(64, 32, 3, 3)
→ (64, 288)
```

行列化は、

$$
W_{mat}
\in
\mathbb{R}^{64 \times (32\cdot3\cdot3)}
=
\mathbb{R}^{64 \times 288}
$$

である。

詳細は、

- [[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]

を参照。

### conv1を主対象にしなかった理由

Fashion-MNISTの `conv1.weight` は、

```text
(32, 1, 3, 3)
→ 行列化すると (32, 9)
```

で最大rankは9。

圧縮余地が小さいため、この単層Conv実験では `conv2` を対象にした。

---

# 2. rankとparameter数

conv2の元parameter数は、

$$
64\times32\times3\times3+64
=
18{,}496
$$

低rank化後は、

$$
r(32\times3\times3)+64r+64
=
352r+64
$$

となる。

元よりparameterが減る条件は、

$$
352r+64 < 18{,}496
$$

より、

$$
r < 52.36\ldots
$$

である。

したがって、full-rankまで因子化すれば必ず圧縮になるわけではない。

```text
SVD factorization
≠
必ずparameter reduction
```

十分小さいrankへtruncationして初めて圧縮になる。

---

# 3. direct rank sweep

direct SVDでは、

- parameters
- validation loss / accuracy
- agreement
- logits RMSE
- retained energy
- MACs

などを評価した。

`parameters` と `validation_loss` のPareto frontierからkneeを求め、Fine-tuning候補を絞った。

corrected runでは、knee近傍として、

```text
20 / 24 / 28
```

を採用した。

ここで重要なのは、retained energyが大きいほどtask accuracyが必ず単調に改善するわけではないこと。

```text
retained energy
→ weight近似の指標

validation accuracy / loss
→ task性能の指標
```

として分ける。

---

# 4. Fine-tuning前後

corrected実験では、3候補を同じseed条件でFine-tuningした。

| Candidate | rank | Acc before | Acc after | Δacc | Loss before | Loss after |
|---|---:|---:|---:|---:|---:|---:|
| Aggressive | 20 | 0.9094 | 0.9252 | +0.0158 | 0.245499 | 0.212728 |
| Balanced | 24 | 0.9154 | **0.9254** | +0.0100 | 0.237839 | 0.210294 |
| Conservative | **28** | **0.9158** | 0.9250 | +0.0092 | **0.235605** | **0.209190** |

Fine-tuning後、accuracyだけを見るとrank24がわずかに高い。

一方、最終比較でvalidation lossが最小だったrank28を採用した。

```text
rank24
Val acc  = 0.9254
Val loss = 0.210294

rank28
Val acc  = 0.9250
Val loss = 0.209190
```

このため、rank28は「accuracy最大rank」ではなく、今回の最終選択規則の下で採用したcandidateである。

---

# 5. なぜFine-tuningで回復するか

truncated SVDは、

$$
\lVert W-W_r\rVert_F
$$

を小さくする低rank近似を与えるが、分類taskのCross Entropyを直接最適化しているわけではない。

したがって、

```text
学習済みweight
↓ SVD
低rank近似
↓ Fine-tuning
task lossに合わせて低rank制約内で再最適化
```

という流れになる。

今回3候補ともFine-tuning後にvalidation lossが改善し、accuracyも回復した。

ただし、Fine-tuningで必ずbaselineを超える保証はない。

---

# 6. 最終Test

| Model | conv2 rank | Parameters | Test loss | Test acc | ΔTest acc |
|---|---:|---:|---:|---:|---:|
| Baseline | — | 421,642 | 0.238263 | 0.9128 | — |
| conv2 SVD + FT | **28** | **413,066** | **0.232226** | **0.9175** | **+0.47pt** |

Parameter reductionは、

```text
2.03%
```

である。

Test lossも今回のrunでは小さくなった。

ただし、この実験はsingle seedなので、

> SVD圧縮によりaccuracyが改善した

とは結論しない。

中心となる解釈は、

> **Convを低rank化し、Fine-tuning後もtask性能をほぼ維持できた**

である。

---

# 7. parameter削減とMACs削減が違う理由

`conv2` のweightはbiasを除くと、

$$
64\times32\times3\times3
=
18{,}432
$$

個。

一方、同じweightを特徴マップ上の多数の位置で繰り返し使う。

そのため、

```text
weight数はモデル全体に対して小さい
しかしforward計算量への寄与は大きい
```

ということが起こる。

Fashion-MNIST CNNでは、

```text
fc1
→ parameter削減に効く

conv2
→ MACs削減に効く
```

という役割分担が見えた。

---

# 8. corrected latency benchmark

旧04では、baselineとcompressedで、

```text
warmup
repeats
```

が揃っていなかった。

そのため旧latency値を正式な速度比較には使わない。

corrected版では、

```text
batch size = 256
same input_batch
warmup = 20
repeats = 2000
```

へ統一した。

結果は、

```text
Baseline   ≈ 0.367 ms/batch
Compressed ≈ 0.370 ms/batch
```

で、ほぼ同等〜わずかにcompressedが遅かった。

ここから言えるのは、

```text
MACsを減らした
≠
GPU実測latencyが必ず短くなる
```

ということ。

低rank化では、元1層が2層へ分かれるため、実測速度は、

- kernel launch
- 中間Tensor
- memory access
- 演算shape
- backend最適化

などにも依存する。

ただし各要因を個別に計測したわけではないので、

> Conv SVDは一般に遅い

とは結論しない。

---

# 9. historical 04との違い

旧04のrank sweep / Fine-tuning結果は学習履歴として残すが、正式な結果引用ではcorrected版を優先する。

特に変更点は、

```text
旧Fine-tuning候補
24 / 28 / 32

corrected Fine-tuning候補
20 / 24 / 28
```

および、latency benchmarkの公平化である。

SVDの数式そのものを変更したわけではない。

詳しくは [[05_SVD基礎実装検証/03_SVD実験で修正した問題と設計原則]] を参照。

---

# 10. 結論

- `conv2.weight` を `(64, 288)` へ行列化してSVDした。
- direct sweepのknee近傍 `20 / 24 / 28` をFine-tuningした。
- 最終rankは `28`。
- Parametersは約2.03%削減した。
- Test accuracyは `91.28% → 91.75%` で、single seedとしては精度維持と解釈する。
- Convではparameter削減率以上にMACs削減へ効く場合がある。
- corrected benchmarkではMACs削減がlatency短縮には直結しなかった。

---

# 関連

- [[00_基礎理論/03_モデル圧縮理論/16_Conv2d重みの行列化とSVD]]
- [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]
- [[00_基礎理論/04_実験設計/19_再現性と乱数管理]]
- [[20_FashionMNIST/08_CNNのLinear SVD]]
- [[20_FashionMNIST/10_CNNのConvとLinear同時圧縮]]
- [[20_FashionMNIST/11_CNNでの実験結果]]
- [[30_CIFAR10_CNN/01_CIFAR10_SVD実験]]
