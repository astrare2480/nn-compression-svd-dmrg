---
title: HOSVD・HOOI Fine-tuning比較
tags:
  - HOSVD
  - HOOI
  - Fine-tuning
  - Tucker
  - CIFAR10
---

# HOSVD・HOOI Fine-tuning比較

## 1. 実験目的

対象：

```text
notebooks/20_tucker/10_cifar10_cnn/05_hosvd_vs_hooi_finetuning.ipynb
```

04で、HOOIはHOSVDよりweight errorを下げたが圧縮直後accuracyは改善しなかった。

そこで、balanced rank `(32,16)` を固定し、HOSVD初期化とHOOI初期化を同じfine-tuning条件で比較した。

固定した主条件：

```text
seed = 0
optimizer = Adam
lr = 3e-4
max epochs = 30
patience = 3
min_delta = 1e-4
batch size = 256
```

trainingのshuffle / augmentation乱数系列もmethod間で揃える設計とした。

---

## 2. baseline

```text
comparison validation accuracy = 0.7344
test accuracy                  = 0.7327
parameters                     = 128,842
```

圧縮後は両手法とも同じrank・同じ3層構造なので

```text
parameters = 117,578
```

で同一。

---

## 3. 結果

正式CSV：`results/20_tucker/10_cifar10_cnn/05_hosvd_vs_hooi_finetuning/hosvd_vs_hooi_finetuning.csv`

| 指標 | HOSVD | HOOI | HOOI - HOSVD |
| --- | ---: | ---: | ---: |
| pre weight error | 0.449042 | 0.442504 | -0.006538 |
| pre validation acc | 0.6280 | 0.6202 | -0.0078 |
| pre test acc | 0.6378 | 0.6257 | -0.0121 |
| post validation acc | 0.7528 | 0.7528 | 0.0000 |
| post test acc | 0.7508 | 0.7555 | +0.0047 |
| validation recovery | +0.1248 | +0.1326 | +0.0078 |
| test recovery | +0.1130 | +0.1298 | +0.0168 |
| post weight error | 0.483660 | 0.481255 | -0.002405 |
| best epoch | 17 | 17 | - |
| epochs ran | 20 | 20 | - |
| best val loss | 0.717830 | 0.714387 | -0.003443 |

---

## 4. final accuracy

validationは完全に同じ。

$$
0.7528-0.7528=0
$$

testではHOOI初期化が

$$
0.7555-0.7508=0.0047
$$

だけ高かった。

ただしseed 0の1runのみなので、0.47 percentage pointの差を再現性のある優位性とは結論しない。

両手法ともbaseline test 0.7327を上回っているが、これは分解だけではなく追加学習を含む。

---

## 5. 回復量

HOOIは圧縮直後accuracyがHOSVDより低かったため、回復量は大きくなる。

```text
HOSVD test: 0.6378 → 0.7508  (+0.1130)
HOOI  test: 0.6257 → 0.7555  (+0.1298)
```

回復量の差だけでHOOIを優位と判断すると、開始点の低さも含んでしまう。

最終endpointと学習曲線を合わせて見る。

---

## 6. convergence speed

両方とも

```text
best epoch = 17
epochs ran = 20
```

だった。

少なくともこのseedでは、HOOI初期化による明確な収束速度向上は観察されなかった。

---

## 7. fine-tuning後weight errorが増えた

非常に重要な結果。

```text
HOSVD: 0.449042 → 0.483660
HOOI : 0.442504 → 0.481255
```

fine-tuning後は元のbaseline `conv2.weight` から**遠ざかった**にもかかわらず、accuracyは大きく上がった。

これは、fine-tuningの目的が

$$
\min\lVert W-\hat W\rVert_F
$$

ではなく、task lossを小さくすることだからである。

$$
\boxed{
\text{weight similarity to original}
\neq
\text{task optimality under low-rank constraint}
}
$$

を実測で確認した例になる。

---

## 8. 04とのpre testの微小差

04のHOOI pre testは

```text
0.6254
```

05のCSVでは

```text
0.6257
```

と0.0003差がある。

このノートでは、04の分解比較は04 CSV、05のfine-tuning比較は05 CSVをcanonicalとし、runをまたいでpre testの絶対値を混ぜない。

原因はこの資料だけでは断定しない。

---

## 9. 03のfine-tuning結果との関係

03のbalanced `(32,16)` HOSVD fine-tuningではtest 0.7682だったのに対し、05の同rank HOSVDは0.7508。

これも別Notebook / 別実験手順のrunなので、絶対値を横断比較しない。

05の目的は**同じ05内でHOSVD vs HOOI初期化の条件を揃えること**である。

---

## 10. 結論

このseedでは、

```text
HOOI
→ 初期weight errorは小さい
→ 圧縮直後accuracyは低い
→ fine-tuning後validationはHOSVDと同じ
→ fine-tuning後testは+0.0047
→ best epochは同じ
```

となった。

したがって、

> HOOIは同rankでweight近似を精密化でき、fine-tuning後もHOSVDと少なくとも同等の性能へ到達した。ただし、HOOI初期化が最終accuracyで優位であると一般化するには複数seedが必要。

とまとめる。
