---
title: HOSVD・HOOI Fine-tuning比較
tags: [HOSVD, HOOI, Fine-tuning, Tucker, CIFAR10]
---

# HOSVD・HOOI Fine-tuning比較

## 1. 条件

対象 `05_hosvd_vs_hooi_finetuning.ipynb`。

```text
rank=(32,16)
seed=0
Adam
lr=3e-4
max_epoch=30
patience=3
min_delta=1e-4
batch=256
```

baseline：val 0.7344 / test 0.7327 / params 128,842。

圧縮後は両手法ともparams 117,578。

## 2. 結果

| 指標 | HOSVD | HOOI | HOOI-HOSVD |
| --- | ---: | ---: | ---: |
| pre weight error | 0.449042 | 0.442504 | -0.006538 |
| pre val acc | 0.6280 | 0.6202 | -0.0078 |
| pre test acc | 0.6378 | 0.6257 | -0.0121 |
| post val acc | 0.7528 | 0.7528 | 0.0000 |
| post test acc | 0.7508 | 0.7555 | +0.0047 |
| post weight error | 0.483660 | 0.481255 | -0.002405 |
| best epoch | 17 | 17 | 0 |
| epochs ran | 20 | 20 | 0 |

## 3. recoveryを途中式で計算

HOSVD validation：

$$
0.7528-0.6280=0.1248.
$$

HOOI validation：

$$
0.7528-0.6202=0.1326.
$$

HOSVD test：

$$
0.7508-0.6378=0.1130.
$$

HOOI test：

$$
0.7555-0.6257=0.1298.
$$

HOOIの回復量が大きいが、開始点が低いことも含むので、recovery差だけで優位性を判断しない。

## 4. final endpoint

validation差：

$$
0.7528-0.7528=0.
$$

test差：

$$
\begin{aligned}
0.7555-0.7508
&=0.0047\\
&=0.47\text{ percentage point}.
\end{aligned}
$$

seed 0の1runなので、この0.47 pointを再現性のある優位性とは断定しない。

baseline testとの差は、HOSVDで

$$
0.7508-0.7327=0.0181
$$

HOOIで

$$
0.7555-0.7327=0.0228.
$$

追加学習込みなので「分解だけでbaselineを上回った」とは言わない。

## 5. fine-tuning後に元weightから遠ざかった

HOSVD：

$$
\begin{aligned}
\Delta e_H
&=0.483660-0.449042\\
&=0.034618.
\end{aligned}
$$

HOOI：

$$
\begin{aligned}
\Delta e_O
&=0.481255-0.442504\\
&=0.038751.
\end{aligned}
$$

両方とも元baseline `conv2.weight` へのrelative errorが増えたのにaccuracyは回復した。

HOOIのweight-error優位は、preで

$$
0.449042-0.442504=0.006538
$$

postで

$$
0.483660-0.481255=0.002405
$$

まで縮小した。

したがって、fine-tuningは

$$
\min\|W-\hat W\|_F
$$

ではなくtask lossを下げる別の最適化であることが数値にも表れている。

## 6. convergence speed

両方ともbest epoch 17、20 epochで停止。少なくともこのrunではHOOI初期化による明確な収束epoch短縮は観察されなかった。

## 7. runを混ぜない

04 HOOI pre test `0.6254` と05 `0.6257` は別CSVのrun。03 balanced HOSVD test `0.7682` と05 HOSVD `0.7508` も別run。各Notebook内比較をcanonicalにする。

## 結論

HOOIは初期weight近似を改善したが、圧縮直後accuracyではHOSVDを下回った。同条件fine-tuning後はvalidation同値、testはHOOIが0.0047高かったが、single seedなので優位性は確定しない。
