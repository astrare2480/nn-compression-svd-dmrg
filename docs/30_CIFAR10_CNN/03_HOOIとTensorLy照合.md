---
title: HOOIとTensorLy照合
tags: [HOOI, HOSVD, TensorLy, Tucker, CIFAR10]
---

# HOOIとTensorLy照合

## 1. 条件

対象 `04_hooi_tucker2.ipynb`。

```text
conv2.weight = (64,32,3,3)
rank = (32,16)
```

比較：自作HOSVD / 自作HOOI / TensorLy `partial_tucker(init="svd")`。

## 2. 結果

| method | weight error | time | iterations | val acc | test acc | params | Conv2 MAC削減 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HOSVD | 0.449042 | 17.60 ms | 0 | 0.6280 | 0.6378 | 117,578 | 61.11% |
| self HOOI | 0.442504 | 41.65 ms | 6 | 0.6202 | 0.6254 | 117,578 | 61.11% |
| TensorLy | 0.442504 | 83.83 ms | 6 | 0.6202 | 0.6254 | 117,578 | 61.11% |

HOOIの絶対error改善量：

$$
\begin{aligned}
\Delta e
&=0.449042-0.442504\\
&=0.006538.
\end{aligned}
$$

HOSVDを分母にした相対改善：

$$
\begin{aligned}
\frac{0.006538}{0.449042}
&\approx0.01456\\
&\approx1.46\%.
\end{aligned}
$$

一方validationは、

$$
0.6202-0.6280=-0.0078
$$

で、HOOIの方が0.78 percentage point低い。

したがって、weight errorの1.46%相対改善が分類accuracy改善へ直接対応しなかった。

## 3. HOOI error history

| sweep | error | 前回からの改善 |
| ---: | ---: | ---: |
| 1 | 0.442817748 | - |
| 2 | 0.442547381 | 0.000270367 |
| 3 | 0.442511767 | 0.000035614 |
| 4 | 0.442505330 | 0.000006437 |
| 5 | 0.442503989 | 0.000001341 |
| 6 | 0.442503690 | 0.000000298 |

例えば1→2 sweepは

$$
0.442817748-0.442547381
=0.000270367.
$$

5→6 sweepは

$$
0.442503989-0.442503690
=2.99\times10^{-7}.
$$

改善量が急速に小さくなり、少数sweepでほぼ収束した。

## 4. TensorLyとの一致

最終weight errorは

$$
0.442504-0.442504\approx0
$$

validation/testも同じ。

TensorLy内部historyの値と外部再構成値に微小差があっても、比較では返されたfactor/coreから同じ自作relative-error式で再計算した値をcanonicalにする。

## 5. timing

```text
self HOOI = 41.65 ms
TensorLy  = 83.83 ms
```

単純比は約2倍だが、両者とも6 iteration。したがって反復回数差では説明できない。

今回のweight要素数は

$$
64\times32\times3\times3=18432
$$

と小さい。汎用backend/API、Python処理、GPU warm-up/同期等の固定overheadが相対的に見えた可能性があり、速度比を一般化しない。

## 6. 結論

自作HOOIは、同rankのHOSVDからweight Frobenius errorを下げ、TensorLyと同じ最終外部再計算errorへ到達した。一方、圧縮直後task accuracyは上がらなかった。
