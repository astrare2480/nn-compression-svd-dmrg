---
title: HOOIとTensorLy照合
tags:
  - HOOI
  - HOSVD
  - TensorLy
  - Tucker
  - CIFAR10
---

# HOOIとTensorLy照合

## 1. 実験目的

対象：

```text
notebooks/20_tucker/10_cifar10_cnn/04_hooi_tucker2.ipynb
```

同じtrained CNNの `conv2.weight=(64,32,3,3)`、balanced rank `(32,16)` で、

```text
自作HOSVD
自作HOOI
TensorLy partial_tucker(init="svd")
```

を比較する。

主目的は、

> 自作HOOIが同rankのHOSVD初期解から再構成誤差を改善し、TensorLyと整合するか。

である。

---

## 2. 分解直後の結果

正式CSV：`results/20_tucker/10_cifar10_cnn/04_hooi_tucker2/hooi_comparison.csv`

| method | weight error | time | iterations | val acc | test acc | params | Conv2 MAC削減 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HOSVD | 0.449042 | 17.60 ms | 0 | 0.6280 | 0.6378 | 117,578 | 61.11% |
| self HOOI | 0.442504 | 41.65 ms | 6 | 0.6202 | 0.6254 | 117,578 | 61.11% |
| TensorLy | 0.442504 | 83.83 ms | 6 | 0.6202 | 0.6254 | 117,578 | 61.11% |

HOOIはHOSVDから

$$
0.449042-0.442504
=
0.006538
$$

だけrelative errorを低下させた。

自作HOOIとTensorLyは、最終weight error、iteration数、validation/test accuracyが一致した。

この一致は、自作partial Tucker-2 HOOIの実装妥当性を強く支持する。

---

## 3. HOOI error history

正式CSV：`hooi_error_history.csv`

| sweep | relative error | 前回からの改善 |
| ---: | ---: | ---: |
| 1 | 0.442817748 | - |
| 2 | 0.442547381 | $2.7037\times10^{-4}$ |
| 3 | 0.442511767 | $3.5614\times10^{-5}$ |
| 4 | 0.442505330 | $6.4373\times10^{-6}$ |
| 5 | 0.442503989 | $1.3411\times10^{-6}$ |
| 6 | 0.442503690 | $2.9802\times10^{-7}$ |

改善量は急速に小さくなり、少数sweepでほぼ収束した。

---

## 4. TensorLy内部errorとの差

TensorLyがiteration内部で報告する誤差と、最終factorから自作関数で再構成して測った誤差には微小差があった。

比較では内部報告値を混ぜず、

$$
\frac{\lVert W-\hat W\rVert_F}{\lVert W\rVert_F}
$$

を同じ関数で再計算した最終 `0.442504` を主結果とする。

これはライブラリ間で評価式・計算順序・丸めが違う可能性を排除するためである。

---

## 5. HOOIでaccuracyが上がらなかった

HOOIはweight errorを下げたが、圧縮直後validationは

```text
HOSVD = 0.6280
HOOI  = 0.6202
```

だった。

したがって、この実験は

$$
\boxed{
\text{lower weight Frobenius error}
\not\Rightarrow
\text{higher classification accuracy}
}
$$

を具体的に示した。

HOOIが失敗したのではなく、HOOIは**指定rank内のweight再構成**を精密化しており、task accuracyを直接最適化していない。

---

## 6. なぜTensorLyの方が時間がかかったか

今回の計測では

```text
self HOOI = 41.65 ms
TensorLy  = 83.83 ms
```

でTensorLyの方が長かった。

しかし両者とも6 iterationなので、TensorLyが余計に多数反復したことが理由ではない。

今回の `64×32×3×3` weightは18,432要素と小さいため、線形代数本体に対して

- 汎用backend抽象化
- mode/rankの一般処理
- factor管理
- validation
- Python/API呼び出し

などの固定overheadが相対的に目立った可能性がある。

また数十ms規模のGPU計測はwarm-up・同期・初回kernel起動などの影響を受けやすい。

したがって、

> この環境・この小規模weight・この設定では自作HOOIの計測時間が短かった。

と記録し、

> 自作HOOIはTensorLyより常に約2倍速い。

とは一般化しない。

---

## 7. HOSVDとHOOIの実用上の位置づけ

今回の結果では、

```text
HOSVD
→ 速い
→ 圧縮直後accuracyもHOOIより高い

HOOI
→ weight errorをさらに下げる
→ ただし時間が増える
→ accuracy優位は出なかった
```

したがって大量rank探索ではHOSVDが合理的で、HOOIは最終候補に対する精密化として使うのが自然。

次の問いは、HOOIの小さいweight errorが**同条件fine-tuning後の回復**に影響するか、である。

→ [[06_Tucker基礎実装検証/04_HOSVD_HOOI_FineTuning比較]]
