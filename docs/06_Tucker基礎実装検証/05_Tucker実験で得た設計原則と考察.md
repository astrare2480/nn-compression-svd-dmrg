---
title: Tucker実験で得た設計原則と考察
tags: [Tucker, HOOI, HOSVD, 実験設計, NN圧縮]
---

# Tucker実験で得た設計原則と考察

数式の完全な途中導出：[[00_基礎理論/01_数学基礎/02_テンソル代数/20_Tucker_HOSVD_HOOI数式の導出]]。

## 1. Tucker化は自動的に圧縮ではない

$$
N_{T2}
=C_{in}R_{in}+R_{out}R_{in}K_hK_w+C_{out}R_{out}.
$$

今回 `(64,32)` なら、

$$
32\times32+9\times64\times32+64\times64
=23552
$$

元は

$$
64\times32\times9=18432
$$

だから、

$$
23552-18432=5120
$$

増える。

分解形式を変えることと圧縮に成功することは別。

## 2. rank_out / rank_inは独立

$(R_{out},R_{in})$ は2つの設計変数。

例：

```text
(16,32) val = 0.5062
(32,16) val = 0.6280
```

積や総rankだけではtask影響を説明できない。

## 3. weight errorとaccuracyは別の目的

HOOI：

$$
0.449042-0.442504=0.006538
$$

だけweight errorを改善。

しかしvalidationは

$$
0.6202-0.6280=-0.0078
$$

だった。

したがって、

$$
\boxed{
\text{weight approximation quality}
\ne
\text{task performance}
}
$$

である。

## 4. HOSVDは弱い方法ではない

今回HOSVDは、HOOIより分解時間が短く、圧縮直後accuracyも高かった。大量rank探索ではHOSVD、固定rankのweight近似精密化では必要に応じてHOOI、という役割分担が自然。

## 5. HOOIの価値

同rank・同params・同MACsのまま、factor方向だけを協調更新する。

```text
HOSVD: 0.449042
HOOI : 0.442504
TensorLy: 0.442504
```

自作HOOIとTensorLyが一致したことが、実装検証として重要。

## 6. fine-tuningは別最適化

HOSVD post error増加：

$$
0.483660-0.449042=0.034618.
$$

HOOI post error増加：

$$
0.481255-0.442504=0.038751.
$$

それでもpost testは0.7508 / 0.7555まで回復した。

```text
HOSVD / HOOI
→ weight-space initialization

Fine-tuning
→ task-space optimization
```

という分離で理解する。

## 7. single seed

05 test差：

$$
0.7555-0.7508=0.0047
$$

= 0.47 percentage point。

1 seedなので「HOOIが統計的に優れる」とは言わない。

## 8. TensorLy timing

```text
self HOOI 41.65 ms
TensorLy  83.83 ms
```

両方6 iteration。TensorLyが余計に反復したからではない。対象weightは

$$
64\times32\times3\times3=18432
$$

要素と小さく、固定overheadが相対的に見えやすい。単発の速度比を一般化しない。

## 9. src化で固定した責務

```text
operations.py → Tensor基本演算
tucker.py     → HOSVD / reconstruction
hooi.py       → 汎用HOOI
conv_tucker.py→ Conv2d Tucker-2
```

partial HOOIは `ranks` のkeyを更新対象modeにすることで表現する。

## 10. 次のTT/MPSへ持ち越す評価軸

```text
weight relative error
parameters
MACs
圧縮直後accuracy
decomposition time
必要ならfine-tuning後accuracy
```

今回得た最大の注意点は、Frobenius weight errorを下げることとtask accuracyを上げることを同一視しないこと。
