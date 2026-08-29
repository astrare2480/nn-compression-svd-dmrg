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

さらにsrc review後は、rankを単なる「数値」ではなくpublic API contractとして扱う。boolをrankとして受けず、mode-n unfolding上で実現可能な最大rankを超える値は明示的に拒否する。不可能なrankを黙ってclipしたり、MACsだけ計算して有効な候補のように見せたりしない。

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

現行srcではHOOIのfeasibilityを反復開始前に検証し、`max_iter=0` でも不可能なrankを見逃さない。また `abs_tol` / `rel_tol` は有限かつ0以上、`max_iter` はboolではない0以上の整数としてcontractを固定した。

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

また、理論MACsと実測latencyは別指標として扱う。演算数が減っても、層分割・kernel launch・メモリアクセス等によりGPU latencyが短くなるとは限らない。

## 9. src化で固定した責務

現行srcではアルゴリズムとvalidationを分離した。

```text
tensor/operations.py
→ Tensor基本演算

tensor/validation.py
→ Tensor shape / mode contract

compression/tucker.py
→ HOSVD / reconstruction / Tucker要素数

compression/tucker_validation.py
→ Tucker rank / dtype / tolerance / max_iter contract

compression/hooi.py
→ 汎用HOOI

compression/conv_tucker.py
→ Conv2d Tucker-2

metrics/tensor_approximation.py
→ relative Frobenius error

metrics/model_comparison.py
→ task性能 / agreement / logits / latency

training/loops.py
→ train / evaluate と状態管理
```

partial HOOIは `ranks` のkeyを更新対象modeにすることで表現する。

## 10. 実数Tensorを仕様として固定する

HOSVD/HOOIのfactor射影は現行実装では `U.T` を使う。実数では正しいが、複素数では共役転置 `U.mH` が必要になる。

`torch.linalg.svd` 自体は複素Tensorを処理できるため、入口で拒否しないと「計算は通るが再構成が誤る」silent failureになりうる。このため現行public APIは複素dtypeを `TypeError` で拒否し、**実数Tensor限定**を仕様とした。

これは複素対応を否定するものではなく、必要になった時点で中心数式を共役転置へ拡張し、別途テストを追加する。

## 11. Autograd境界はAPIの役割で分ける

```text
tucker2_hooi(weight, ...)
→ Tensor-level decomposition
→ 入力をdetachしない

build_tucker2_conv(conv, ...)
→ Module construction / initialization
→ 元weightをdetachして新しいParameterへcopy
```

低レベル分解関数が勝手に計算グラフを切らない一方、module置換では元modelと新modelがParameterを共有しない。この境界を明示したことで、fine-tuning前提の圧縮moduleを安全に扱える。

## 12. 評価関数はmodel状態を壊さない

評価・比較処理で `model.eval()` を使う場合、rootの `model.training` だけを保存して `model.train(state)` で戻すと、frozen BatchNormなどsubmodule固有のtrain/eval状態を上書きしてしまう。

現行srcではrootと全submoduleの `.training` を個別保存し、正常終了時・例外時の両方で完全復元する。

これは圧縮手法固有ではなく、今後TT/MPS/DMRGをNNへ適用する際にも守る共通評価contractとする。

## 13. DataLoaderを暗黙に仮定しない

`evaluate()` / agreement / logits RMSE / `train_one_epoch()` は、`len(loader)` で空判定せず実際の走査結果で判定する。このため `IterableDataset` のような `len()` を持たないloaderも扱える。

一方、`collect_compression_metrics()` は同じloaderを複数の指標で再走査するため、現状は**再走査可能なloader**を前提とする。one-shot iterable対応は後続課題。

## 14. validation helperの重複は依存方向とのトレードオフ

整数rank等の小さなvalidation helperは、現状 `compression.svd`、`metrics.macs`、`metrics.model_comparison` などに類似実装がある。

これは単純な整理漏れだけではなく、`compression.hooi` が `metrics` を利用しているため、`metrics` から `compression` のhelperを再利用すると循環importを作る問題がある。

同様に `tensor/validation.py` と `compression/tucker_validation.py` にも類似helperが残る。

現段階では大規模な共通validation module新設より、依存方向を壊さず局所重複を許す方針とした。TT/MPS実装で同種contractがさらに増えた時点で、共通層の新設を再検討する。

## 15. 次のTT/MPSへ持ち越す評価軸

```text
weight relative error
parameters
MACs
圧縮直後accuracy
decomposition time
必要ならfine-tuning後accuracy
```

さらにsrc reviewで固定した次のcontractも持ち越す。

```text
shape / rankを入口で検証する
不可能なrankを黙ってclipしない
Tensor-level APIとModule constructionのautograd境界を分ける
評価前後でmodel/submodule状態を保存する
empty / Iterable loaderを意識する
理論MACsと実測latencyを分ける
```

今回得た最大の注意点は、Frobenius weight errorを下げることとtask accuracyを上げることを同一視しないこと。
